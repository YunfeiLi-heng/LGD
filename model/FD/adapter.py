import torch
from torch import nn
from timm import create_model
from torch.nn import functional as F

from model.layer import Fusion, MLP, PatchEmbed, VT_LN
from functools import partial
from model.vit_tiny_16_224 import vit_tiny_patch16_224, vit_small_patch16_224

class Mask_Decoder(nn.Module):
    def __init__(self, in_dim, mlp_dim=512, out_dim=256, mlp_num_layers=3, head_num=16):
        super().__init__()
        self.in_dim = in_dim
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.head_num = head_num
        dense_affine_func = partial(nn.Conv2d, kernel_size=1)
        self.query_mlp = MLP(in_dim, mlp_dim, out_dim, mlp_num_layers)  # L R L R L
        self.xray_mlp = MLP(in_dim, mlp_dim, out_dim, mlp_num_layers, affine_func=dense_affine_func)
        self.attn_mlp = MLP(in_dim, mlp_dim, out_dim * self.head_num, mlp_num_layers, affine_func=dense_affine_func)
        self.bias_scaling = nn.Linear(1, 1)

    def forward(self, query, x): #query:(16,128,192) x:(16,192,16,16)
        # query (N,QL,D) x (N D H W)
        query = self.query_mlp(query) #(16,128,128)
        xray = self.xray_mlp(x) #(16,128,16,16)
        attn = self.attn_mlp(x) #(16,2048,16,16)
        patch_x = x.reshape(x.shape[0], x.shape[1], -1) #(N D L)(16,192,256)
        patch_x = patch_x.permute(0, 2, 1) #(N L D)(16,256,192)
        xray_pred = torch.einsum('NQD,NDhw->NQhw', query, xray) #(16,128,16,16)
        n, _, h, w = x.shape
        d = 128
        attn = attn.reshape(n, self.head_num, d, h, w)  # (N Head*D,h,w)->(N Head D h w)(16,16,128,16,16)
        attn_bias = torch.einsum('NQD,NHDhw->NHQhw', query, attn) #(16,16,128,16,16)
        attn_bias = self.bias_scaling(attn_bias[..., None]).squeeze(-1) #(16,16,128,16,16)
        # attn_bias = torch.sigmoid(attn_bias)
        return xray_pred, attn_bias


class Adapter(nn.Module):
    def __init__(self, vit_name, num_quires, fusion_map, mlp_dim, mlp_out_dim, head_num, device):
        super().__init__()
        self.device = device
        # self.vit_model = create_model(vit_name,
        #                               pretrained=False,
        #                               fc_norm=False,
        #                               num_classes=2,
        #                               embed_layer=PatchEmbed,
        #                               ).to(device=self.device)

        self.vit_model = vit_tiny_patch16_224(pretrained=False, num_classes=2)

        if self.vit_model.cls_token is not None:
            self.vit_model.pos_embed = nn.Parameter(self.vit_model.pos_embed[:, 1:, ...])  # 去掉cls的位置
        del self.vit_model.cls_token
        self.vit_model.cls_token = None
        del self.vit_model.norm
        self.vit_model.norm = nn.Identity()
        self.num_quires = num_quires
        self.num_features = self.vit_model.num_features
        self.query_embed = nn.Parameter(torch.zeros(1, self.num_quires, self.num_features))  # (1,Q_L,D)
        self.query_pos_embed = nn.Parameter(torch.zeros(1, self.num_quires, self.num_features))
        self.fusion_map = fusion_map
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.normal_(self.query_pos_embed, std=0.02)
        self.mask_decoder = Mask_Decoder(in_dim=self.num_features, mlp_dim=mlp_dim, out_dim=mlp_out_dim,
                                         mlp_num_layers=3, head_num=head_num)
        self.ln_pre = VT_LN(self.num_features)
        self.patch_conv = nn.Conv2d(in_channels=3, out_channels=self.num_features, kernel_size=16, stride=16,
                                    bias=True)
        self.norm_middle = nn.LayerNorm(192)
        # self.cross_attn = nn.MultiheadAttention(192, 3)
        # self.scale = nn.Linear(1, 1)

        # self._register_hooks()
        # self.features = None
        # self.gradients = None

    def fuse(self, block_idx, x, clip_features, spatial_shape):
        if block_idx in self.fusion_map.keys():
            clip_layer = self.fusion_map[block_idx]
            adapter_layer = block_idx
            
            # clip_layer='layer'+str(clip_layer)
            clip_dim = clip_features[clip_layer].shape[2]  # clip features NLD, 1024

            fusion = Fusion(clip_dim, self.num_features).to(self.device)# 1024, 192
            L = spatial_shape[0] * spatial_shape[1] #256
            x = torch.cat(
                [
                    x[:, :-L, ...],  # query
                    # fuse vision token x(N,a_L,D) clip_f[i] (N,c_L,D)
                    fusion(x[:, -L:, ...], clip_features[clip_layer], spatial_shape)
                ],
                dim=1
            )
            return x

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.features = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        blk = self.vit_model.blocks[6].mlp.fc2
        blk.register_forward_hook(forward_hook)
        blk.register_full_backward_hook(backward_hook)


    def intra_contra(self, block_idx, x, patch_labels, image_labels, spatial_shape):
        loss = 0
        if block_idx == 4 or block_idx == 5 or block_idx == 6 :
            L = spatial_shape[0] * spatial_shape[1]  # 14 * 14 = 196

            embeddings = x[:, -L:, ...]  # (N 196 192) (NLD)
            embeddings = nn.functional.normalize(embeddings, dim=-1)
            fake_index = (image_labels == 1).nonzero(as_tuple=True)[0]
            fake_embeddings = embeddings[fake_index]  # f_N L D
            fake_nums = len(fake_index)

            ff_patch_labels = patch_labels[fake_index]  # f_N L


            fr_patch_labels = torch.logical_not(ff_patch_labels)
 
            f_fake_part = fake_embeddings * ff_patch_labels.unsqueeze(-1)  # f_N L D *  f_N L 1 -> f_N L D
            f_real_part = fake_embeddings * fr_patch_labels.unsqueeze(-1)  # f_N L D *  f_N L 1 -> f_N L D

            negative = torch.bmm(f_fake_part, f_real_part.permute(0, 2, 1)) / 0.5  # f_N L L
            positive1 = torch.bmm(f_real_part, f_real_part.permute(0, 2, 1)) / 0.5  # f_N L L
            positive2 = torch.bmm(f_fake_part, f_fake_part.permute(0, 2, 1)) / 0.5  # f_N L L

            l_neg = torch.sum(torch.exp(negative))
            l_pos1 = torch.sum(torch.exp(positive1))
            l_pos2 = torch.sum(torch.exp(positive2))

            loss_real_intra = -torch.log(l_pos1 / (l_neg + l_pos1))
            loss_fake_intra = -torch.log(l_pos2 / (l_neg + l_pos2))


            #loss = loss_real_intra + loss_fake_intra
            loss = loss_real_intra

        return loss

    def forward(self, data_dict, clip_features, step, attn_blk, feat_blk, inference):

        image = data_dict['image']
        x = self.patch_conv(image) #(b,192,16,16)
        x = x.reshape(x.shape[0], x.shape[1], -1) #（b,192,256)
        x = x.permute(0, 2, 1) #(b, 256, 192)(b, h*w, c)

        pos_embed = self.vit_model.pos_embed  # (N L D)(1, 196, 192)
        pos_embed = pos_embed.permute(0, 2, 1) #(NDL)
        pos_embed = F.interpolate(pos_embed.reshape(pos_embed.shape[0],pos_embed.shape[1], 14, 14),
                                  size=(16, 16),
                                  mode='bilinear',
                                  align_corners=False).reshape(pos_embed.shape[0],pos_embed.shape[1],256).permute(0,2,1) #NDL->NLD(1, 256, 192)
        pos_embed = torch.cat([self.query_pos_embed.expand(pos_embed.shape[0], -1, -1), pos_embed],
                              dim=1) #(1, 384, 192) 384=128+256
        v_L = x.shape[1]  # vision token L 196
        (h, w) = 16,16  # h w 16,16

        x = torch.cat([self.query_embed.expand(x.shape[0], -1, -1), x], dim=1)  # (N ,Q_L+L,D)(1, 128, 192)(b, 256, 192)(b, 384, 192)
        x = x + pos_embed
        x = self.ln_pre(x)
        outs = []
        out_layers = [8]
        loss_intra = 0
        attn_list=[]
    
        # self.fuse(0, x, clip_features, (h, w))
        for i, block in enumerate(self.vit_model.blocks, start=1):  # total 1-12 ,only use 1-8    
            if i == feat_blk:
                x_block = self.norm_middle(x.float())
            if i in attn_blk:
                attn_list.append(attn)

            x, attn = block(x)  # (N, Q_L+L, D)(16,384,192)
            self.fuse(i, x, clip_features, (h, w))
            if not inference:  #train
                loss_tmp_intra = self.intra_contra(i, x, data_dict['patch_label'], data_dict['label'], (h, w))
                loss_intra = loss_tmp_intra + loss_intra if loss_tmp_intra != 0 else loss_intra
            if i in out_layers:
                n, _, d = x.shape 
                outs.append(
                    {
                        'query': x[:, :-v_L, ...],
                        'x': x[:, -v_L:, ...]
                        .permute(0, 2, 1)
                        .reshape(n, d, h, w),
                    }
                )
            x = x + pos_embed
            if i == max(out_layers):
                break
        
        if (i+1) in attn_blk:
            attn_list.append(attn)
        if i == feat_blk - 1:
            x_block = x
        attn_block = torch.cat(attn_list, dim=1) #（16,80,385,385）在head维度拼接, 60代表的是5层*12个注意力头
        L, N, D = x_block.size()
        x_block = x_block[:, -256:, :].reshape(L, 16, 16, D) #(16,16,16,1024) 去除cls，并展开patch维度

      
        attn_biases = []
        xray_preds = []

        for feature in outs:
            xray_pred, attn_bias = self.mask_decoder(feature['query'], feature['x'])
            attn_biases.append(attn_bias)
            xray_preds.append(xray_pred)

        return attn_bias, loss_intra, x_block, attn_block, xray_preds
