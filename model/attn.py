from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from model.layer import MLP
import numpy as np
from trainer.metrics.utils import visualize_M_binary

# def adaptive_artifact_cutout(
#     vision_tokens,
#     index_map,
#     labels,
#     alpha=0.7,
#     top_ratio=0.15,
#     apply_prob=0.5,
# ):
#     """
#     Mask top-K patches ONLY inside valid (M > 0) region,
#     where K depends on valid region size.
#     """

#     device = vision_tokens.device

#     L, B, D = vision_tokens.shape
#     _, H, W = index_map.shape

#     # ---------------------------------------/--
#     # 1. flatten
#     # -----------------------------------------
#     score = index_map.reshape(B, -1)

#     # -----------------------------------------
#     # 2. fake-only mask
#     # -----------------------------------------
#     fake_mask = (labels == 1).float().unsqueeze(1).to(device)

#     # -----------------------------------------
#     # 3. valid region (M > 0)
#     # -----------------------------------------
#     valid_region = (score > 0).float()

#     # number of valid patches per sample
#     valid_num = valid_region.sum(dim=1)  # (B,)

#     # -----------------------------------------
#     # 4. compute K PER SAMPLE (key fix)
#     # -----------------------------------------
#     K = (valid_num * top_ratio).long()
#     K = torch.clamp(K, min=1)  # at least 1

#     # -----------------------------------------
#     # 5. score only inside valid region
#     # -----------------------------------------
#     masked_score = score * valid_region

#     # fallback for empty region
#     has_valid = valid_num > 0
#     masked_score = torch.where(
#         has_valid.unsqueeze(1),
#         masked_score,
#         score
#     )

#     # -----------------------------------------
#     # 6. top-K per sample (dynamic K)
#     # -----------------------------------------
#     mask_flat = torch.zeros_like(score)

#     for b in range(B):
#         _, topk_idx = torch.topk(masked_score[b], K[b].item())
#         mask_flat[b, topk_idx] = 1.0

#     # -----------------------------------------
#     # 7. apply fake-only constraint
#     # -----------------------------------------
#     mask_flat = mask_flat * fake_mask

#     mask_vi = mask_flat.view(B, H, W)

#     # -----------------------------------------
#     # 8. token mask
#     # -----------------------------------------
#     token_mask = mask_flat.permute(1, 0).unsqueeze(-1)

#     vision_tokens_aug = vision_tokens * (1 - alpha * token_mask)

#     # -----------------------------------------
#     # 9. stochastic apply
#     # -----------------------------------------
#     if torch.rand(1, device=device) < apply_prob:
#         return vision_tokens_aug.half(), mask_vi
#     else:
#         return vision_tokens, mask_vi
  

def random_center_cutout(
    vision_tokens,
    labels,
    center_ratio=0.1,
    mask_ratio=0.1,
    alpha=0.7,
    apply_prob=0.5,
):
    """
    Randomly mask patches inside the center square region.
    
    Args:
        vision_tokens:
            CLIP vision tokens, shape (L, B, D)
        
        labels:
            binary labels, 1 for fake and 0 for real, shape (B,)
        
        center_ratio:
            ratio of the center square region w.r.t. the whole patch map.
            e.g. 0.5 means using the central 50% x 50% area.
        
        mask_ratio:
            masking ratio inside the center square region.
        
        alpha:
            masking strength.
        
        apply_prob:
            probability of applying augmentation.
    """

    device = vision_tokens.device

    L, B, D = vision_tokens.shape


    # -------------------------------------------------
    # 1. infer patch spatial size
    # -------------------------------------------------
    H = W = int(L ** 0.5)

    assert H * W == L, \
        "Token number cannot be reshaped into square patch map"


    # -------------------------------------------------
    # 2. fake-only constraint
    # -------------------------------------------------
    fake_mask = (labels == 1).float().to(device)


    # -------------------------------------------------
    # 3. generate center square region
    # -------------------------------------------------
    center_mask = torch.zeros(
        (B, H, W),
        device=device
    )


    region_h = int(H * center_ratio)
    region_w = int(W * center_ratio)


    start_h = (H - region_h) // 2
    start_w = (W - region_w) // 2


    center_mask[
        :,
        start_h:start_h + region_h,
        start_w:start_w + region_w
    ] = 1.0


    center_mask_flat = center_mask.reshape(B, -1)


    # -------------------------------------------------
    # 4. randomly select patches
    # -------------------------------------------------
    mask_flat = torch.zeros_like(center_mask_flat)


    region_num = center_mask_flat.sum(dim=1)


    K = (region_num * mask_ratio).long()

    K = torch.clamp(K, min=1)


    for b in range(B):

        if fake_mask[b] == 0:
            continue

        candidate_idx = torch.where(
            center_mask_flat[b] > 0
        )[0]


        if len(candidate_idx) > 0:

            perm = torch.randperm(
                len(candidate_idx),
                device=device
            )

            selected = candidate_idx[
                perm[:K[b]]
            ]

            mask_flat[b, selected] = 1.0



    # -------------------------------------------------
    # 5. apply fake constraint
    # -------------------------------------------------
    mask_flat = mask_flat * fake_mask.unsqueeze(1)


    mask_map = mask_flat.reshape(
        B, H, W
    )


    # -------------------------------------------------
    # 6. token masking
    # -------------------------------------------------
    token_mask = (
        mask_flat
        .permute(1, 0)
        .unsqueeze(-1)
    )


    vision_tokens_aug = (
        vision_tokens *
        (1 - alpha * token_mask)
    )


    # -------------------------------------------------
    # 7. stochastic apply
    # -------------------------------------------------
    if torch.rand(1, device=device) < apply_prob:

        return (
            vision_tokens_aug.half(),
            mask_map
        )

    else:

        return (
            vision_tokens,
            mask_map
        )

class ClipIntraBlock(nn.Module):
    def __init__(self,num_features):
        super().__init__()
        self.num_features = num_features
        self.conv_first =nn.Conv1d(in_channels=self.num_features, out_channels=192, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv_second =nn.Conv1d(in_channels=192, out_channels=self.num_features, kernel_size=1)

    def forward(self, x, data_dict, clip_L, inference):
        intra_x = x.permute(1, 2, 0)  # LND -> NDL
        intra_x = intra_x[:, :, -clip_L:].float()  # N  clipD 256/196

        intra_x = self.conv_first(intra_x)  # N D 256
        intra_x = self.relu(intra_x)
        intra_x = intra_x.permute(0, 2, 1)  # NDL-> NLD
        if not inference:
            loss_clip = self.intra_contra(intra_x, data_dict['clip_patch_label'], data_dict['label'], (16, 16))
        else:
            loss_clip = 0
        intra_x = intra_x.permute(0, 2, 1)  # NLD-> NDL
        intra_x = self.conv_second(intra_x)  # NDL

        intra_x = intra_x.permute(2, 0, 1)  # NDL- >LND
      
        return intra_x, loss_clip

    def intra_contra(self, x, patch_labels, image_labels, spatial_shape):
        if True:
            L = spatial_shape[0] * spatial_shape[1]  # 14 * 14 = 196  16 * 16=256

            embeddings = x[:, -L:, ...]  # (N 196 192) (NLD) (N 256 128)
            embeddings = nn.functional.normalize(embeddings, dim=-1)
            fake_index = (image_labels == 1).nonzero(as_tuple=True)[0]
            fake_embeddings = embeddings[fake_index]  # f_N L D

            ff_patch_labels = patch_labels[fake_index]  # f_N L
            fr_patch_labels = torch.logical_not(ff_patch_labels)

            f_fake_part = fake_embeddings * ff_patch_labels.unsqueeze(-1)  # f_N L D *  f_N L 1 -> f_N L D
            f_real_part = fake_embeddings * fr_patch_labels.unsqueeze(-1)  # f_N L D *  f_N L 1 -> f_N L D

            negative = torch.bmm(f_fake_part, f_real_part.permute(0, 2, 1)) / 0.5  # f_N L L
            positive1 = torch.bmm(f_real_part, f_real_part.permute(0, 2, 1)) / 0.5 # f_N L L
            l_neg = torch.sum(torch.exp(negative))

            l_pos1 = torch.sum(torch.exp(positive1))
            loss_real_intra = -torch.log(l_pos1 / (l_neg + l_pos1))
            #---------------fake----------------------
            positive2 = torch.bmm(f_fake_part, f_fake_part.permute(0, 2, 1)) / 0.5 # f_N L L
            l_pos2 = torch.sum(torch.exp(positive2))
            loss_fake_intra = -torch.log(l_pos2 / (l_neg + l_pos2))
            #loss_intra = loss_fake_intra + loss_real_intra
            loss_intra =  loss_real_intra

            real_index = (image_labels == 0).nonzero(as_tuple=True)[0]
            real_nums = len(real_index)
            fake_nums = len(fake_index)

            if real_nums != 0 and fake_nums != 0 :
                real_embeddings = embeddings[real_index]  # N r_L D
                if fake_nums >= real_nums:
                    random_fake_index = torch.randperm(fake_nums)[:real_nums]
                    random_fake_embeddings = fake_embeddings[random_fake_index]
                    real_neg = torch.bmm(real_embeddings, random_fake_embeddings.permute(0, 2, 1)) / 0.5
                    real_pos = torch.bmm(real_embeddings, real_embeddings.permute(0, 2, 1)) / 0.5

                else:
                    random_real_index = torch.randperm(real_nums)[:fake_nums]
                    random_real_embeddings = real_embeddings[random_real_index]
                    real_neg = torch.bmm(random_real_embeddings, fake_embeddings.permute(0, 2, 1)) / 0.5
                    real_pos = torch.bmm(random_real_embeddings, random_real_embeddings.permute(0, 2, 1)) / 0.5
                loss_real_neg = torch.sum(torch.exp(real_neg))
                loss_real_pos = torch.sum(torch.exp(real_pos))
                loss_inter = -torch.log(loss_real_pos / (loss_real_pos + loss_real_neg))
                loss_clip = loss_inter  + loss_intra

            else:
                loss_clip = loss_intra

        return loss_clip


class RecAttnClip(nn.Module):
    def __init__(self, vit, num_quires, device):
        super().__init__()
        self.vit = vit
        # self.vit.reload_self_attn(layers=6)
        self.resblocks = self.vit.transformer.resblocks
        self.first_layer = 0
        self.clss_nums = num_quires
        self.ln_post = self.vit.ln_post
        self.proj = self.vit.proj
        self.num_features = self.vit.width
        self.device = device
        self.intra_scale = nn.Parameter(torch.zeros(1))
        self.intra_map = {8:0}
        self.clip_intra_blocks = nn.ModuleList([ClipIntraBlock(self.num_features).to(self.device) for _ in range(1)])
        # self.norm_middle = nn.LayerNorm(1024)
        self._freeze()

     

    def build_attn_mask(self, attn_bias):

        num_heads = self.resblocks[0].attn.num_heads
        n, Head, q, h, w = attn_bias.shape

        assert (
                Head == num_heads
        ), f"num_head={Head} is not supported. Modify to {num_heads}"
        attn_bias = attn_bias.reshape(n * Head, q, -1) #(256,128,256)(n*head, dim, l)
        l = attn_bias.shape[-1]
        attn_mask = attn_bias.new_zeros(q + l, q + l) #(385,385)
        attn_mask[:, :q] = -100
        attn_mask[torch.arange(q), torch.arange(q)] = 0
        attn_mask[:q, q] = -100
        attn_mask = attn_mask[None, ...].expand(
            n * Head, -1, -1
        ).clone()
        attn_mask[:, :q, -l:] = attn_bias
        # attn_mask (n*head,1+q+l,1+q+l)
        attn_biases = [attn_mask for _ in self.resblocks.children()]
        return attn_biases

    def _freeze(self):
        for name, param in self.named_parameters():
            if 'clip_intra_blocks' in name :
                param.requires_grad = True
            else:
                param.requires_grad = False
    

    # def _register_hooks(self):
    #     def forward_hook(module, inp, out):
    #         self.features = out

    #     def backward_hook(module, grad_in, grad_out):
    #         self.gradients = grad_out[0]

    #     blk = self.resblocks[18].ln_2
    #     blk.register_forward_hook(forward_hook)
    #     blk.register_full_backward_hook(backward_hook)


    def forward(self, data_dict, clip_features, attn_biases, step, feature_lvc, attn_blk, feat_blk,  index_map, step_cnt, test_step, inference=False, normalize=False):
        cls_token = clip_features[f'layer_{self.first_layer}_cls'].unsqueeze(1).permute(1, 0, 2).clone()  # ND->N1D->1ND
        vision_tokens = clip_features[self.first_layer].permute(1, 0, 2).clone()  # NLD->LND
        # clss_token = cls_token.repeat(self.clss_nums, 1, 1)  # 1ND -> clss_nums, N,D

        # if index_map is not None and inference is False:
        #     # vision_tokens_aug = None
        #     fakeindex = np.where(data_dict['label'].cpu().numpy() ==1.0)[0]
        #     L, N, D = cls_token.size()
        #     # vision_tokens = vision_tokens.reshape(N, 16, 16, D)
        #     B = 4 * index_map * (1-index_map)
        #     score = B * index_map
        #     score_flat = score.view(N, -1)
        #     prob = torch.softmax(score_flat, dim=1)

        #     topk_idx = torch.multinomial(prob, 3)
        #     mask_flat = torch.zeros_like(score_flat)
        #     mask_flat.scatter_(1, topk_idx, 1)

        #     mask_vi = mask_flat.view(N, 16, 16)     

        #     if (step_cnt + 1) % test_step == 0:       
        #         visualize_M_binary(B[fakeindex[:4]], save_dir="./B_bw_1_adap{step_cnt}")
        #         visualize_M_binary(score[fakeindex[:4]], save_dir="./Score_bw_1_adap{step_cnt}")
        #         visualize_M_binary(mask_vi[fakeindex[:4]], save_dir="./mask_bw_1_adap{step_cnt}")

        #     mask_flat = mask_flat.permute(1, 0).unsqueeze(-1)     
        #     vision_tokens_aug = vision_tokens * (1-mask_flat * 0.7).half()
        #     if torch.rand(1) < 0.5:
        #         vision_tokens = vision_tokens_aug

        if torch.isnan(vision_tokens).any():
            print("before mask contains NaN")

        if inference is False:
            if index_map is not None and inference is False:
                vision_tokens, mask_vi = random_center_cutout(vision_tokens=vision_tokens,
                                        labels=data_dict['label'],
                                        )
            if (step_cnt + 1) % test_step == 0:     
                    fakeindex = np.where(data_dict['label'].cpu().numpy() ==1.0)[0]  
                    visualize_M_binary(mask_vi[fakeindex[:6]], save_dir=f"./mask__{test_step}")

        if torch.isnan(vision_tokens).any():
            print("after mask contains NaN")

        x = torch.cat(
            [
                # clss_token,
                cls_token,
                vision_tokens
            ],
            dim=0
        )  # (1+Q+L,N,D)
        x.requires_grad = True 
        clip_L = vision_tokens.shape[0] #

        if torch.isnan(attn_biases).any():
                print("attn_biases contains NaN")
        if torch.isinf(attn_biases).any():
            print("attn_biases contains INF")

        attn_biases = self.build_attn_mask(attn_biases) #(16,16,128,16,16)(batchsize, head, )

                

        loss_clip = 0
        attn_list = []
        for i, blocks in enumerate(self.resblocks.children()):
            # if i == feat_blk:
            #     x_block = self.norm_middle(x.float()) #i +1 = attn_blk,所以添加的是上一层的特征，（385,16,1024）
            # if i in attn_blk:
            #     attn_list.append(attn) #(16,16,385,385) 一个数据维度      
            if torch.isnan(x).any():
                print(f"Before block {i} NaN")
                break

            x = blocks(x, attn_biases[-1], feature_lvc)

            if torch.isnan(x).any():
                print(f"After block {i} NaN")
                break
           
           
            if i == 8:
                intra_x, loss_clip_tmp = self.clip_intra_blocks[self.intra_map[i]](x, data_dict, clip_L, inference)
                if torch.isnan(intra_x).any():
                    print("intra_x contains NAN")
                    break
                loss_clip = loss_clip_tmp + loss_clip
        
                x[-clip_L:, ...] = intra_x * 0.05 + x[-clip_L:, ...]
                if torch.isnan(x).any():
                    print("x in layer 8 contains NAN")
                    break
            # if i == 18:
            #     intra_x = self.tune(x).half()
            #     x = intra_x * 0.05 + x

        # if (i+1) in attn_blk:
        #     attn_list.append(attn)
        # if i == feat_blk - 1:
        #     x_block = x
        # attn_block = torch.cat(attn_list, dim=1) #（16,80,385,385）在head维度拼接, 60代表的是5层*12个注意力头
        # L, N, D = x_block.size()
        # x_block = x_block[-256:, :].reshape(N, 16, 16, D) #(16,16,16,1024) 去除cls，并展开patch维度
        
        x = x.permute(1, 0, 2)  # LND -> NLD(385,16,1024)-(16,385,1024)
        clss_token = x[:, :self.clss_nums, :] #(16,128,1024)
        clss_token = self.ln_post(clss_token)
       
        if self.proj is not None:
            clss_token = clss_token @ self.proj
        if normalize:
            clss_token = F.normalize(clss_token, dim=-1)
        patch_token = x[:, -clip_L:, :]
        
        

        return clss_token, loss_clip, patch_token
