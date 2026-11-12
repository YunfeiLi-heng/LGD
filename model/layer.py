from torch import nn
import torch
from torch.nn import functional as F
import warnings
from timm.models.layers import to_2tuple
from trainer.metrics.utils import mahalanobis_distance


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
            self, input_dim, hidden_dim, output_dim, num_layers, affine_func=nn.Linear
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            affine_func(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            # L R L R L
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Fusion(nn.Module):
    def __init__(self, clip_dim, adapter_dim):
        super().__init__()
        self.clip_dim = clip_dim
        self.adapter_dim = adapter_dim
        self.proj = nn.Sequential(
            LayerNorm(clip_dim),
            nn.Conv2d(clip_dim, adapter_dim, kernel_size=1),
        )

    def forward(self, x, clip_x, spatial_shape):
        h, w = spatial_shape
        n, l, d = clip_x.shape

        if l == h*w:
            clip_x = clip_x.permute(0, 2, 1).view(n, d, h, w)  # NLD->NDL->NDhw
        else:
            clip_x = clip_x.permute(0, 2, 1).view(n, d, 14, 14)  # NLD->NDL->NDhw
            clip_x = F.interpolate(
                clip_x.contiguous(),
                size= (16, 16),
                mode="bilinear",
                align_corners=False,
            ) #ND 14 14 => N D 16 16
        clip_x = self.proj(clip_x).view(n, self.adapter_dim, h * w).permute(0, 2, 1)
        x = x + clip_x  # NLD

        return x



class MaskPostXrayProcess(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.process = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=3, stride=1, padding=1),  # (N Q h,w)->(N 64 h,w))
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=3, stride=1, padding=1),  # (N 32 h,w)
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_c, out_channels=1, kernel_size=1, stride=1, padding=0),  # (N 16 h,w)

            nn.ConvTranspose2d(in_channels=1, out_channels=1 , kernel_size=16, stride=16) 
        )
        self.upsample = nn.ConvTranspose2d(in_channels=1, out_channels=1 , kernel_size=16, stride=16) # (N 16 h,w)->(N 1 256 256)
        # self.alpha = nn.Linear(1, 256)
        # self.beta = nn.Linear(1, 256)


    def forward(self, x, if_boundaries):
        x = x.reshape(x.shape[0], x.shape[1], -1) #(N Q 256)
        x = x.permute(0, 2, 1)#(N L Q)
        if_boundaries = if_boundaries.unsqueeze(-1)  # (NL1) 不是boundry的patch块置为0

        x = x * if_boundaries #(N L Q) * (N L 1)
        x = x.permute(0, 2, 1) #(N Q L)
        x = x.reshape(x.shape[0], x.shape[1], 16, 16)

        post_x = self.process(x)  #(N 1 224 224)
        return post_x



class PostClipProcess(nn.Module):
    """
    NQD -> ND -> N2

    """

    def __init__(self, num_quires, embed_dim):
        super().__init__()

        self.first_process = nn.Sequential(
            nn.Conv1d(in_channels=num_quires, out_channels=num_quires // 2, kernel_size=3, stride=1, padding=1), # NQD->N1D
            nn.BatchNorm1d(num_quires // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_quires // 2, out_channels=num_quires // 4 , kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_quires // 4),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_quires // 4, out_channels=1, kernel_size=3, stride=1, padding=1),

        )
        #self.norm = VT_LN(embed_dim)
        self.second_process = nn.Sequential(  # ND->N2
            nn.Linear(in_features=embed_dim, out_features=embed_dim // 2),
            nn.ReLU(),
            nn.Linear(in_features=embed_dim // 2, out_features=embed_dim // 4),
            nn.ReLU(),
            #nn.Linear(in_features=embed_dim // 4, out_features=embed_dim // 8),
            #nn.ReLU(),
            # nn.Linear(in_features=embed_dim // 4, out_features=2)
        )
        self.cls =  nn.Linear(in_features=embed_dim // 4, out_features=2)



    def forward(self, x):
        # x = self.first_process(x)  #NQD->N1D(16,128,768)-(16,1,768)
        # x = torch.cat((x, fea), dim=-1)
        x = x.squeeze()
        # feat = x
        x = self.second_process(x)
        feat = x
        x = self.cls(x)
        return x, feat
    

class PostAdapterProcess(nn.Module):
    """
    NQD -> ND -> N2

    """

    def __init__(self, num_quires, embed_dim):
        super().__init__()

        self.first_process = nn.Sequential(
            nn.Conv1d(in_channels=num_quires, out_channels=num_quires // 2, kernel_size=3, stride=1, padding=1), # NQD->N1D
            nn.BatchNorm1d(num_quires // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_quires // 2, out_channels=num_quires // 4 , kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_quires // 4),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_quires // 4, out_channels=1, kernel_size=3, stride=1, padding=1),

        )
        #self.norm = VT_LN(embed_dim)
        self.second_process = nn.Sequential(  # ND->N2
            nn.Linear(in_features=embed_dim, out_features=embed_dim // 2),
            nn.ReLU(),
            nn.Linear(in_features=embed_dim // 2, out_features=embed_dim // 4),
            nn.ReLU(),
            #nn.Linear(in_features=embed_dim // 4, out_features=embed_dim // 8),
            #nn.ReLU(),
            nn.Linear(in_features=embed_dim // 4, out_features=2)
        )
        # self.second_process = MLP(192, 256, 2, 3)
        # self.pool = nn.AdaptiveMaxPool1d(1)


    def forward(self, x):
        x = self.first_process(x)  #NQD->N1D(16,256,192)
        x = x.squeeze(1)
        x = self.second_process(x)
        x = torch.sigmoid(x)
        return x



class VT_LN(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class PatchEmbed(nn.Module):

    def __init__(
            self,
            img_size=256,
            patch_size=16,
            in_chans=3,
            embed_dim=192,
            norm_layer=None,
            bias=False,
            **kwargs
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )

        self.norm = VT_LN(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # NDL
        x = x.permute(0, 2, 1)  # NDL->NLD
        # x = self.norm(x)
        return x


# class RankConsistencyLoss(nn.Module):

#     def __init__(self,
#                  feat_dim=192,
#                  pos_thresh=0.7,
#                  neg_thresh=0.3,
#                  margin=1.2):
#         super().__init__()

#         self.pos_thresh = pos_thresh
#         self.neg_thresh = neg_thresh
#         self.margin = margin

#         # token scorer
#         self.scorer = nn.Sequential(
#             nn.Linear(feat_dim, feat_dim // 2),
#             nn.ReLU(),
#             nn.Linear(feat_dim // 2, 1)
#         ).cuda()

#     def forward(self, M, P_i):
#         """
#         M:   (B, H, W)
#         P_i: (B, HW, C)
#         """

#         B, HW, C = P_i.shape

#         # --------------------------------------------------
#         # flatten mask
#         # --------------------------------------------------
#         M_flat = M.flatten(1)  # (B, HW)

#         # --------------------------------------------------
#         # token forgery score
#         # --------------------------------------------------
#         s = self.scorer(P_i).squeeze(-1)  # (B, HW)

#         # --------------------------------------------------
#         # positive / negative masks
#         # --------------------------------------------------
#         pos_mask = (M_flat > self.pos_thresh).float()   # (B, HW)
#         neg_mask = (M_flat < self.neg_thresh).float()   # (B, HW)

#         # --------------------------------------------------
#         # avoid division by zero
#         # --------------------------------------------------
#         pos_num = pos_mask.sum(dim=1)  # (B,)
#         neg_num = neg_mask.sum(dim=1)  # (B,)

#         valid_mask = (pos_num > 0) & (neg_num > 0)

#         # --------------------------------------------------
#         # mean positive score
#         # --------------------------------------------------
#         s_pos = (s * pos_mask).sum(dim=1) / (pos_num + 1e-6)

#         # --------------------------------------------------
#         # mean negative score
#         # --------------------------------------------------
#         s_neg = (s * neg_mask).sum(dim=1) / (neg_num + 1e-6)

        
#         loss = F.softplus(
#             -(s_pos - s_neg)
#         )
#         loss = loss[valid_mask]

#         return loss.mean()

class DistributionSeparationLoss(nn.Module):

    def __init__(
        self,
        topk_ratio=0.2
    ):
        super().__init__()

        self.topk_ratio = topk_ratio

    def forward(
        self,
        M,
        P_i,
        mu_real,
        inv_cov_real
    ):
        """
        M:            (B,H,W)
        P_i:          (B,HW,C)

        mu_real:      (C,)
        inv_cov_real: (C,C)
        """

        B, HW, C = P_i.shape

        M_flat = M.flatten(1)

        # --------------------------------------------------
        # Top-K / Bottom-K
        # --------------------------------------------------

        k = max(int(HW * self.topk_ratio), 1)

        topk_idx = torch.topk(
            M_flat,
            k,
            dim=1,
            largest=True
        ).indices

        bottomk_idx = torch.topk(
            M_flat,
            k,
            dim=1,
            largest=False
        ).indices

        # --------------------------------------------------
        # gather features
        # --------------------------------------------------

        topk_feat = torch.gather(
            P_i,
            dim=1,
            index=topk_idx.unsqueeze(-1).expand(-1, -1, C)
        )

        bottomk_feat = torch.gather(
            P_i,
            dim=1,
            index=bottomk_idx.unsqueeze(-1).expand(-1, -1, C)
        )

        # --------------------------------------------------
        # reshape
        # --------------------------------------------------

        topk_feat = topk_feat.reshape(-1, C)
        bottomk_feat = bottomk_feat.reshape(-1, C)

        # --------------------------------------------------
        # Mahalanobis distance
        # --------------------------------------------------

        d_pos = mahalanobis_distance(
            topk_feat,
            mu_real,
            inv_cov_real
        )

        d_neg = mahalanobis_distance(
            bottomk_feat,
            mu_real,
            inv_cov_real
        )

        # --------------------------------------------------
        # restore batch dimension
        # --------------------------------------------------

        d_pos = d_pos.reshape(B, k).mean(dim=1)
        d_neg = d_neg.reshape(B, k).mean(dim=1)

        # --------------------------------------------------
        # Soft Ranking
        #
        # want:
        # d_pos > d_neg
        # --------------------------------------------------

        loss = F.softplus(
            -(0.001 * (d_pos - d_neg) + 1e-8)
        )

        return loss.mean()