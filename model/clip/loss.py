import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):

        if torch.isnan(input).any():
            print("Dice input contains NaN")

        if torch.isinf(input).any():
            print("Dice input contains Inf")

        if torch.isnan(targets).any():
            print("Dice target contains NaN")

        if torch.isinf(targets).any():
            print("Dice target contains Inf")

        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        if torch.isnan(intersection).any():
            print("Dice inter contains Inf")
        if torch.isnan(input_flat.sum(1) + targets_flat.sum(1)).any():
            print("Dice sum contains Inf")

        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
        return loss

# class BinaryDiceLoss(nn.Module):
#     def __init__(self):
#         super(BinaryDiceLoss, self).__init__()

#     def forward(self, input, targets):

#         smooth = 1e-5
#         eps = 1e-7

#         N = targets.size(0)

#         # 防止fp16计算异常
#         input = input.float()
#         targets = targets.float()

#         input_flat = input.reshape(N, -1)
#         targets_flat = targets.reshape(N, -1)

#         intersection = input_flat * targets_flat

#         numerator = 2 * intersection.sum(dim=1) + smooth
#         denominator = (
#             input_flat.sum(dim=1)
#             +
#             targets_flat.sum(dim=1)
#             +
#             smooth
#         )

#         # 防止异常
#         dice = numerator / (denominator + eps)

#         loss = 1 - dice.mean()

#         return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        l = max_val - min_val
    else:
        l = val_range

    padd = window_size//2
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    c1 = (0.01 * l) ** 2
    c2 = (0.03 * l) ** 2

    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret, ssim_map

