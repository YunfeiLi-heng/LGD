from sklearn import metrics
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F


def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != 'avg':
            str= str+ f"| {key}: "
            for k,v in value.items():
                str = str + f" {k}={v} "
            str= str+ "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == 'dataset_dict':
                    for key,value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str


def get_test_metrics(y_pred, y_true, img_names):
    def get_video_metrics(image, pred, label):
        result_dict = {}
        new_label = []
        new_pred = []
        # print(image[0])
        # print(pred.shape)
        # print(label.shape)
        for item in np.transpose(np.stack((image, pred, label)), (1, 0)):

            s = item[0]
            if '\\' in s:
                parts = s.split('\\')
            else:
                parts = s.split('/')
            a = parts[-2]
            b = parts[-1]

            if a not in result_dict:
                result_dict[a] = []

            result_dict[a].append(item)
        image_arr = list(result_dict.values())

        for video in image_arr:
            pred_sum = 0
            label_sum = 0
            leng = 0
            for frame in video:
                pred_sum = pred_sum + float(frame[1])
                label_sum  = label_sum +  int(frame[2])
                leng  = leng +  1
            new_pred.append(pred_sum / leng)
            new_label.append(int(label_sum / leng))
        fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
        v_auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return v_auc, v_eer


    y_pred = y_pred.squeeze()
    # auc
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # eer
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # ap
    ap = metrics.average_precision_score(y_true, y_pred)
    # acc
    prediction_class = (y_pred > 0.5).astype(int)
    correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
    acc = correct / len(prediction_class)
    if type(img_names[0]) is not list:
        # calculate video-level auc for the frame-level methods.
        v_auc, _ = get_video_metrics(img_names, y_pred, y_true)
    else:
        # video-level methods
        v_auc=auc

    return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'video_auc': v_auc, 'label': y_true}


def fit_inv_covariance(samples, clip_value=5.0):
    """
    稳定计算特征的逆协方差矩阵（LedoitWolf）
    """
    # 1️⃣ 转 CPU 并转换为 numpy
    samples_cpu = samples.detach().float().cpu().numpy()

    # 2️⃣ 检查并清理异常值
    if not np.isfinite(samples_cpu).all():
        print("⚠️ NaN/Inf detected in features, replacing with 0.")
        samples_cpu = np.nan_to_num(samples_cpu, nan=0.0, posinf=1e6, neginf=-1e6)

    # 3️⃣ 标准化（防止特征分量尺度差异大）
    samples_cpu = (samples_cpu - samples_cpu.mean(axis=0, keepdims=True)) / (   #当 samples_cpu 的形状为 (0, 192) 时，mean(axis=0) 会返回全 NaN，随后所有操作都会产生 NaN，最终 LedoitWolf 报错。
        samples_cpu.std(axis=0, keepdims=True) + 1e-8
    )

    # 4️⃣ 限幅裁剪，防止数值溢出
    samples_cpu = np.clip(samples_cpu, -clip_value, clip_value)

    # 5️⃣ Ledoit–Wolf 协方差估计（自动带正则项）
    try:
        lw = LedoitWolf().fit(samples_cpu)
        precision = lw.precision_
    except Exception as e:
        print("⚠️ LedoitWolf failed:", e)
        precision = np.eye(samples_cpu.shape[1])  # 回退策略

    # 6️⃣ 转回 GPU
    return torch.from_numpy(precision).to(samples.device)

# def fit_inv_covariance(samples): #得到估计的逆协方差矩阵
#     return torch.Tensor(LedoitWolf().fit(samples.cpu()).precision_).to(
#         samples.device
#     )


class Attention_Correlation_weight_reshape_loss(nn.Module):
    def __init__(self, c_out, c_in, c_cross) -> None:
        super(Attention_Correlation_weight_reshape_loss, self).__init__()
        self.c_out = c_out
        self.c_in = c_in
        self.c_cross = c_cross
        self.device = c_out.device #(6,196,196)  (90,196,196)   (90,14,14)
    def forward(self, correlation_map_real, correlation_map_fake, fake_weight):
        if correlation_map_real.shape[0] == 0:
            loss_real = 0
        else:
            B, PP, PP = correlation_map_real.shape #correlation_map_real代表的是注意力图，所以对应的GT为对角线为1，剩下的值设定为原论文公式5的c_1，c_2等值。
            C_matrix = torch.eye(PP).to(self.device) * (1-self.c_out) + self.c_out*torch.ones(PP).to(self.device) #对角线全是1，剩下全为0.6的矩阵（196，196）
            C_matrix = C_matrix.expand(B, -1, -1) #（6，196，196）
            loss_real = torch.sum(torch.abs(correlation_map_real - C_matrix))/(B*(PP*PP-PP)) #这里减去的PP猜测为对角线元素，因为自己和自己的损失不需要计算。

        if correlation_map_fake.shape[0] == 0:
            loss_fake = 0
        else:
            B, PP, PP = correlation_map_fake.shape
            C_matrix_fake_all = []
            for b in range(B):
                fake_index = torch.where(fake_weight[b].reshape(-1)>0)[0] #42个值，伪造的像素点
                real_index = torch.where(fake_weight[b].reshape(-1)<=0)[0] #154个值
                C_matrix_fake = torch.zeros((PP, PP)).to(self.device) + self.c_cross #（196，196） 0 + 0.2，先设定为全为0.2，意思是真实和伪造区域的对比
                for i in fake_index:
                    C_matrix_fake[i, fake_index] = self.c_in #196，196
                for j in real_index:
                    C_matrix_fake[j, real_index] = self.c_out #上面几行代码的意思是扫描所有的伪造像素点和真实像素点，如果都为真实或者伪造，那么对其进行赋值，即c_in和c_out，剩下的即为c_cross=0.2.
                C_matrix_fake_all.append(C_matrix_fake)
            C_matrix_fake_all = torch.stack(C_matrix_fake_all).to(self.device) #（90，196，196）
            # C_matrix_fake = C_matrix_fake.expand(B, -1, -1).cuda()
            loss_fake = torch.sum(torch.abs(correlation_map_fake - C_matrix_fake_all))/(B*(PP*PP-PP))

        loss = loss_fake + loss_real
        return loss
    
def mahalanobis_distance(
        values: torch.Tensor, mean: torch.Tensor, inv_covariance: torch.Tensor
    ) -> torch.Tensor:
        """Compute the batched mahalanobis distance.

        values is a batch of feature vectors.
        mean is either the mean of the distribution to compare, or a second
        batch of feature vectors.
        inv_covariance is the inverse covariance of the target distribution.
        """
        assert values.dim() == 2
        assert 1 <= mean.dim() <= 2
        assert inv_covariance.dim() == 2
        assert values.shape[1] == mean.shape[-1]
        assert mean.shape[-1] == inv_covariance.shape[0]
        assert inv_covariance.shape[0] == inv_covariance.shape[1]

        if mean.dim() == 1:  # Distribution mean.
            mean = mean.unsqueeze(0)
        x_mu = values - mean  # batch x features
        # Same as dist = x_mu.t() * inv_covariance * x_mu batch wise
        dist = torch.einsum("im,mn,in->i", x_mu.float(), inv_covariance.float(), x_mu)
        return dist.sqrt()

def eval_state(probs, labels, thr):
    predict = probs >= thr
    TN = np.sum((labels == 0) & (predict == False))
    FN = np.sum((labels == 1) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    TP = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP

def calculate_threshold(probs, labels, threshold):
    TN, FN, FP, TP = eval_state(probs, labels, threshold)
    ACC = (TP + TN) / labels.shape[0]
    return ACC


def save_cam_simple(image, cam, save_path):
    """最简单的CAM保存函数"""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    if isinstance(image, torch.Tensor):
        image = image.cpu().permute(1, 2, 0).numpy()
    plt.imshow(image)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    if isinstance(cam, torch.Tensor):
        cam = cam.cpu().numpy()
    plt.imshow(cam.squeeze(), cmap='jet')
    plt.title('CAM')
    plt.axis('off')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Saved: {save_path}")


def visualize_M_binary(index_map, save_dir="./vis_M_bw", max_images=4):
    """
    index_map: Tensor (B, H, W), 值为 maha_real - maha_fake
    """
    os.makedirs(save_dir, exist_ok=True)

    # 只取前 max_images 张
    B = index_map.shape[0]
    num = min(B, max_images)

    # CPU & Numpy
    M = index_map[:num].detach().cpu().numpy()

    for i in range(num):
        # 二值化：>0 = 1， else = 0
        bw_mask = (M[i] > 0).astype(np.uint8)

        plt.figure(figsize=(4, 4))
        plt.imshow(bw_mask, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')

        save_path = os.path.join(save_dir, f"M_bw_{i}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"[Saved] {save_path}")

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.proj_2 = nn.Linear(embed_dim, embed_dim)
        # self.proj_3 = nn.Linear(embed_dim*2, embed_dim)

    def forward(self, x):
        # x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = F.relu(x)
        x = self.proj_2(x)
        return x
    
class LayerDifferenceHead(nn.Module):
    """
    Layer Difference Fusion for Deepfake Detection

    input:
        x_all: list of length N
               each element: (L,B,D)

    output:
        (B,C,H,W)
    """

    def __init__(
            self,
            in_channels=1024,
            embedding_dim=256,
            index=9,
    ):
        super().__init__()

        self.indexes = index

        self.linears = nn.ModuleList([
            MLP(input_dim=in_channels,
                embed_dim=embedding_dim)
            for _ in range(index)
        ])

        # 因为是相邻层差值，所以共有 index-1 个
        self.linear_fuse = nn.Conv2d(
            embedding_dim * (index - 1),
            embedding_dim,
            kernel_size=1,
            bias=False
        )

        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x_all):

        feats = []

        # ---------------------------------
        # Step1. projection
        # ---------------------------------
        for i in range(self.indexes):

            x = x_all[i]                    # (L,B,D)

            L, B, D = x.shape

            x = self.linears[i](x.float())

            H = W = int(L ** 0.5)

            x = x.permute(1, 2, 0).reshape(
                B,
                -1,
                H,
                W
            )

            feats.append(x)
        feats = torch.stack(feats, dim=0)   #9(N, d, h, w)

        # ---------------------------------
        # Step2. adjacent difference
        # ---------------------------------
        diff_feats = []

        diff_feats = feats[1:] - feats[:-1]     #保留norm可能性
        diff_feats = diff_feats.permute(1,0,2,3,4)
        B, N, C, H, W = diff_feats.size()
        diff_feats = diff_feats.reshape(B,N*C,H,W)

        # ---------------------------------
        # Step3. fusion
        # ---------------------------------
        # diff_feats = torch.cat(diff_feats, dim=1)

        out = self.linear_fuse(diff_feats)

        out = self.dropout(out)

        return out
    

def cams_to_affinity_label(cam_label, mask=None, ignore_index=255):
    
    b,h,w = cam_label.shape
    
    cam_label_resized = F.interpolate(cam_label.unsqueeze(1).type(torch.float32), size=[h//16, w//16], mode="nearest")

    _cam_label = cam_label_resized.reshape(b, 1, -1)
    _cam_label_rep = _cam_label.repeat([1, _cam_label.shape[-1], 1])
    _cam_label_rep_t = _cam_label_rep.permute(0,2,1)
    aff_label = (_cam_label_rep == _cam_label_rep_t).type(torch.long)
    #aff_label[(_cam_label_rep+_cam_label_rep_t) == 0] = ignore_index
    # for i in range(b):

    #     if mask is not None:
    #         aff_label[i, mask==0] = ignore_index

    #     aff_label[i, :, _cam_label_rep[i, 0, :]==ignore_index] = ignore_index
    #     aff_label[i, _cam_label_rep[i, 0, :]==ignore_index, :] = ignore_index
    # mask
    if mask is not None:
        if mask.dim() == 2:
            aff_label[:, mask == 0] = ignore_index
        else:
            aff_label[mask == 0] = ignore_index

    # (B,N)
    invalid = (_cam_label_rep[:, 0, :] == ignore_index)

    # 列置为 ignore
    aff_label = aff_label.masked_fill(
        invalid.unsqueeze(1),
        ignore_index
    )

    # 行置为 ignore
    aff_label = aff_label.masked_fill(
        invalid.unsqueeze(2),
        ignore_index
    )
    return aff_label

def get_aff_loss(inputs, targets):

    pos_label = (targets == 1).type(torch.int16)
    pos_count = pos_label.sum() + 1
    neg_label = (targets == 0).type(torch.int16)
    neg_count = neg_label.sum() + 1
    #inputs = torch.sigmoid(input=inputs)

    pos_loss = torch.sum(pos_label * (1 - inputs)) / pos_count
    neg_loss = torch.sum(neg_label * (inputs)) / neg_count

    return 0.5 * pos_loss + 0.5 * neg_loss, pos_count, neg_count