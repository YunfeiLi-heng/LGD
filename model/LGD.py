import numpy as np
from sklearn import metrics
from torch import nn
from model.clip.clip import load
import torch
from model.FD.adapter import Adapter
from .attn import RecAttnClip
from .layer import PostClipProcess, MaskPostXrayProcess, PostAdapterProcess
import torch.nn.functional as F
from trainer.metrics.base_metrics_class import calculate_metrics_for_train
from trainer.metrics.utils import LayerDifferenceHead
from trainer.metrics.utils import mahalanobis_distance
import open_clip
from model.clip.loss import BinaryDiceLoss
from trainer.metrics.utils import visualize_M_binary
from trainer.metrics.utils import cams_to_affinity_label, get_aff_loss

class LGD(nn.Module):
    def __init__(self, clip_name,
                 adapter_vit_name,
                 num_quires,
                 fusion_map,
                 mlp_dim,
                 mlp_out_dim,
                 head_num,
                 device,
                 mode='video'):
        super().__init__()
        self.device = device
        self.clip_model, self.processor = load(clip_name, device=device,download_root='pretrain')
        # self.clip_model.visual.reload_self_attn(layers=13)
        self.adapter = Adapter(vit_name=adapter_vit_name, num_quires=num_quires, fusion_map=fusion_map, mlp_dim=mlp_dim,
                               mlp_out_dim=mlp_out_dim, head_num=head_num, device=self.device)
        self.rec_attn_clip = RecAttnClip(self.clip_model.visual, num_quires,device=self.device)  # 全部参数被冻结
        self.masked_xray_post_process = MaskPostXrayProcess(in_c=num_quires).to(self.device)
        self.clip_post_process = PostClipProcess(num_quires=num_quires, embed_dim=768)
        self.adap_post_process = PostAdapterProcess(num_quires=256, embed_dim=192)
        self.fts_fuse = LayerDifferenceHead(in_channels=1024,embedding_dim=256, index=9)
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.trainable_layer = nn.Linear(1024, 768).to(device)
        self.conv = nn.Conv2d(in_channels=num_quires, out_channels=num_quires, kernel_size=1).to(device)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
       
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        self.mode = mode
        self._freeze()


    def _freeze(self):
        for name, param in self.named_parameters():
            if 'clip_model' in name:
                param.requires_grad = False

    def get_losses(self, data_dict, pred_dict):
        label = data_dict['label'] #N
        xray = data_dict['xray']
        pred = pred_dict['cls']  #N2
        xray_pred = pred_dict['xray_pred']
        loss_intra = pred_dict['loss_intra']
        loss_clip = pred_dict['loss_clip']

        if pred_dict['loc_label'] is not None:
            pred_map = pred_dict['pred_map']
            loc_label = pred_dict['loc_label'].unsqueeze(-1)
            loc_label[loc_label > 0.5], loc_label[loc_label <= 0.5] = 1, 0
      
        criterion = nn.CrossEntropyLoss()
        loss1 = criterion(pred.float(), label)
        loss_dice = BinaryDiceLoss()
        loss_mse = F.mse_loss(xray_pred.squeeze().float(), xray.squeeze().float()) 
        loss_map = None
        if xray is not None:

            if pred_dict['loc_label'] is not None:
                loss_map = loss_dice(pred_map[:, 1, :, :], loc_label) 
                
                loss = 10 * loss1 + 20 * loss_intra + 10 * loss_clip  + 200 * loss_mse + 10 * loss_map

                loss_dict = {
                    'cls': loss1,
                    'xray':loss_mse,
                    'intra': loss_intra,
                    'loss_clip':loss_clip,
                    'loss_map' : loss_map,
                    'overall': loss
                }
            else:
                loss = 10 * loss1 + 20 * loss_intra + 10 * loss_clip + 200 * loss_mse

                loss_dict = {
                    'cls': loss1,
                    'xray': loss_mse,
                    'intra': loss_intra,
                    'loss_clip':loss_clip,
                    'overall': loss
                }
            return loss_dict
        else:
            loss_dict = {
                'cls': loss1,
                'overall': loss1
            }
            return loss_dict

    def get_train_metrics(self, data_dict, pred_dict):
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap,}
        return metric_batch_dict

    def get_test_metrics(self):
        y_pred = np.concatenate(self.prob)
        y_true = np.concatenate(self.label)
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true, y_pred)
        # acc
        acc = self.correct / self.total
        # reset the prob and label
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'label': y_true}

    def forward(self, data_dict, step, attn_blk, feat_blk, step_cnt=None, test_step=None, mean_real=None, inv_covariance_real=None, mean_fake=None, inv_covariance_fake=None, inference=False):
        images = data_dict['image']
        name = data_dict['name']
        clip_images = F.interpolate(
            images,
            size=(224, 224),
            mode='bilinear',
            align_corners=False,
        )


        clip_features, feature_lvc = self.clip_model.extract_features(clip_images, self.adapter.fusion_map.values())
        feature_lvc = self.fts_fuse(feature_lvc)
      

        attn_biases, loss_adapter_intra, feat_block, attn_block, xray_preds = self.adapter(data_dict, clip_features,step, attn_blk, feat_blk,
                                                                                inference) #xray:(16,128,16,16)
        

        clip_map = None
        pred_map = None
        attn_pred = None
        if mean_real is not None:
            B, H, W, C = feat_block.size()
            feat_patch_1 = mahalanobis_distance(feat_block.reshape((B*H*W, C)), mean_real.cuda(), inv_covariance_real.cuda()) #平均值为-9.3e-10, 983.64(16,16,16,192)(192)(192,192)
            feat_patch_2 = mahalanobis_distance(feat_block.reshape((B*H*W, C)), mean_fake.cuda(), inv_covariance_fake.cuda())
            feat_map = (feat_patch_1 - feat_patch_2).reshape((B, H, W)) #几乎全是大于0的值
            clip_map = torch.relu(feat_map)

        data_dict['if_boundary'] = data_dict['if_boundary'].to(self.device)
        xray_preds = [self.masked_xray_post_process(xray_pred, data_dict['if_boundary']) for xray_pred in xray_preds]
        clip_output, loss_clip, patch_tokens = self.rec_attn_clip(data_dict, clip_features, attn_biases, step, feature_lvc, attn_blk, feat_blk, clip_map, step_cnt, test_step, inference, normalize=True)

        if torch.isnan(clip_output).any():
            print("clip_output contains NaN")
        if torch.isnan(patch_tokens).any():
            print("patch_tokens before contains NaN")

        clip_cls_output, cls_feat = self.clip_post_process(clip_output.float()) # N2

        if clip_map is not None:
            real_index = np.where(data_dict['label'].cpu().numpy() ==0.0)[0]
            clip_map = clip_map.clone()  # 创建一份独立的副本
            clip_map[real_index] = 0 

            hard_prompts_real = ["This is a real photo of person, vis."]
            hard_prompts_fake = ["This is a fake photo of person, vis."]

            text_feature = []
            B = len(name)

            # 提取所有图片特征
            id_emb = clip_output.squeeze(1)                 # (B, D)
           
            # L2 normalize
            id_emb = id_emb / (id_emb.norm(dim=1, keepdim=True) + 1e-6)
  
            # 为每个样本复制一份prompt
            prompts_real = hard_prompts_real * B
            prompts_fake = hard_prompts_fake * B

            # Tokenize
            token_real = self.tokenizer(prompts_real).to(self.device)   # (B, 77)
            token_fake = self.tokenizer(prompts_fake).to(self.device)

            # 文本编码
            text_features_real = self.clip_model.encode_text(
                token_real,
                id_emb
            )   # (B, D)
          

            text_features_fake = self.clip_model.encode_text(
                token_fake,
                id_emb
            )   # (B, D)

            # L2 normalize
            text_features_real = text_features_real / text_features_real.norm(
                dim=-1, keepdim=True
            )

            text_features_fake = text_features_fake / text_features_fake.norm(
                dim=-1, keepdim=True
            )

            # 拼接
            # shape: (B, 2, D)
            text_feature = torch.stack(
                [text_features_real, text_features_fake],
                dim=1
            )
            # text_feature = torch.stack(text_feature)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            text_feature = text_feature.squeeze(2)

            patch_tokens = self.trainable_layer(patch_tokens.float())
            patch_tokens = patch_tokens / (patch_tokens.norm(dim=-1, keepdim=True) + 1e-6)

         

            anomaly_map = 100.0 * torch.matmul(patch_tokens, text_feature.float().transpose(1, 2))


            B, L, C = anomaly_map.shape
            H = int(np.sqrt(L))
            # pred_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
            #                         size=256, mode='bilinear', align_corners=True)
            pred_map = anomaly_map.permute(0, 2, 1).view(B, 2, H, H)
            pred_map = torch.softmax(pred_map, dim=1)



       
        outputs = {
            'xray_pred': xray_preds[-1],  # N 1 224 224
            'clip_cls_output': clip_cls_output,
            
        }

        prob = torch.softmax(outputs['clip_cls_output'], dim=1)[:, 1]
        
      
        pred_dict = {
            'cls': outputs['clip_cls_output'],
            'prob': prob,
            'xray_pred': outputs['xray_pred'],
            'loss_intra': loss_adapter_intra,
            'loss_clip':loss_clip,
            'pred_map': pred_map,
            'loc_label': clip_map,
            'attn_pred': attn_pred,
            'feat': cls_feat

   
            
        }

        if inference:
            self.prob.append(
                pred_dict['prob']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            self.label.append(
                data_dict['label']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            # deal with acc
            _, prediction_class = torch.max(outputs['clip_cls_output'], 1)
            correct = (prediction_class == data_dict['label']).sum().item()
            self.correct = self.correct + correct
            self.total = self.total +  data_dict['label'].size(0)

        return pred_dict, feat_block, attn_block
