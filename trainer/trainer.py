import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import datetime
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from trainer.metrics.utils import get_test_metrics
from torch.nn import DataParallel
from trainer.metrics.utils import Attention_Correlation_weight_reshape_loss
from trainer.metrics.utils import fit_inv_covariance
from trainer.metrics.utils import mahalanobis_distance
import torch.nn.functional as F
from trainer.metrics.utils import calculate_threshold
from sklearn import metrics
from trainer.metrics.utils import save_cam_simple
from trainer.metrics.utils import visualize_M_binary
from model.layer import DistributionSeparationLoss
FFpp_pool = ['FaceForensics++', ]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(
            self,
            config,
            model,
            optimizer,
            scheduler,
            logger,
            metric_scoring='auc',
            time_now=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
            swa_model=None
    ):
        # check if all the necessary components are implemented
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.writers = {}  # dict to maintain different tensorboard writers for each dataset and metric
        self.logger = logger
        self.metric_scoring = metric_scoring
        # maintain the best metric of all epochs
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf')
            if self.metric_scoring != 'eer' else float('inf'))
        )
        self.speed_up()  # move model to GPU

        # get current time
        self.timenow = time_now
        # create directory path
        if 'task_target' not in config:
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + '_' + self.timenow
            )
        else:
            task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + task_str + '_' + self.timenow
            )
        os.makedirs(self.log_dir, exist_ok=True)

    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            # update directory path
            writer_path = os.path.join(
                self.log_dir,
                phase,
                dataset_key,
                metric_key,
                "metric_board"
            )
            os.makedirs(writer_path, exist_ok=True)
            # update writers dictionary
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]

    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        #self.model = DataParallel(self.model,device_ids=[0,1])

        if self.config['ddp'] == True:
            num_gpus = torch.cuda.device_count()
            print(f'avai gpus: {num_gpus}')
            # local_rank=[i for i in range(0,num_gpus)]
            self.model = DDP(self.model, device_ids=[self.config['local_rank']], find_unused_parameters=True,
                             output_device=self.config['local_rank'])
            # self.optimizer =  nn.DataParallel(self.optimizer, device_ids=[int(os.environ['LOCAL_RANK'])])


    def DP_speed_up(self):
        self.model.device = device
        self.model.to(device)
        self.model = DataParallel(self.model,device_ids=[0,1])


    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, dataset_key, ckpt_info=None):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config['ddp'] == True:
            torch.save(self.model.state_dict(), save_path)
        else:
            if 'svdd' in self.config['model_name']:
                torch.save({'R': self.model.R,
                            'c': self.model.c,
                            'state_dict': self.model.state_dict(), }, save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_swa_ckpt(self):
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"swa.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")

    def save_feat(self, phase, fea, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = fea
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    # def train_step(self, data_dict, step, attn_blk, feat_blk):
    #     if self.config['optimizer']['type'] == 'sam':
    #         for i in range(2):
    #             predictions = self.model(data_dict)
    #             losses = self.model.get_losses(data_dict, predictions)
    #             if i == 0:
    #                 pred_first = predictions
    #                 losses_first = losses
    #             self.optimizer.zero_grad()
    #             losses['overall'].backward()
    #             if i == 0:
    #                 self.optimizer.first_step(zero_grad=True)
    #             else:
    #                 self.optimizer.second_step(zero_grad=True)
    #         return losses_first, pred_first
    #     else:
    #         self.optimizer.zero_grad()
    #         predictions, feat_patch, attn_map = self.model(data_dict, step, attn_blk, feat_blk)
    #         if type(self.model) is DDP:
    #             losses = self.model.module.get_losses(data_dict, predictions)
    #         else:
    #             losses = self.model.get_losses(data_dict, predictions)
    #         #self.optimizer.zero_grad()
    #         losses['overall'].backward()
    #         self.optimizer.step()

    #         return losses, predictions

    def train_epoch(
            self,
            epoch,
            train_data_loader,
            mean_real,
            mean_fake, 
            inv_covariance_real, inv_covariance_fake, feat_tensorlist, feat_tensorlist_fake,
            mean_real_ori, mean_fake_ori, inv_covariance_real_ori, inv_covariance_fake_ori,
            test_data_loaders=None,
    ):

        self.logger.info("===> Epoch[{}] start!".format(epoch))

        if epoch >= 3:
            times_per_epoch = 1
        elif 1<= epoch <3  :
            times_per_epoch = 10
        else:
            times_per_epoch = 5
        

        test_step = len(train_data_loader) // times_per_epoch
     
        step_cnt = epoch * len(train_data_loader)

        # define training recorder
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        min_threshold_H = [int(i) for i in self.config['min_threshold_H'].split()]
        min_threshold = [int(i) for i in self.config['min_threshold'].split()]
        # attention_loss = Attention_Correlation_weight_reshape_loss(c_out=c_out, c_in=c_in, c_cross=c_cross)
        # feat_tensorlist = []
        # feat_tensorlist_fake = []
        criterion = torch.nn.BCEWithLogitsLoss()
        loss_con = DistributionSeparationLoss()

        for iteration, data_dict in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):

            global_iter = epoch * len(train_data_loader) + iteration + 1
            max_iter = len(train_data_loader) * self.config['nEpochs'] 
            step = global_iter / max_iter

            if (global_iter != 0 and global_iter % int(self.config['update_epoch'] * len(train_data_loader)) == 0):
                feat_tensorlist = torch.cat(feat_tensorlist, dim=0).cuda() #(234956,192)
                # feat_tensorlist = feat_tensorlist[:80000, :]
                inv_covariance_real = fit_inv_covariance(feat_tensorlist).cpu()
                mean_real = feat_tensorlist.mean(dim=0).cpu()  
                mean_real_ori, inv_covariance_real_ori = mean_real, inv_covariance_real
                # gauss_param_real = {'mean': mean_real.tolist(), 'covariance': inv_covariance_real.tolist()}

                feat_tensorlist = []

                feat_tensorlist_fake = torch.cat(feat_tensorlist_fake, dim=0).cuda()    #(65952,192)  
                # feat_tensorlist_fake = feat_tensorlist_fake[:50000, :]                                             
                inv_covariance_fake = fit_inv_covariance(feat_tensorlist_fake).cpu()
                mean_fake = feat_tensorlist_fake.mean(dim=0).cpu() 
                # gauss_param_fake = {'mean': mean_fake.tolist(), 'covariance': inv_covariance_fake.tolist()}
                mean_fake_ori, inv_covariance_fake_ori = mean_fake, inv_covariance_fake
                feat_tensorlist_fake = []

            if epoch > 0:
                inv_covariance_real = 0.7 * inv_covariance_real_ori + 0.3 * inv_covariance_real
                mean_real = 0.7 * mean_real_ori + 0.3 * mean_real

                inv_covariance_fake = 0.7 * inv_covariance_fake_ori + 0.3 * inv_covariance_fake
                mean_fake = 0.7 * mean_fake_ori + 0.3 * mean_fake

            self.setTrain()
            for key in data_dict.keys():
                if data_dict[key] != None and key != 'name':
                    data_dict[key] = data_dict[key].to(device)

            self.optimizer.zero_grad()

            
            predictions, feat_patch, attn_map = self.model(data_dict, step=step, attn_blk=self.config['attn_blk'], feat_blk=self.config['feat_blk'], mean_real=mean_real, inv_covariance_real=inv_covariance_real,
                                                                          mean_fake=mean_fake, inv_covariance_fake=inv_covariance_fake, step_cnt=step_cnt, test_step=test_step)
            '''
            # score_fake = predictions['cls'][:, 1].sum()
            # self.optimizer.zero_grad()
            # score_fake.backward(retain_graph=True)
            # feat_fake = self.model.rec_attn_clip.features.detach()
            # grad_fake = self.model.rec_attn_clip.gradients.detach()
            # weights_fake = grad_fake.mean(dim=-1, keepdim=True)
            # cam_fake = F.relu((weights_fake * feat_fake).sum(dim=-1))
            # cam = (cam_fake - cam_fake.min()) / (cam_fake.max() - cam_fake.min() + 1e-8)
            # L, N = cam_fake.size()
            # cam_fake = cam_fake[-256:, :].reshape(N, 16, 16, 1)
            # mask_fake = (cam_fake > 0.5)

            # score_real = predictions['cls'][:, 0].sum()
            # self.optimizer.zero_grad()
            # score_real.backward(retain_graph=True)
            # feat_real = self.model.rec_attn_clip.features.detach()
            # grad_real = self.model.rec_attn_clip.gradients.detach()
            # weights_real = grad_real.mean(dim=-1, keepdim=True)
            # cam_real = F.relu((weights_real * feat_real).sum(dim=-1))
            # cam_real = (cam_real - cam_real.min()) / (cam_real.max() - cam_real.min() + 1e-8)
            # L, N = cam_real.size()
            # cam_real = cam_real[-256:, :].reshape(N, 16, 16, 1)
            # mask_real = (cam_real > 0.5)

            # k = 0.25
            # thresh = torch.quantile(cam.view(N, -1).float(), 1-k, dim=1)
            # mask = cam > thresh.view(N, 1, 1, 1)
            
            # 第二种高效实现方式，获取特征张量（需要梯度）
            features = self.model.adapter.features

            # 使用autograd.grad直接计算梯度，避免多次backward

            score_fake = predictions['cls_adap'][:, 1].sum()
            grad_fake = torch.autograd.grad(score_fake, features, retain_graph=True)[0]

            # score_real = predictions['cls'][:, 0].sum() 
            # grad_real = torch.autograd.grad(score_real, features, retain_graph=True)[0]

            # 计算CAM（使用detach避免梯度传播）
            feat_detached = features.detach()
            weights_fake = grad_fake.mean(dim=-1, keepdim=True)
            cam_fake = F.relu((weights_fake * feat_detached).sum(dim=-1))
            cam_fake = (cam_fake - cam_fake.min()) / (cam_fake.max() - cam_fake.min() + 1e-8)
            N, L = cam_fake.size()
            cam_fake = cam_fake[:, -256:].reshape(N, 16, 16, 1)
            # mask_fake = (cam_fake > 0.5)
            k = 0.25
            thresh_fake = torch.quantile(cam_fake.view(N, -1).float(), 1-k, dim=1)
            mask_fake = cam_fake > thresh_fake.view(N, 1, 1, 1)

            # weights_real = grad_real.mean(dim=-1, keepdim=True)
            # cam_real = F.relu((weights_real * feat_detached).sum(dim=-1))
            # cam_real = (cam_real - cam_real.min()) / (cam_real.max() - cam_real.min() + 1e-8)
            # # L, N = cam_real.size()
            # cam_real = cam_real[:, -256:].reshape(N, 16, 16, 1)
            # # mask_real = (cam_real > 0.5)
            # k = 0.50
            # thresh_real = torch.quantile(cam_real.view(N, -1).float(), 1-k, dim=1)
            # mask_real = cam_real > thresh_real.view(N, 1, 1, 1)
            
            # self.optimizer.zero_grad()
            '''
            realindex = np.where(data_dict['label'].cpu().numpy() == 0.0)[0] 
            attn_map_real = torch.sigmoid(torch.mean(attn_map[realindex,:, -256:, -256:], dim=1)) 
            feat_patch_real = feat_patch[realindex[:8]] 
            B, H, W, C = feat_patch_real.size() #(4,16,16,192)
            # mask_real = mask_real[realindex[:8]]
            # feat_patch_real = feat_patch_real * mask_real
            feat_tensorlist.append(feat_patch_real.reshape(B*H*W, C).cpu().detach()) #(1024,1024)，一直在列表里叠加

            fakeindex = np.where(data_dict['label'].cpu().numpy() == 1.0)[0] 
            feat_patch_fake = feat_patch[fakeindex[:8]] #（16,16,16,1024）——(4,16,16,1024)
            attn_map_fake = torch.sigmoid(torch.mean(attn_map[fakeindex,:, -256:, -256:], dim=1))

            del attn_map
            
            # if epoch == 0:
            #     feat_patch_fake_inner = feat_patch_fake[:, min_threshold_H[0]:min_threshold_H[1], min_threshold[0]:min_threshold[1], :]  #（4,16,16,192）
            #     # mask_fake = mask_fake[fakeindex[:8]]
            #     # feat_patch_fake_inner = feat_patch_fake * mask_fake
            #     B, H, W, C = feat_patch_fake_inner.size()#(8,36,192)
            #     feat_tensorlist_fake.append(feat_patch_fake_inner.reshape(B*H*W, C).cpu().detach())  #（256,1024）
            # else:
            #     B, H, W, C = feat_patch_fake.size()
            #     maha_patch_1 = mahalanobis_distance(feat_patch_fake.reshape((B*H*W, C)), mean_real.cuda(), inv_covariance_real.cuda()) 
            #     maha_patch_2 = mahalanobis_distance(feat_patch_fake.reshape((B*H*W, C)), mean_fake.cuda(), inv_covariance_fake.cuda()) 
            #     score = (maha_patch_1 - maha_patch_2).reshape((B, -1))
            #     topk = int(0.05 * H * W)
            #     _, topk_indices = torch.topk(score, topk, dim=1)
            #     topk_indices = topk_indices.unsqueeze(-1).expand(-1, -1, C).cpu()
            #     feat_patch_fake_inner = feat_patch_fake.reshape(B, H*W, C).cpu().detach().gather(1, topk_indices)#(8,38,192)
            #     feat_tensorlist_fake.append(feat_patch_fake_inner.reshape(-1, C))  #（256,1024）

            if epoch == 0:
                feat_patch_fake_inner = feat_patch_fake[:, min_threshold_H[0]:min_threshold_H[1], min_threshold[0]:min_threshold[1], :]  #（4,16,16,192）
                # mask_fake = mask_fake[fakeindex[:8]]
                # feat_patch_fake_inner = feat_patch_fake * mask_fake
                B, H, W, C = feat_patch_fake_inner.size()#(8,36,192)
                feat_tensorlist_fake.append(feat_patch_fake_inner.reshape(B*H*W, C).cpu().detach())  #（256,1024）
            else:
                B, H, W, C = feat_patch_fake.size()
                maha_patch_1 = mahalanobis_distance(feat_patch_fake.reshape((B*H*W, C)), mean_real.cuda(), inv_covariance_real.cuda()) 
                maha_patch_2 = mahalanobis_distance(feat_patch_fake.reshape((B*H*W, C)), mean_fake.cuda(), inv_covariance_fake.cuda()) 
                score = (maha_patch_1 - maha_patch_2).reshape((B, H, W))
                # soft weight
                w = torch.sigmoid(score / 0.5)

                weight_mask = (w > 0.5)[:, min_threshold_H[0]:min_threshold_H[1], min_threshold[0]:min_threshold[1]]

                # weighted feature aggregation
                feat_fake_inner = feat_patch_fake[:, min_threshold_H[0]:min_threshold_H[1], min_threshold[0]:min_threshold[1], :][weight_mask].cpu().detach()  # (B, C)

                feat_tensorlist_fake.append(feat_fake_inner)

                # score_v = torch.relu((maha_patch_1 - maha_patch_2).reshape((B, H, W)))
                # feat_v = feat_patch_fake_inner.reshape()


            if type(self.model) is DDP:
                losses = self.model.module.get_losses(data_dict, predictions) 
            else:
                losses = self.model.get_losses(data_dict, predictions) 

            if  epoch >0 and global_iter >= int(self.config['update_epoch'] * len(train_data_loader)):
                fakeindex = np.where(data_dict['label'].cpu().numpy() ==1.0)[0]
                B, H, W, C = feat_patch.size()
                maha_patch_1 = mahalanobis_distance(feat_patch.reshape((B*H*W, C)), mean_real.cuda(), inv_covariance_real.cuda()) #平均值为-9.3e-10, 983.64(16,16,16,192)(192)(192,192)
                maha_patch_2 = mahalanobis_distance(feat_patch.reshape((B*H*W, C)), mean_fake.cuda(), inv_covariance_fake.cuda()) #9.3e-10, 9.93e-8
                if iteration % 300 ==0:
                    print(inv_covariance_real.mean())
                    print(inv_covariance_fake.mean())
                del feat_patch
                patch_map = (maha_patch_1 - maha_patch_2).reshape((B, H, W)) #几乎全是大于0的值
                index_map = torch.relu(patch_map)
                
                # B1, HW, C1 = feature_token.size()
                # maha_real = mahalanobis_distance(feature_token.reshape((B1*HW, C1)), mean_real.cuda(), inv_covariance_real.cuda())
                # maha_fake = mahalanobis_distance(feature_token.reshape((B1*HW, C1)), mean_fake.cuda(), inv_covariance_fake.cuda())
                # loc_map = torch.relu((maha_real - maha_fake)).reshape((B, H, W))               
                # loss_rank = loss_con(index_map[fakeindex], feature_token[fakeindex], mean_real.cuda(), inv_covariance_real.cuda())
               
                losses['overall'] = losses['overall'] 
                # losses['rank'] = loss_rank
                # losses['map'] = loss_map
                # losses['mse'] = loss_mse
               
            else:
                losses['overall'] = losses['overall']
                # losses['map'] = loss_map
                # losses['mse'] = loss_mse

            losses['overall'].backward() #cls 0.05,. inter 0.34
            for name, p in self.model.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    print(f"NaN gradient: {name}")

            for name, p in self.model.named_parameters():
                if torch.isnan(p).any():
                    print(f"NaN first appears before opti in parameter: {name}")
                    raise RuntimeError("Parameter became NaN")
            self.optimizer.step()
            for name, p in self.model.named_parameters():
                if torch.isnan(p).any():
                    print(f"NaN first appears in parameter: {name}")
                    raise RuntimeError("Parameter became NaN")
            

            if 'SWA' in self.config and self.config['SWA'] and epoch > self.config['swa_start']:
                self.swa_model.update_parameters(self.model)

            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
            ## store loss
            for name, value in losses.items():
                train_recorder_loss[name].update(value)

            # run tensorboard to visualize the training process
            if iteration % 300 == 0:
                if 'SWA' in self.config and self.config['SWA'] and (epoch > self.config['swa_start'] or self.config['dry_run']):
                    self.scheduler.step()
                # info for loss
                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    if v_avg == None:
                        loss_str += f"training-loss, {k}: not calculated"
                        continue
                    loss_str += f"training-loss, {k}: {v_avg}    "
                    # tensorboard-1. loss
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_loss/{k}', v_avg, global_step=step_cnt)
                self.logger.info(loss_str)
                # info for metric
                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    if v_avg == None:
                        metric_str += f"training-metric, {k}: not calculated    "
                        continue
                    metric_str += f"training-metric, {k}: {v_avg}    "
                    # tensorboard-2. metric
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_metric/{k}', v_avg, global_step=step_cnt)
                self.logger.info(metric_str)

                # clear recorder.
                # Note we only consider the current 300 samples for computing batch-level loss/metric
                for name, recorder in train_recorder_loss.items():  # clear loss recorder
                    recorder.clear()
                for name, recorder in train_recorder_metric.items():  # clear metric recorder
                    recorder.clear()

            # if (step_cnt + 1) % test_step == 0:
            #     save_cam_simple(image=data_dict['image'][fakeindex[0]],cam=cam_fake[0],save_path=f"./cam_vis/step_{step_cnt}.png")

            if epoch >0 and global_iter > int(self.config['update_epoch'] * len(train_data_loader)) and (step_cnt + 1) % test_step == 0 :
                visualize_M_binary(index_map[fakeindex[:4]], save_dir=f"./forgery_bw_1_adap_{global_iter}")
                # visualize_M_binary(loc_map[realindex[:4]], save_dir=f"./forgery_bw_2_adap_{global_iter}")

            # run test
            if (step_cnt + 1) % test_step == 0:
                if test_data_loaders is not None :
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                        step_g=step, attn_blk=self.config['attn_blk'], feat_blk=self.config['feat_blk'],
                        mean_real=mean_real, mean_fake=mean_fake, inv_covariance_real=inv_covariance_real, inv_covariance_fake=
                        inv_covariance_fake,
                        train_data_loader=train_data_loader, 
                        global_iter = global_iter,
                        test_step=test_step
                        
                    )

                    # total_end_time = time.time()
            # total_elapsed_time = total_end_time - total_start_time
            # print("总花费的时间: {:.2f} 秒".format(total_elapsed_time))
            step_cnt = step_cnt +  1
        return test_best_metric, mean_real, mean_fake, inv_covariance_real, inv_covariance_fake, feat_tensorlist, feat_tensorlist_fake, mean_real_ori, mean_fake_ori, inv_covariance_real_ori, inv_covariance_fake_ori

    def get_respect_acc(self, prob, label):
        pred = np.where(prob > 0.5, 1, 0)
        judge = (pred == label)
        zero_num = len(label) - np.count_nonzero(label)
        acc_fake = np.count_nonzero(judge[zero_num:]) / len(judge[zero_num:])
        acc_real = np.count_nonzero(judge[:zero_num]) / len(judge[:zero_num])
        return acc_real, acc_fake

    def test_one_dataset(self, data_loader, step, attn_blk, feat_blk, mean_real, mean_fake, inv_covariance_real,
                   inv_covariance_fake, epoch, train_data_loader, global_iter, test_step):
        # define test recorder
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        feature_lists = []
        label_lists = []
        tol_label_val = np.array([], dtype=float)
        tol_pred_maha = np.array([],dtype=float)
        for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
            # get data
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')  # remove the specific label
            data_dict['label'] = torch.where(data_dict['label'] != 0, 1, 0)  # fix the label to 0 and 1 only
            # move data to GPU elegantly
            for key in data_dict.keys():
                if data_dict[key] != None and key != 'name':
                    data_dict[key] = data_dict[key].cuda()
            # model forward without considering gradient computation
            predictions, feat_patch, attn_map, tol_label_val, tol_pred_maha = self.inference(data_dict, step, attn_blk, feat_blk, mean_real, mean_fake, inv_covariance_real,
                   inv_covariance_fake, epoch, train_data_loader, global_iter, tol_label_val, tol_pred_maha, test_step)
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            #feature_lists += list(predictions['feat'].cpu().detach().numpy())
            if type(self.model) is not AveragedModel:
                # compute all losses for each batch data
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)

                # store data by recorder
                for name, value in losses.items():
                    test_recorder_loss[name].update(value)

        # if epoch > 0:
        auc_mvg = metrics.roc_auc_score(tol_label_val, tol_pred_maha)
        acc_mvg = calculate_threshold(tol_pred_maha, tol_label_val, 0)
        # else:
        #     auc_mvg = 0
        #     acc_mvg = 0


        #return test_recorder_loss, np.array(prediction_lists), np.array(label_lists), np.array(feature_lists)
        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists), auc_mvg, acc_mvg

    def save_best(self, epoch, iteration, step, losses_one_dataset_recorder, key, metric_one_dataset, auc_mvg, acc_mvg):
        best_metric = self.best_metrics_all_time[key].get(self.metric_scoring,
                                                          float('-inf') if self.metric_scoring != 'eer' else float(
                                                              'inf'))
        # Check if the current score is an improvement
        improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (
                metric_one_dataset[self.metric_scoring] < best_metric)
        if improved:
            # Update the best metric
            self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict']
            # Save checkpoint, feature, and metrics if specified in config
            if self.config['save_ckpt'] and key not in FFpp_pool:
                self.save_ckpt('test', key, f"{epoch}+{iteration}")
            self.save_metrics('test', metric_one_dataset, key)
        if losses_one_dataset_recorder is not None:
            # info for each dataset
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                writer = self.get_writer('test', key, k)
                v_avg = v.average()
                if v_avg == None:
                    print(f'{k} is not calculated')
                    continue
                # tensorboard-1. loss
                writer.add_scalar(f'test_losses/{k}', v_avg, global_step=step)
                loss_str += f"testing-loss, {k}: {v_avg}    "
            self.logger.info(loss_str)
        # tqdm.write(loss_str)
        metric_str = f"dataset: {key}    step: {step}    "
        for k, v in metric_one_dataset.items():
            if k == 'pred' or k == 'label' or k == 'dataset_dict':
                continue
            metric_str += f"testing-metric, {k}: {v}    "
            # tensorboard-2. metric
            writer = self.get_writer('test', key, k)
            writer.add_scalar(f'test_metrics/{k}', v, global_step=step)
        if 'pred' in metric_one_dataset:
            acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
            metric_str += f'testing-metric, acc_real:{acc_real}; acc_fake:{acc_fake}'
            writer.add_scalar(f'test_metrics/acc_real', acc_real, global_step=step)
            writer.add_scalar(f'test_metrics/acc_fake', acc_fake, global_step=step)
        self.logger.info(metric_str)
        self.logger.info(f"auc_mvg, acc_mvg is {auc_mvg, acc_mvg}")

    def test_epoch(self, epoch, iteration, test_data_loaders, step, step_g, attn_blk, feat_blk, mean_real, mean_fake, inv_covariance_real,
                   inv_covariance_fake, train_data_loader, global_iter, test_step):
        # set model to eval mode
        self.setEval()

        # define test recorder
        losses_all_datasets = {}
        metrics_all_datasets = {}
        best_metrics_per_dataset = defaultdict(dict)  # best metric for each dataset, for each metric
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0, 'video_auc': 0, 'dataset_dict': {}}
        # testing for all test data
        keys = test_data_loaders.keys()
        auc_mvg_all = 0
        acc_mvg_all = 0
       
        for key in keys:
            # save the testing data_dict
            data_dict = test_data_loaders[key].dataset.data_dict
            self.save_data_dict('test', data_dict, key)

            # compute loss for each dataset
            #losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps = self.test_one_dataset(
            #    test_data_loaders[key])
            losses_one_dataset_recorder, predictions_nps, label_nps, auc_mvg, acc_mvg = self.test_one_dataset(test_data_loaders[key], step_g, attn_blk, feat_blk, mean_real, mean_fake, inv_covariance_real
                   ,inv_covariance_fake, epoch, train_data_loader, global_iter, test_step)
            # print(f'stack len:{predictions_nps.shape};{label_nps.shape};{len(data_dict["image"])}')
            losses_all_datasets[key] = losses_one_dataset_recorder
            metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                                  img_names=data_dict['image'])
            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name] += value
            avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]
            if type(self.model) is AveragedModel:
                metric_str = f"Iter Final for SWA:    "
                for k, v in metric_one_dataset.items():
                    metric_str += f"testing-metric, {k}: {v}    "
                self.logger.info(metric_str)
                self.logger.info(f"auc_mvg, acc_mvg is {auc_mvg, acc_mvg}")
                continue
            auc_mvg_all += auc_mvg
            acc_mvg_all += acc_mvg 
            self.save_best(epoch, iteration, step, losses_one_dataset_recorder, key, metric_one_dataset, auc_mvg, acc_mvg)

        if len(keys) > 0 and self.config.get('save_avg', False):
            # calculate avg value
            for key in avg_metric:
                if key != 'dataset_dict':
                    avg_metric[key] /= len(keys)
            auc_mvg_avg =auc_mvg_all / len(keys)
            acc_mvg_avg = acc_mvg_all / len(keys)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric, auc_mvg_avg, acc_mvg_avg)

        self.logger.info('===> Test Done!')
        return self.best_metrics_all_time  # return all types of mean metrics for determining the best ckpt

    @torch.no_grad()
    def inference(self, data_dict, step, attn_blk, feat_blk, mean_real, mean_fake, inv_covariance_real,
                   inv_covariance_fake, epoch, train_data_loader, global_iter, tol_label_val, tol_pred_maha, test_step):
        
        predictions, feat_patch, attn_map = self.model(data_dict, step, attn_blk, feat_blk, step_cnt=step, test_step=test_step, inference=True)
        
        if  global_iter >= int(self.config['update_epoch'] * len(train_data_loader)):
            B, H, W, C = feat_patch.size()
            feat = feat_patch.reshape((B*H*W, C)).cpu().detach()
            maha_real = mahalanobis_distance(feat, mean_real, inv_covariance_real).reshape((B, H*W)).cpu().data/torch.tensor(C**(0.5))
            maha_fake = mahalanobis_distance(feat, mean_fake, inv_covariance_fake).reshape((B, H*W)).cpu().data/torch.tensor(C**(0.5))
            maha = maha_real - maha_fake
            maha = maha.numpy()
            tol_pred_maha = np.concatenate((tol_pred_maha, np.array(np.mean(maha, axis=1), dtype=float)))
        
        tol_label_val = np.concatenate((tol_label_val, data_dict['label'].cpu().numpy()))

        return predictions, feat_patch, attn_map, tol_label_val, tol_pred_maha
