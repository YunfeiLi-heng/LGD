import os
import argparse
import random
import datetime
import yaml
from datetime import timedelta
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from model.ds import DS
from trainer.trainer import Trainer
from trainer.metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter
import torch.nn as nn
from torch.utils.data import Sampler
import numpy as np

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--config_path', type=str,
                    default='config/train.yaml',
                    help='path to detector YAML file')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=False)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', type=int, default=3)
parser.add_argument('--task_target', type=str, default="", help='specify the target of current training task')
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)



def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])

# # ====== 1️⃣ 定义平衡采样器 ======
# class BalancedBatchSampler(Sampler):
#     def __init__(self, labels, batch_size):
#         self.labels = np.array(labels)
#         self.pos_indices = np.where(self.labels == 1)[0]  # 假样本索引
#         self.neg_indices = np.where(self.labels == 0)[0]  # 真样本索引
#         self.batch_size = batch_size
#         assert batch_size % 2 == 0, "batch_size 必须是偶数！"

#     def __iter__(self):
#         np.random.shuffle(self.pos_indices)
#         np.random.shuffle(self.neg_indices)
#         min_len = min(len(self.pos_indices), len(self.neg_indices))
#         half_batch = self.batch_size // 2
#         num_batches = min_len // half_batch

#         for i in range(num_batches):
#             pos_batch = self.pos_indices[i * half_batch:(i + 1) * half_batch]
#             neg_batch = self.neg_indices[i * half_batch:(i + 1) * half_batch]
#             batch = np.concatenate([pos_batch, neg_batch])
#             np.random.shuffle(batch)
#             yield batch.tolist()

#     def __len__(self):
#         return min(len(self.pos_indices), len(self.neg_indices)) // (self.batch_size // 2)


def prepare_training_data(config):
    # Only use the blending dataset class in training
    if True:
        train_set = DeepfakeAbstractBaseDataset(config, mode='train')
        print(' Train set : FF++ 23')
    
    # train_sampler = BalancedBatchSampler(train_set.label_list, batch_size=config['train_batchSize']) #可能影响性能。



    if config['ddp']:
        sample = DistributedSampler(train_set)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                sampler=sample
            )
    else:
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                shuffle=True,
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
            )
        # train_data_loader = torch.utils.data.DataLoader(
        #     dataset = train_set,
        #     batch_sampler=train_sampler,
        #     num_workers=int(config['train_batchSize']),
        #     collate_fn=train_set.collate_fn,
        #     pin_memory=True
        # )
    

    return train_data_loader  


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
            config=config,
            mode='test',
        )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_optimizer(model, config, c_cross, c_in, c_out):
    opt_name = config['optimizer']['type']
    
    if config['lambda_c_param'] == '0 0 0':
        optimizer_dict = [{"params": model.parameters(), 'lr': config['optimizer'][opt_name]['lr']}]
    else:
        optimizer_dict = [{"params": model.parameters(), 'lr': config['optimizer'][opt_name]['lr'],},
                        {"params": [c_in, c_out, c_cross], 'lr': config['optimizer'][opt_name]['lr']},]


    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=optimizer_dict,
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=optimizer_dict,
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))



def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
        return scheduler
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def main():
    # parse options and load config
    torch.autograd.set_detect_anomaly(True)

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['local_rank'] = args.local_rank
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat']=False
    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat

    # create logger
    timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
    logger_path =  os.path.join(
                config['log_dir'],
                config['model_name'] + task_str + '_' + timenow
            )
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info('Save log to {}'.format(logger_path))
    config['ddp']= args.ddp
    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True
    if config['ddp']:
        # dist.init_process_group(backend='gloo')
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=30)
        )
        logger.addFilter(RankFilter(0))
    # prepare the training data loader
    train_data_loader = prepare_training_data(config)

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model

    model = DS(clip_name=config['clip_model_name'],
           adapter_vit_name=config['vit_name'],
           num_quires=config['num_quires'],
           fusion_map=config['fusion_map'],
           mlp_dim=config['mlp_dim'],
           mlp_out_dim=config['mlp_out_dim'],
           head_num=config['head_num'],
           device=config['device'])

    [c_cross, c_in, c_out] = [nn.Parameter(torch.tensor(float(i)).cuda()) for i in config['c_initilize'].split()]
    # prepare the optimizer
    optimizer = choose_optimizer(model, config, c_cross, c_in, c_out)

    # prepare the scheduler
    scheduler = choose_scheduler(config, optimizer)

    # prepare the metric
    metric_scoring = choose_metric(config)

    # prepare the trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring)
    # start training
    # trainer.load_ckpt('logs/ds_ _2026-07-18-02-10-31/test/avg/ckpt_best.pth')
    # trainer.load_ckpt('logs/ds_ _2026-10-01-03-57-06/test/avg/ckpt_best.pth')
    mean_real, mean_fake = None, None
    inv_covariance_real, inv_covariance_fake = None, None
    mean_real_ori, mean_fake_ori = 0, 0
    inv_covariance_real_ori, inv_covariance_fake_ori = 0, 0
    feat_tensorlist = []
    feat_tensorlist_fake = [] 

    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        trainer.model.epoch = epoch
        best_metric, mean_real, mean_fake, inv_covariance_real, inv_covariance_fake, feat_tensorlist, feat_tensorlist_fake, mean_real_ori, mean_fake_ori, inv_covariance_real_ori, inv_covariance_fake_ori = trainer.train_epoch(
                    epoch=epoch,
                    train_data_loader=train_data_loader,
                    test_data_loaders=test_data_loaders,
                    mean_real=mean_real, mean_fake=mean_fake,
                    inv_covariance_real=inv_covariance_real, inv_covariance_fake=inv_covariance_fake, feat_tensorlist = feat_tensorlist, feat_tensorlist_fake=feat_tensorlist_fake,
                    mean_real_ori=mean_real_ori, mean_fake_ori=mean_fake_ori, inv_covariance_real_ori=inv_covariance_real_ori, inv_covariance_fake_ori=inv_covariance_fake_ori
                )
        if best_metric is not None:
            logger.info(f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!")
            logger.info(f"c_cross, c_in, c_out is {c_cross, c_in, c_out}")
            
    logger.info("Stop Training on best Testing metric {}".format(parse_metric_for_print(best_metric))) 
    # update
    if 'svdd' in config['model_name']:
        model.update_R(epoch)
    if scheduler is not None:
        scheduler.step()

    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()



if __name__ == '__main__':
    main()
