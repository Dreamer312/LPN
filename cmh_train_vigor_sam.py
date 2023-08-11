from __future__ import print_function, division
import argparse
import torch
# import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms
import time
import os
# os.environ['CUDA_LAUNCH_BLOCKING']='1'
os.environ["WANDB_MODE"] = "disabled"
from model import two_view_net_swin_infonce_region_cluster
from utils import save_network
from types import SimpleNamespace
from timm.optim.optim_factory import create_optimizer
from torchmetrics import Accuracy
from vigor_dataset_simple import TrainDataloader
from torch.nn import functional as F
import random
from PIL import ImageFilter, ImageOps
from random_erasing import RandomErasing
import yaml
from tqdm import tqdm

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
wandb.init("UniQT")
# from accelerate import Accelerator
# torch.cuda.empty_cache()


version = torch.__version__

 
######################################################################
# Options
# --------
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups




def get_args_parser():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name', default='two_view', type=str, help='output model name')
    parser.add_argument('--pool', default='avg', type=str, help='pool avg')
    parser.add_argument('--data_dir', default='./data/train', type=str, help='training dir path')
    parser.add_argument('--train_all', action='store_true', help='use all training data')
    parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
    parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
    parser.add_argument('--stride', default=2, type=int, help='stride')
    parser.add_argument('--pad', default=10, type=int, help='padding')
    parser.add_argument('--h', default=384, type=int, help='height')
    parser.add_argument('--w', default=384, type=int, help='width')
    parser.add_argument('--views', default=2, type=int, help='the number of views')
    parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
    parser.add_argument('--use_dense', action='store_true', help='use densenet121')
    parser.add_argument('--use_NAS', action='store_true', help='use NAS')
    parser.add_argument('--use_swin', action='store_true', help='use swin transformer 224*224')
    parser.add_argument('--optimizer', default="Adamw", type=str, help="type of optimizer")
    parser.add_argument('--use_vit', action='store_true', help='use vit_B_16 224*224')
    parser.add_argument('--swin_au_block', action='store_true', help='use swin block in au classifiers')
    parser.add_argument('--branch3_weight', default=0, type=float, help='branch3 loss weight')
    parser.add_argument('--branch4_weight', default=0, type=float, help='branch4 loss weight')
    parser.add_argument('--same_area', action='store_true', help='same_area')
    parser.add_argument('--use_stage3_branch', action='store_true', help='block3')
    parser.add_argument('--use_stage4_branch', action='store_true', help='block4')
    parser.add_argument('--se', action='store_true', help='se')
    parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
    parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
    parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation')
    parser.add_argument('--resume', action='store_true', help='use resume trainning')
    parser.add_argument('--share', action='store_true', help='share weight between different view')
    parser.add_argument('--extra_Google', action='store_true', help='using extra noise Google')
    parser.add_argument('--fp16', action='store_true',
                        help='use float16 instead of float32, which will save about 50% memory')
    parser.add_argument('--block', default=6, type=int, help='the num of block')
    parser.add_argument('--class_dim', default=512, type=int, help='part feature dim')
    parser.add_argument('--backbone', default="swint", type=str, help='backbone')
    parser.add_argument('--dataset', default="vigor", type=str, help='dataset')

    parser.add_argument('--epoch', default=100, type=int, help='epoch number')


    return parser





def train_model(opt):

    # accelerate = Accelerator(mixed_precision='fp16', log_with="wandb")
    
    # accelerate.init_trackers("UniQT")
    ddp_rank = int(os.environ['RANK'])
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 #


    data_dir = opt.data_dir

    transform_train_list_street = [
    transforms.Resize((320, 640), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomCrop((320, 640)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]


    transform_train_list_sate = [
    transforms.Resize((320, 320), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomCrop((320, 320)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    transform_train_list_street = [transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4)], p=0.8),
                                    gray_scale(p=0.3),
                                    Solarization(p=0.2),
                                    GaussianBlur(p=0.3),
                                  ] + transform_train_list_street + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]
    
    transform_train_list_sate = [transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4)], p=0.8),
                                    gray_scale(p=0.3),
                                    Solarization(p=0.2),
                                    GaussianBlur(p=0.3),
                                  ] + transform_train_list_sate + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]



    transform_val_list = [
        transforms.Resize(size=(opt.h, opt.w), interpolation=3),  # Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]




    if master_process:
        print(transform_train_list_street)
        
        print(f"opt.same_area {opt.same_area}")
    # assert(0)

    data_transforms = {
            'train_street': transforms.Compose(transform_train_list_street),
            'train_sate' : transforms.Compose(transform_train_list_sate),
            'val': transforms.Compose(transform_val_list)}

    
    image_datasets = TrainDataloader(data_dir, data_transforms['train_street'], data_transforms['train_sate'], opt.same_area)



    # for data in tqdm(image_datasets):
    #     None
    #     # print()
    # assert(0)

    sampler = DistributedSampler(image_datasets)

    # shuffle设置为False    让sample去打乱
    dataloaders = torch.utils.data.DataLoader(image_datasets, sampler=sampler, batch_size=opt.batchsize,
                                              shuffle=False, num_workers=8, pin_memory=False, drop_last=True)

    # class_names = image_datasets.classes
    if master_process:
        print(f'there are {len(image_datasets)} IDs')
    opt.nclasses = len(image_datasets)

    if master_process:
        dir_name = os.path.join('./model', opt.name)
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        # save opts
        with open('%s/opts.yaml' % dir_name, 'w') as fp:
            yaml.dump(vars(opt), fp, default_flow_style=False)

    # ============ building networks ... ============
    model = two_view_net_swin_infonce_region_cluster(len(image_datasets), droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
                                      LPN=True, block=opt.block, model=opt.backbone, class_dim=opt.class_dim, dataset=opt.dataset)


    accuracy = Accuracy(num_classes=len(image_datasets), task='multiclass').cuda()
    
    
    #accelerate.print(model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if master_process:
        print(f'number of params: {n_parameters//1000000} M')

    
    
    num_epochs = opt.epoch
    start_epoch = 0

    #============= Loss ===============================
    criterion_class = nn.CrossEntropyLoss()
    infonce = SupConLoss(temperature=0.1)

    # ============ preparing optimizer ... ============
    lr_skip_keywords = {"model_1", "model_2", "plpn"} #"model_4"
    wd_skip_keywords = {'absolute_pos_embed', 'relative_position_bias_table', 'norm', "pos_embed"}

    
    model.to(device)
    model = DDP(
        model,
        device_ids=[local_rank],
    )
    parameters = set_wd_lr_normal(model.module, wd_skip_keywords, lr_skip_keywords, opt.lr)

    #=================scheduler =======================
    base_optimizer = torch.optim.SGD
   
    optimizer = SAM(parameters, base_optimizer, rho=2.5, lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
    
    
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[80,120,160])
    # resume =False
    # model, criterion_class, infonce, optimizer, dataloaders = accelerate.prepare(model, criterion_class, infonce, optimizer, dataloaders)
    # if resume:
    #     accelerate.load_sate("/home/minghach/Data/CMH/LPN/model/vigor-swint-infonce-UniQT-accelerate-16/each_epoch")

    #########################################################
    since = time.time()
    for epoch in range(num_epochs - start_epoch):
        epoch = epoch + start_epoch
        if master_process:
            print('=========================================')
            print(f'Epoch {epoch}/{(num_epochs - 1)}')
            print('-' * 10)
        

        train_one_epoch_SAM(sampler, device, master_process, model, epoch, criterion_class, infonce, 
                        optimizer, accuracy, dataloaders, scheduler, num_epochs ,opt)
                        
        time_elapsed = time.time() - since
        if master_process:
            print('Training complete in {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))
    dist.destroy_process_group()



def fix_random_seeds(seed):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)

    print(f"seed is {seed}")




##############################################################################
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels_column = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            #mask = torch.eq(labels, labels.T).float().to(device)
            mask = (labels_column == labels).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # print(f"contrast_feature {contrast_feature.size()}")
        # print(f"mask {mask.size()}")
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # print(f"mask repeat {mask.size()}")
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # print(f"logits {logits.size()}")
        # assert(0)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss




#https://github.com/facebookresearch/deit/blob/main/augment.py
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img



def set_wd_lr_normal(model, wd_skip_keywords=(), lr_skip_keywords=(), lr=0):
    backbone_has_decay = []
    backbone_no_decay = []
    others_has_decay = []
    others_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or check_keywords_in_name(name, wd_skip_keywords):

            if check_keywords_in_name(name, lr_skip_keywords):
                backbone_no_decay.append(param)
                #print(f"backbone {name} has no weight decay")
            else:
                others_no_decay.append(param)
                #print(f"others {name} has no weight decay")
        else:
            if check_keywords_in_name(name, lr_skip_keywords):
                backbone_has_decay.append(param)
                #print(f"backbone {name} has weight decay")
            else:
                others_has_decay.append(param)
                #print(f"others {name} has weight decay")

    return [{'params': backbone_has_decay, 'lr': lr * 0.1},
            {'params': backbone_no_decay, 'lr': lr * 0.1, 'weight_decay': 0.},
            {'params': others_has_decay},
            {'params': others_no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
            # print("======================")
            # print(name)
            # print(keyword)
            # print("======================")
    return isin




def one_LPN_output(outputs, labels, criterion, block):
    # part = {}

    sm = nn.Softmax(dim=1)
    num_part = block
    score = 0


    loss = 0
    for i in range(num_part):
        part = outputs[i]
        score += sm(part)
        loss += criterion(part, labels)

    _, preds = torch.max(score.data, 1)
    loss = loss / num_part

    return preds, loss


def train_one_epoch_SAM(sampler, device, master_process, model, epoch, criterion_class, infonce, optimizer, accuracy, dataloaders, scheduler, num_epochs, opt):

    running_loss = 0.0
    running_loss_main = 0.0
    running_loss_infonce = 0.0
    running_loss_branch4 = 0.0
    step_corrects_accu = 0.0
    step_corrects2_accu = 0.0
    step_corrects3_accu = 0.0
    step_lpn_corrects_accu = 0.0
    step_lpn_corrects2_accu = 0.0
    step_lpn_corrects3_accu = 0.0
    one_epoch_step = 0
    running_loss_global = 0.0



    for data in (tqdm(dataloaders) if master_process else dataloaders):
        sampler.set_epoch(epoch)
        step_corrects = 0.0
        step_corrects2 = 0.0
        step_corrects3 = 0.0
        step_lpn_corrects = 0.0
        step_lpn_corrects2 = 0.0
        step_lpn_corrects3 = 0.0
        loss_main = 0.0
        

        sate_data, street_data,  all_label = data
        sate_data = sate_data.to(device)
        street_data =street_data.to(device)
        all_label = all_label.to(device)


        # sate_data = sate_data.cuda(non_blocking=True)
        # street_data = street_data.cuda(non_blocking=True)
        # all_label = all_label.cuda(non_blocking=True)

        print(all_label)
        assert(0)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        
        result = model(sate_data, street_data)
        y1_s4_logits, y2_s4_logits = result['global_logits']
        _, preds = torch.max(y1_s4_logits.data, 1)
        _, preds2 = torch.max(y2_s4_logits.data, 1)

        # print(y1_s4_logits.size())
        # print(all_label.size())


        loss_global = criterion_class(y1_s4_logits, all_label) + criterion_class(y2_s4_logits, all_label)

        # print(f'loss_global')
        # assert(0)
        loss_main = loss_main + loss_global
        #loss_main = torch.tensor([0.]).cuda()
        
        ################################
        sate_embd, street_embd= result['global_embedding']
        sate_embd_norm = F.normalize(sate_embd, dim=1)
        street_embd_norm = F.normalize(street_embd, dim=1)
        features = torch.cat([sate_embd_norm.unsqueeze(1), street_embd_norm.unsqueeze(1)], dim=1)
        loss_infonce = infonce(features, all_label)
        loss_main = loss_main + loss_infonce
        ################################
        
        
        
        #########################################
    
        y1_s4_res_logits, y2_s4_res_logits = result['part_logits']


        branch4_preds, branch4_loss = one_LPN_output(y1_s4_res_logits, all_label, criterion_class, opt.block)      
        branch4_preds2, branch4_loss2 = one_LPN_output(y2_s4_res_logits, all_label, criterion_class, opt.block)
        loss_branch4 = branch4_loss +  branch4_loss2
        loss_branch4 = loss_branch4 #/ 3.0
    
        loss = loss_main + loss_branch4
        #loss = loss_branch4
        
        # if one_epoch_step%100==0:
        #     print(f"loss {loss.item() }")
        #     print(f"loss_branch4 {loss_branch4.item() }")
        #     print(f"loss_global {loss_global.item() }")

        
        with model.no_sync():
            loss.backward()
        optimizer.first_step(zero_grad=True)


        result = model(sate_data, street_data, epoch)
        y1_s4_logits_2, y2_s4_logits_2 = result['global_logits']
        loss_global_2 = criterion_class(y1_s4_logits_2, all_label) + criterion_class(y2_s4_logits_2, all_label)
        y1_s4_res_logits_2, y2_s4_res_logits_2 = result['part_logits']
        _, branch4_loss_2 = one_LPN_output(y1_s4_res_logits_2, all_label, criterion_class, opt.block)      
        _, branch4_loss2_2 = one_LPN_output(y2_s4_res_logits_2, all_label, criterion_class, opt.block)

        sate_embd, street_embd= result['global_embedding']
        sate_embd_norm = F.normalize(sate_embd, dim=1)
        street_embd_norm = F.normalize(street_embd, dim=1)
        features = torch.cat([sate_embd_norm.unsqueeze(1), street_embd_norm.unsqueeze(1)], dim=1)
        loss_infonce_2 = infonce(features, all_label)

        loss_branch4_2 = branch4_loss_2 +  branch4_loss2_2
        loss2 = loss_global_2 + loss_branch4_2 + loss_infonce_2
        

        loss2.backward()
        optimizer.second_step(zero_grad=True)



        scheduler.step()

        wandb.log({"step loss": loss.item()})

        running_loss += loss.item() 
        running_loss_main += loss_main.item() 
        running_loss_branch4 += loss_branch4.item()
        running_loss_global += loss_global.item()
        running_loss_infonce += loss_infonce.item() #torch.tensor([0.]).cuda()#
        

        step_corrects = accuracy(preds, all_label)      
        step_corrects2 = accuracy(preds2, all_label)
        step_lpn_corrects = accuracy(branch4_preds, all_label)  
        step_lpn_corrects2 = accuracy(branch4_preds2, all_label)
        
       
        step_corrects_accu += step_corrects
        step_corrects2_accu += step_corrects2
        step_corrects3_accu += step_corrects3
        step_lpn_corrects_accu += step_lpn_corrects
        step_lpn_corrects2_accu += step_lpn_corrects2
        step_lpn_corrects3_accu += step_lpn_corrects3
        
       
        one_epoch_step += 1

    epoch_loss = running_loss / one_epoch_step
    # epoch_loss_main = running_loss_main / one_epoch_step
    epoch_loss_branch4 = running_loss_branch4 / one_epoch_step
    epoch_infonce_loss = running_loss_infonce / one_epoch_step
    epoch_acc_sate = step_corrects_accu / one_epoch_step
    epoch_acc_street = step_corrects2_accu / one_epoch_step
    epoch_lpn_acc_sate = step_lpn_corrects_accu / one_epoch_step
    epoch_lpn_acc_street = step_lpn_corrects2_accu / one_epoch_step
    epoch_loss_global = running_loss_global / one_epoch_step
    
    #wandb.log({"epoch_loss": epoch_loss})
    if master_process:
        print(f'Loss: {epoch_loss:.4f}')
        print(f"epoch_loss_global: {epoch_loss_global:.4f},       epoch_loss_branch4: {epoch_loss_branch4:.4f}     epoch_infonce_sup: {epoch_infonce_loss} ")
        print(f"Satellite_Acc:{100*epoch_acc_sate:.4f}%    Street_Acc:{100*epoch_acc_street:.4f}%")
        print(f"Satellite_LPN_Acc:{100*epoch_lpn_acc_sate:.4f}%    Street_LPN_Acc:{100*epoch_lpn_acc_street:.4f}%")


    if master_process:
        if (epoch+1) == num_epochs: 
            unwrp_model = accelerate.unwrap_model(model)        
            save_network(unwrp_model, opt.name, epoch)
        
        if (epoch+1) % 10 == 0:
            inter_epoch_path = os.path.join('./model',opt.name, f'epoch_{epoch}')
            if not os.path.isdir(inter_epoch_path):
                os.mkdir(inter_epoch_path)
            accelerate.save_state(inter_epoch_path)
        
        each_epoch_path = os.path.join('./model',opt.name,'each_epoch')
        if not os.path.isdir(each_epoch_path):
            os.mkdir(each_epoch_path)
        accelerate.save_state(each_epoch_path)
        


if __name__ =='__main__':
    # fix_random_seeds(114514)
    parser = get_args_parser()
    opt = parser.parse_args() 
    train_model(opt)