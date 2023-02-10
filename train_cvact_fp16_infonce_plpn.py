from __future__ import print_function, division
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import time
import os
# os.environ['CUDA_LAUNCH_BLOCKING']='1'
from model import two_view_net_swin_infonce_plpn
from utils import update_average, load_network, save_network
import wandb
from types import SimpleNamespace
from timm.optim.optim_factory import create_optimizer
from torchmetrics import Accuracy
from wty_new_image_folder import CVACT_Data
from torch.nn import functional as F
import random
from PIL import ImageFilter, ImageOps
from random_erasing import RandomErasing
import yaml
from tqdm import tqdm
wandb.init(project="university", entity="dreamer0312")

version = torch.__version__


######################################################################
# Options
# --------

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
    parser.add_argument('--l2_loss', action='store_true', help='l2')
    parser.add_argument('--circle_loss', action='store_true', help='circle_loss')
    parser.add_argument('--kl_loss', action='store_true', help='kl_loss')
    parser.add_argument('--triplet_loss', action='store_true', help='triplet_loss')
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

    return parser





def train_model(opt):
    data_dir = opt.data_dir

    transform_train_list_street = [
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomCrop((opt.h, opt.w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    transform_train_list_sate = [
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomCrop((opt.h, opt.w)),
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




    print(transform_train_list_street)
    data_transforms = {
        'train_street': transforms.Compose(transform_train_list_street),
        'train_sate' : transforms.Compose(transform_train_list_sate),
        'val': transforms.Compose(transform_val_list)}



    image_datasets = CVACT_Data(data_dir, data_transforms['train_street'], data_transforms['train_sate'])
    

    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize,
                                              shuffle=True, num_workers=9, pin_memory=True, drop_last=True)
    print(image_datasets.__len__())

    class_names = image_datasets.classes
    print(f'there are {len(class_names)} IDs')

    # ============ building networks ... ============
    model = two_view_net_swin_infonce_plpn(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
                                      LPN=True, block=opt.block)

    accuracy = Accuracy(num_classes=len(class_names)).cuda()
    
    
    print(model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of params: {n_parameters//1000000} M')

    
    model = model.cuda()
    num_epochs = 120
    start_epoch = 0

    #============= Loss ===============================
    criterion_class = nn.CrossEntropyLoss()
    infonce = SupConLoss(temperature=0.1)

    # ============ preparing optimizer ... ============
    lr_skip_keywords = {"model_1", "model_2", "plpn"} #"model_4"
    wd_skip_keywords = {'absolute_pos_embed', 'relative_position_bias_table', 'norm', "pos_embed"}

    parameters = set_wd_lr_normal(model, wd_skip_keywords, lr_skip_keywords, opt.lr)

    if opt.optimizer == "Adamw":
        args = SimpleNamespace()
        args.weight_decay = 0.05
        args.opt = 'adamw'
        args.lr = opt.lr
        args.momentum = 0.9
        optimizer = create_optimizer(args, parameters)
    elif opt.optimizer == "SGD":
        args = SimpleNamespace()
        args.weight_decay = 5e-4
        args.opt = 'sgd'
        args.lr = opt.lr
        args.momentum = 0.9
        args.nesterov = True
        optimizer = create_optimizer(args, parameters)

    # =================FP 16 scaler====================
    scaler = torch.cuda.amp.GradScaler()

    #=================scheduler =======================
    scheduler = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)



    #########################################################
    since = time.time()
    for epoch in range(num_epochs - start_epoch):
        epoch = epoch + start_epoch
        print('=========================================')
        print(f'Epoch {epoch}/{(num_epochs - 1)}')
        print('-' * 10)
        

        train_one_epoch(model, epoch, criterion_class, infonce, 
                        optimizer, accuracy, dataloaders, scheduler, scaler, num_epochs ,opt)
                        
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))





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


def train_one_epoch(model, epoch, criterion_class, infonce, optimizer, accuracy, dataloaders, scheduler, scaler, num_epochs, opt):

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


    for data in tqdm(dataloaders):
        
        step_corrects = 0.0
        step_corrects2 = 0.0
        step_corrects3 = 0.0
        step_lpn_corrects = 0.0
        step_lpn_corrects2 = 0.0
        step_lpn_corrects3 = 0.0
        loss_main = 0.0
        

        sate_data, street_data,  all_label = data
        sate_data = sate_data.cuda(non_blocking=True)
        street_data = street_data.cuda(non_blocking=True)
        all_label = all_label.cuda(non_blocking=True)
        

        # zero the parameter gradients
        optimizer.zero_grad()

        

        with torch.cuda.amp.autocast():
            result = model(sate_data, street_data)
            y1_s4_logits, y2_s4_logits = result['global_logits']
            _, preds = torch.max(y1_s4_logits.data, 1)
            _, preds2 = torch.max(y2_s4_logits.data, 1)

    
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

        wandb.log({"step loss": loss})

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
    epoch_loss_main = running_loss_main / one_epoch_step
    epoch_loss_branch4 = running_loss_branch4 / one_epoch_step
    epoch_infonce_loss = running_loss_infonce / one_epoch_step
    epoch_acc_sate = step_corrects_accu / one_epoch_step
    epoch_acc_street = step_corrects2_accu / one_epoch_step
    epoch_lpn_acc_sate = step_lpn_corrects_accu / one_epoch_step
    epoch_lpn_acc_street = step_lpn_corrects2_accu / one_epoch_step
    epoch_loss_global = running_loss_global / one_epoch_step
    
    wandb.log({"epoch_loss": epoch_loss})

    print(f'Loss: {epoch_loss:.4f}')
    print(f"epoch_loss_global: {epoch_loss_global:.4f},       epoch_loss_branch4: {epoch_loss_branch4:.4f}     epoch_infonce_sup: {epoch_infonce_loss} ")
    print(f"Satellite_Acc:{100*epoch_acc_sate:.4f}%    Street_Acc:{100*epoch_acc_street:.4f}%")
    print(f"Satellite_LPN_Acc:{100*epoch_lpn_acc_sate:.4f}%    Street_LPN_Acc:{100*epoch_lpn_acc_street:.4f}%")

    
    scheduler.step()

    if (epoch+1) % 20 == 0:
        save_network(model, opt.name, epoch)




if __name__ =='__main__':
    #fix_random_seeds(114514)

    parser = get_args_parser()
    opt = parser.parse_args() 
    opt.nclasses = 35531
    
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    # set gpu ids
    if len(gpu_ids) > 0:
        # torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        print(f' training is on GPU {gpu_ids[0]}')
        
    dir_name = os.path.join('./model', opt.name)
    if not opt.resume:
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        # save opts
        with open('%s/opts.yaml' % dir_name, 'w') as fp:
            yaml.dump(vars(opt), fp, default_flow_style=False)
    
    
    
    
    train_model(opt)

