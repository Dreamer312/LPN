from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms
import os
import yaml
from utils import load_network
from vigor_dataset_simple import TestDataloader_sat, TestDataloader_grd
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='./data/cvpr2017_cvusa/val_pt',type=str, help='./test_data')
parser.add_argument('--name', default='three_view_long_share_d0.75_256_s1_google', type=str, help='save model path')
parser.add_argument('--pool', default='avg', type=str, help='avg|max')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='views')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--LPN', action='store_true', help='use LPN' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
opt = parser.parse_args()

###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
opt.fp16 = config['fp16'] 
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
# opt.use_vgg16 = config['use_vgg16']
opt.stride = config['stride']
opt.views = config['views']
# opt.LPN = config['LPN']
opt.LPN = True
opt.block = config['block']

if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']


opt.nclasses = config['nclasses']
name = opt.name
test_dir = opt.test_dir

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

opt.class_dim = config["class_dim"] 
opt.backbone= config["backbone"]
opt.dataset= config["dataset"]


# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True



data_transforms_sat = transforms.Compose([
    transforms.Resize((320, 320), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

data_transforms_street = transforms.Compose([
    transforms.Resize((320, 640), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])


data_dir = test_dir

same_area = config['same_area']
print(f"same_area {same_area}")
image_datasets = {}
image_datasets['satellite'] = TestDataloader_sat(test_dir, data_transforms_sat, same_area)
image_datasets['street'] = TestDataloader_grd(test_dir, data_transforms_street, same_area)

# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
#                                             shuffle=False, num_workers=9) for x in ['satellite','street']}

dataloaders = {}
dataloaders['satellite'] = torch.utils.data.DataLoader(image_datasets['satellite'],
                                                       batch_size=320,
                                                       shuffle=False, 
                                                       num_workers=9
                                                       )

dataloaders['street'] = torch.utils.data.DataLoader(image_datasets['street'],
                                                       batch_size=160,
                                                       shuffle=False, 
                                                       num_workers=9
                                                       )


model, _, epoch = load_network(opt.name, opt)


# actual_weight = "/home/cmh/cmh/projects/LPN/model/vigor-swint-infonce-UniQT-accelerate-29/epoch_89/pytorch_model.bin"
# dict_inter = torch.load(actual_weight)
# msg = model.load_state_dict(dict_inter)
# print(msg)
# print(actual_weight)


ema = False
if ema:
    actual_weight = "/home/cmh/cmh/projects/LPN/model/vigor-swint-infonce-UniQT-accelerate-30/each_epoch_ema.pth"
    dict_inter = torch.load(actual_weight)
    msg = model.load_state_dict(dict_inter)
    print(msg)
    print(actual_weight)

if opt.LPN:
    print('use LPN')
    # model = two_view_net_test(model)
    for i in range(opt.block):
        name = 'classifier'+str(i)
        c = getattr(model, name)
        c.classifier = nn.Sequential()
else:
    model.classifier.classifier = nn.Sequential()
model = model.eval()
model = model.cuda()



def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip




dim=512
with torch.no_grad():
    sat_features = torch.FloatTensor()
    for step, sat_img in enumerate(tqdm(dataloaders['satellite'])):
        n, c, h, w = sat_img.size()

        if step == 0:
            print(f"image size: {n, c, h, w}")

        ff = torch.FloatTensor(n,dim,opt.block).zero_().cuda()

        for i in range(2):
            if(i==1):
                sat_img = fliplr(sat_img)
            sat_input_img = sat_img.cuda()
            sat_result = model(sat_input_img, None) 
            sat_outputs, _ = sat_result['part_logits']
        ff += sat_outputs

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(opt.block) 
        ff = ff.div(fnorm.expand_as(ff))

        ff = ff.view(ff.size(0), -1)
        sat_features = torch.cat((sat_features,ff.data.cpu()), 0)
    print(sat_features.size())
    sat_global_descriptor = sat_features.cpu().numpy()

    if ema:
        np.save("./vigor_sat_ema.npy", sat_global_descriptor)
    else:
        np.save(f"./vigor_sat_gpu{gpu_ids[0]}.npy", sat_global_descriptor)
    
    torch.cuda.empty_cache()

    street_features = torch.FloatTensor()
    for step, street_img in enumerate(tqdm(dataloaders['street'])):
        n, c, h, w = street_img.size()

        if step == 0:
            print(f"image size: {n, c, h, w}")
            
        ff = torch.FloatTensor(n,dim,opt.block).zero_().cuda()

        for i in range(2):
            if(i==1):
                street_img = fliplr(street_img)
            street_input_img = street_img.cuda()
            street_result = model(None, street_input_img) 
            _, street_outputs = street_result['part_logits']
        ff += street_outputs

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(opt.block) 
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)
        street_features = torch.cat((street_features,ff.data.cpu()), 0)
    grd_global_descriptor = street_features.cpu().numpy()

    if ema:
        np.save("./vigor_street_ema.npy", grd_global_descriptor)
    else:
        np.save(f"./vigor_street_gpu{gpu_ids[0]}.npy", grd_global_descriptor)
        
         

print('compute accuracy')


# Initialize metrics
recall_at_1_percent = 0.0
accuracy_top1 = 0.0
accuracy_top5 = 0.0
accuracy_top10 = 0.0
accuracy_top100 = 0.0
accuracy_hit = 0.0

data_amount = 0.0

# Load precomputed descriptors
if ema:
    sat_global_descriptor = np.load("./vigor_sat_ema.npy", allow_pickle=True)
    grd_global_descriptor = np.load("./vigor_street_ema.npy", allow_pickle=True)
else:
    sat_global_descriptor = np.load(f"./vigor_sat_gpu{gpu_ids[0]}.npy", allow_pickle=True)
    grd_global_descriptor = np.load(f"./vigor_street_gpu{gpu_ids[0]}.npy", allow_pickle=True)

# Compute pairwise cosine distance
dist_array = 2 - 2 * np.matmul(grd_global_descriptor, sat_global_descriptor.T)
print(dist_array.shape)

# Define the top-k parameters
top1_percent = int(dist_array.shape[1] * 0.01) + 1
top1 = 1
top5 = 5
top10 = 10
top100 = 100

# For each image in the ground dataset
for i in range(dist_array.shape[0]):
    # Get the distance of the ground truth match
    gt_dist = dist_array[i, dataloaders['street'].dataset.test_label[i][0]]

    # Get the predicted rank of the ground truth match
    prediction = np.sum(dist_array[i, :] < gt_dist)

    # Create a mask that excludes the semi-positive matches
    dist_temp = np.ones(dist_array[i, :].shape[0])
    dist_temp[dataloaders['street'].dataset.test_label[i][1:]] = 0

    # Get the predicted rank of the ground truth match, excluding semi-positive matches
    prediction_hit = np.sum((dist_array[i, :] < gt_dist) * dist_temp)

    # Update the metrics
    if prediction < top1_percent:
        recall_at_1_percent += 1.0
    if prediction < top1:
        accuracy_top1 += 1.0
    if prediction < top5:
        accuracy_top5 += 1.0
    if prediction < top10:
        accuracy_top10 += 1.0
    if prediction < top100:
        accuracy_top100 += 1.0
    if prediction_hit < top1:
        accuracy_hit += 1.0

    data_amount += 1.0

# Compute the final metrics
recall_at_1_percent /= data_amount
accuracy_top1 /= data_amount
accuracy_top5 /= data_amount
accuracy_top10 /= data_amount
accuracy_top100 /= data_amount
accuracy_hit /= data_amount

print('Recall@1%%: %.2f%%, Top1: %.2f%%, Top5: %.2f%%, Top10: %.2f%%, Top100: %.2f%%, Hit rate: %.2f%%' % (
    recall_at_1_percent * 100.0, accuracy_top1 * 100.0, accuracy_top5 * 100.0, accuracy_top10 * 100.0, accuracy_top100 * 100.0, accuracy_hit * 100.0))


    
# accuracy = 0.0
# accuracy_top1 = 0.0
# accuracy_top5 = 0.0
# accuracy_top10 = 0.0
# accuracy_top100 = 0.0
# accuracy_hit = 0.0

# data_amount = 0.0
# sat_global_descriptor = np.load("./vigor_sat.npy", allow_pickle=True)
# grd_global_descriptor = np.load("./vigor_street.npy", allow_pickle=True)
    
# dist_array = 2 - 2 * np.matmul(grd_global_descriptor, sat_global_descriptor.T)

# print(dist_array.shape)

# top1_percent = int(dist_array.shape[1] * 0.01) + 1
# top1 = 1
# top5 = 5
# top10 = 10
# top100 = 100

# for i in range(dist_array.shape[0]):
#     #gt_dist = dist_array[i, test_loader_grd.dataset.test_label[i][0]] # positive sat

#     gt_dist = dist_array[i, dataloaders['street'].dataset.test_label[i][0]]
    

#     prediction = np.sum(dist_array[i, :] < gt_dist)

#     dist_temp = np.ones(dist_array[i, :].shape[0])
    
#     #dist_temp[test_loader_grd.dataset.test_label[i][1:]] = 0 # cover semi-positive sat

#     dist_temp[dataloaders['street'].dataset.test_label[i][1:]] = 0
    
#     prediction_hit = np.sum((dist_array[i, :] < gt_dist) * dist_temp)

#     if prediction < top1_percent:
#         accuracy += 1.0
#     if prediction < top1:
#         accuracy_top1 += 1.0
#     if prediction < top5:
#         accuracy_top5 += 1.0
#     if prediction < top10:
#         accuracy_top10 += 1.0
#     if prediction < top100:
#         accuracy_top100 += 1.0
#     if prediction_hit < top1:
#         accuracy_hit += 1.0
#     data_amount += 1.0

# accuracy /= data_amount
# accuracy_top1 /= data_amount
# accuracy_top5 /= data_amount
# accuracy_top10 /= data_amount
# accuracy_top100 /= data_amount
# accuracy_hit /= data_amount

# print('accuracy = %.2f%% , top1: %.2f%%, top5: %.2f%%, top10: %.2f%%, top100: %.2f%%,hit_rate: %.2f%%' % (
#             accuracy * 100.0, accuracy_top1 * 100.0, accuracy_top5 * 100.0, accuracy_top10 * 100.0, accuracy_top100 * 100.0, accuracy_hit * 100.0))