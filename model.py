import argparse
import math
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
from swin import SwinConfig, SwinModel, SwinStageLast
from timm.models.layers import  Mlp, DropPath, trunc_normal_
######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x

class ft_net_VGG16(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_VGG16, self).__init__()
        model_ft = models.vgg16_bn(pretrained=True)
        # avg pooling to global pooling
        #if stride == 1:
        #    model_ft.layer4[0].downsample[0].stride = (1,1)
        #    model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.features(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
            x = x.view(x.size(0), x.size(1))

        #x = self.classifier(x)
        return x

# Define the VGG16-based part Model
class ft_net_VGG16_LPN(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=8, row = False):
        super(ft_net_VGG16_LPN, self).__init__()
        model_ft = models.vgg16_bn(pretrained=True)
        # avg pooling to global pooling
        #if stride == 1:
        #    model_ft.layer4[0].downsample[0].stride = (1,1)
        #    model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        self.model = model_ft
        self.block = block
        self.avgpool = nn.AdaptiveAvgPool2d((1,block))
        self.maxpool = nn.AdaptiveMaxPool2d((1,block))
        if row:  # row partition the ground view image
            self.avgpool = nn.AdaptiveAvgPool2d((block,1))
            self.maxpool = nn.AdaptiveMaxPool2d((block,1))
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.features(x)
        # print(x.size())
        if self.pool == 'avg+max':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.avgpool(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(1), -1)
            # print(x)
        elif self.pool == 'max':
            x = self.maxpool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        #x = self.classifier(x)
        return x

# Define vgg16 based square ring partition for satellite images of cvusa/cvact
class ft_net_VGG16_LPN_R(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=4):
        super(ft_net_VGG16_LPN_R, self).__init__()
        model_ft = models.vgg16_bn(pretrained=True)
        # avg pooling to global pooling
        #if stride == 1:
        #    model_ft.layer4[0].downsample[0].stride = (1,1)
        #    model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        self.model = model_ft
        self.block = block
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.features(x)
        # print(x.size())
        if self.pool == 'avg+max':
            x1 = self.get_part_pool(x, pool='avg')
            x2 = self.get_part_pool(x, pool='max')
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.get_part_pool(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(1), -1)
            # print(x)
        elif self.pool == 'max':
            x = self.get_part_pool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        #x = self.classifier(x)
        return x
    # VGGNet's output: 8*8 part:4*4, 6*6, 8*8
    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1,1)) 
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        per_h, per_w = H/(2*self.block),W/(2*self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H+(self.block-c_h)*2, W+(self.block-c_w)*2
            x = nn.functional.interpolate(x, size=[new_H,new_W], mode='bilinear')
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H/2), int(W/2)
            per_h, per_w = H/(2*self.block),W/(2*self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                x_curr = x[:,:,(c_h-i*per_h):(c_h+i*per_h),(c_w-i*per_w):(c_w+i*per_w)]
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)] 
                    x_pad = F.pad(x_pre,(per_h,per_h,per_w,per_w),"constant",0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.insert(0, avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    pad_h = c_h-(i-1)*per_h
                    pad_w = c_w-(i-1)*per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2)+2*pad_h == H:
                        x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    else:
                        ep = H - (x_pre.size(2)+2*pad_h)
                        x_pad = F.pad(x_pre,(pad_h+ep,pad_h,pad_w+ep,pad_w),"constant",0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.insert(0, avgpool)
        return torch.cat(result, dim=2)

# resnet50 backbone
class ft_net_cvusa_LPN(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=6, row=False):
        super(ft_net_cvusa_LPN, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        self.model = model_ft
        self.block = block
        self.avgpool = nn.AdaptiveAvgPool2d((1,block))
        self.maxpool = nn.AdaptiveMaxPool2d((1,block))
        if row:
            self.avgpool = nn.AdaptiveAvgPool2d((block,1))
            self.maxpool = nn.AdaptiveMaxPool2d((block,1))
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # print(x.size())
        if self.pool == 'avg+max':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.avgpool(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(1), -1)
            # print(x)
        elif self.pool == 'max':
            x = self.maxpool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        #x = self.classifier(x)
        return x

class ft_net_cvusa_LPN_R(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=6):
        super(ft_net_cvusa_LPN_R, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        self.model = model_ft
        self.block = block
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # print(x.size())
        if self.pool == 'avg+max':
            x1 = self.get_part_pool(x, pool='avg')
            x2 = self.get_part_pool(x, pool='max')
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.get_part_pool(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(1), -1)
            # print(x)
        elif self.pool == 'max':
            x = self.get_part_pool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        #x = self.classifier(x)
        return x

    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1,1)) 
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        per_h, per_w = H/(2*self.block),W/(2*self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H+(self.block-c_h)*2, W+(self.block-c_w)*2
            x = nn.functional.interpolate(x, size=[new_H,new_W], mode='bilinear')
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H/2), int(W/2)
            per_h, per_w = H/(2*self.block),W/(2*self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                x_curr = x[:,:,(c_h-i*per_h):(c_h+i*per_h),(c_w-i*per_w):(c_w+i*per_w)]
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)] 
                    x_pad = F.pad(x_pre,(per_h,per_h,per_w,per_w),"constant",0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.insert(0, avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    pad_h = c_h-(i-1)*per_h
                    pad_w = c_w-(i-1)*per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2)+2*pad_h == H:
                        x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    else:
                        ep = H - (x_pre.size(2)+2*pad_h)
                        x_pad = F.pad(x_pre,(pad_h+ep,pad_h,pad_w+ep,pad_w),"constant",0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.insert(0, avgpool)
        return torch.cat(result, dim=2)

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # print(x.size())
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
            x = x.view(x.size(0), x.size(1))
        #x = self.classifier(x)
        return x

# Define the ResNet50-based part Model
class ft_net_LPN(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=6):
        super(ft_net_LPN, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        self.model = model_ft
        self.block = block
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # print(x.shape)
        if self.pool == 'avg+max':
            x1 = self.get_part_pool(x, pool='avg')
            x2 = self.get_part_pool(x, pool='max')
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.get_part_pool(x)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'max':
            x = self.get_part_pool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        #x = self.classifier(x)
        return x

    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1,1)) 
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        per_h, per_w = H/(2*self.block),W/(2*self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H+(self.block-c_h)*2, W+(self.block-c_w)*2
            x = nn.functional.interpolate(x, size=[new_H,new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H/2), int(W/2)
            per_h, per_w = H/(2*self.block),W/(2*self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                x_curr = x[:,:,(c_h-i*per_h):(c_h+i*per_h),(c_w-i*per_w):(c_w+i*per_w)]
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)] 
                    x_pad = F.pad(x_pre,(per_h,per_h,per_w,per_w),"constant",0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.append(avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    pad_h = c_h-(i-1)*per_h
                    pad_w = c_w-(i-1)*per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2)+2*pad_h == H:
                        x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    else:
                        ep = H - (x_pre.size(2)+2*pad_h)
                        x_pad = F.pad(x_pre,(pad_h+ep,pad_h,pad_w+ep,pad_w),"constant",0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.append(avgpool)
        return torch.cat(result, dim=2)

# For cvusa/cvact
class two_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride = 1, pool = 'avg', share_weight = False, VGG16=False, LPN=False, block=2):
        super(two_view_net, self).__init__()
        self.LPN = LPN
        self.block = block
        self.sqr = True # if the satellite image is square ring partition and the ground image is row partition, self.sqr is True. Otherwise it is False.
        if VGG16:
            if LPN:
                # satelite
                self.model_1 = ft_net_VGG16_LPN(class_num, stride=stride, pool=pool, block = block)
                if self.sqr:
                    self.model_1 = ft_net_VGG16_LPN_R(class_num, stride=stride, pool=pool, block=block)
            else:
                self.model_1 =  ft_net_VGG16(class_num, stride=stride, pool = pool)
                # self.vgg1 = models.vgg16_bn(pretrained=True)
                # self.model_1 = SAFA()
                # self.model_1 = SAFA_FC(64, 32, 8)
        else:
            #resnet50 LPN cvusa/cvact
            self.model_1 =  ft_net_cvusa_LPN(class_num, stride=stride, pool = pool, block=block)
            if self.sqr:
                self.model_1 = ft_net_cvusa_LPN_R(class_num, stride=stride, pool=pool, block=block)
            self.block = self.model_1.block
        if share_weight:
            self.model_2 = self.model_1
        else:
            if VGG16:
                if LPN:
                    #street
                    self.model_2 = ft_net_VGG16_LPN(class_num, stride=stride, pool=pool, block = block, row = self.sqr)
                else:
                    self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
                    # self.vgg2 = models.vgg16_bn(pretrained=True)
                    # self.model_2 = SAFA()
                    # self.model_2 = SAFA_FC(64, 32, 8)
            else:
                self.model_2 =  ft_net_cvusa_LPN(class_num, stride = stride, pool = pool, block=block, row = self.sqr)
        if LPN:
            if VGG16:
                if pool == 'avg+max':
                    for i in range(self.block):
                        name = 'classifier'+str(i)
                        setattr(self, name, ClassBlock(1024, class_num, droprate))
                else:
                    for i in range(self.block):
                        name = 'classifier'+str(i)
                        setattr(self, name, ClassBlock(512, class_num, droprate))
            else:
                if pool == 'avg+max':
                    for i in range(self.block):
                        name = 'classifier'+str(i)
                        setattr(self, name, ClassBlock(4096, class_num, droprate))
                else:
                    for i in range(self.block):
                        name = 'classifier'+str(i)
                        setattr(self, name, ClassBlock(2048, class_num, droprate))
        else:    
            self.classifier = ClassBlock(2048, class_num, droprate)
            if pool =='avg+max':
                self.classifier = ClassBlock(4096, class_num, droprate)
            if VGG16:
                self.classifier = ClassBlock(512, class_num, droprate)
                # self.classifier = ClassBlock(4096, class_num, droprate, num_bottleneck=512) #safa 情况下
                if pool =='avg+max':
                    self.classifier = ClassBlock(1024, class_num, droprate)

    def forward(self, x1, x2):
        if self.LPN:
            if x1 is None:
                y1 = None
            else:
                x1 = self.model_1(x1)
                y1 = self.part_classifier(x1)

            if x2 is None:
                y2 = None
            else:
                x2 = self.model_2(x2)
                y2 = self.part_classifier(x2)
        else:
            if x1 is None:
                y1 = None
            else:
                # x1 = self.vgg1.features(x1)
                x1 = self.model_1(x1)
                y1 = self.classifier(x1)

            if x2 is None:
                y2 = None
            else:
                # x2 = self.vgg2.features(x2)
                x2 = self.model_2(x2)
                y2 = self.classifier(x2)
        return y1, y2

    def part_classifier(self, x):
        part = {}
        predict = {}
        for i in range(self.block):
            # part[i] = torch.squeeze(x[:,:,i])
            part[i] = x[:,:,i].view(x.size(0),-1)
            name = 'classifier'+str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y
    
    
    
    
#==========================================================================================================================================
class ft_net_cvusa_LPN_R_swin(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=6):
        super(ft_net_cvusa_LPN_R_swin, self).__init__()
        configuration = SwinConfig(image_size=(256, 256), output_hidden_states=True)
        model = SwinModel.from_pretrained(pretrained_model_name_or_path='microsoft/swin-tiny-patch4-window7-224',
                                      config=configuration,
                                      ignore_mismatched_sizes=True,
                                      )
    

        self.pool = pool
        self.model = model
        self.block = block
        

    def forward(self, x):
        output, hidden_no_dowample = self.model(x)
        
        # x.size() torch.Size([150, 64, 768])
        x = output.last_hidden_state
        x = x.transpose(2, 1).reshape(-1, 768, 8, 8)

        if self.pool == 'avg+max':
            x1 = self.get_part_pool(x, pool='avg')
            x2 = self.get_part_pool(x, pool='max')
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.get_part_pool(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(1), -1)
            # print(x)
        elif self.pool == 'max':
            x = self.get_part_pool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        #x = self.classifier(x)
        return x

    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1,1)) 
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        per_h, per_w = H/(2*self.block),W/(2*self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H+(self.block-c_h)*2, W+(self.block-c_w)*2
            x = nn.functional.interpolate(x, size=[new_H,new_W], mode='bilinear')
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H/2), int(W/2)
            per_h, per_w = H/(2*self.block),W/(2*self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                x_curr = x[:,:,(c_h-i*per_h):(c_h+i*per_h),(c_w-i*per_w):(c_w+i*per_w)]
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)] 
                    x_pad = F.pad(x_pre,(per_h,per_h,per_w,per_w),"constant",0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.insert(0, avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    pad_h = c_h-(i-1)*per_h
                    pad_w = c_w-(i-1)*per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2)+2*pad_h == H:
                        x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    else:
                        ep = H - (x_pre.size(2)+2*pad_h)
                        x_pad = F.pad(x_pre,(pad_h+ep,pad_h,pad_w+ep,pad_w),"constant",0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.insert(0, avgpool)
        return torch.cat(result, dim=2)
    


class ft_net_cvusa_LPN_swin(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=6, row=False):
        super(ft_net_cvusa_LPN_swin, self).__init__()
        
        configuration = SwinConfig(image_size=(256, 256), output_hidden_states=True)
        model = SwinModel.from_pretrained(pretrained_model_name_or_path='microsoft/swin-tiny-patch4-window7-224',
                                      config=configuration,
                                      ignore_mismatched_sizes=True,
                                      )
        self.pool = pool
        self.model = model
        self.block = block
        self.avgpool = nn.AdaptiveAvgPool2d((1,block))
        self.maxpool = nn.AdaptiveMaxPool2d((1,block))
        if row:
            self.avgpool = nn.AdaptiveAvgPool2d((block,1))
            self.maxpool = nn.AdaptiveMaxPool2d((block,1))
        

    def forward(self, x):
        output, hidden_no_dowample = self.model(x)
        x = output.last_hidden_state
        x = x.transpose(2, 1).reshape(-1, 768, 8, 8)
        
        if self.pool == 'avg+max':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.avgpool(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(1), -1)
            # print(x)
        elif self.pool == 'max':
            x = self.maxpool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        #x = self.classifier(x)
        return x


    
    
class two_view_net_swin(nn.Module):
    def __init__(self, class_num, droprate, stride = 1, pool = 'avg', share_weight = False, VGG16=False, LPN=False, block=2):
        super(two_view_net_swin, self).__init__()
        self.LPN = LPN
        self.block = block
        self.sqr = True # if the satellite image is square ring partition and the ground image is row partition, self.sqr is True. Otherwise it is False.
        if VGG16:
            if LPN:
                # satelite
                self.model_1 = ft_net_VGG16_LPN(class_num, stride=stride, pool=pool, block = block)
                if self.sqr:
                    self.model_1 = ft_net_VGG16_LPN_R(class_num, stride=stride, pool=pool, block=block)
            else:
                self.model_1 =  ft_net_VGG16(class_num, stride=stride, pool = pool)
                # self.vgg1 = models.vgg16_bn(pretrained=True)
                # self.model_1 = SAFA()
                # self.model_1 = SAFA_FC(64, 32, 8)
        else:
            #resnet50 LPN cvusa/cvact
            self.model_1 =  ft_net_cvusa_LPN_swin(class_num, stride=stride, pool = pool, block=block)
            if self.sqr:
                self.model_1 = ft_net_cvusa_LPN_R_swin(class_num, stride=stride, pool=pool, block=block)
            self.block = self.model_1.block
        if share_weight:
            self.model_2 = self.model_1
        else:
            if VGG16:
                if LPN:
                    #street
                    self.model_2 = ft_net_VGG16_LPN(class_num, stride=stride, pool=pool, block = block, row = self.sqr)
                else:
                    self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
                    # self.vgg2 = models.vgg16_bn(pretrained=True)
                    # self.model_2 = SAFA()
                    # self.model_2 = SAFA_FC(64, 32, 8)
            else:
                self.model_2 =  ft_net_cvusa_LPN_swin(class_num, stride = stride, pool = pool, block=block, row = self.sqr)
        if LPN:
            if VGG16:
                if pool == 'avg+max':
                    for i in range(self.block):
                        name = 'classifier'+str(i)
                        setattr(self, name, ClassBlock(1024, class_num, droprate))
                else:
                    for i in range(self.block):
                        name = 'classifier'+str(i)
                        setattr(self, name, ClassBlock(512, class_num, droprate))
            else:
                if pool == 'avg+max':
                    for i in range(self.block):
                        name = 'classifier'+str(i)
                        setattr(self, name, ClassBlock(4096, class_num, droprate))
                else:
                    for i in range(self.block):
                        name = 'classifier'+str(i)
                       #setattr(self, name, ClassBlock(2048, class_num, droprate))
                        setattr(self, name, ClassBlock(768, class_num, droprate, num_bottleneck=512))
                        
                        
        else:    
            #self.classifier = ClassBlock(2048, class_num, droprate)
            self.classifier = ClassBlock(768, class_num, droprate, num_bottleneck=1024)
            if pool =='avg+max':
                self.classifier = ClassBlock(4096, class_num, droprate)
            if VGG16:
                self.classifier = ClassBlock(512, class_num, droprate)
                # self.classifier = ClassBlock(4096, class_num, droprate, num_bottleneck=512) #safa 情况下
                if pool =='avg+max':
                    self.classifier = ClassBlock(1024, class_num, droprate)

    def forward(self, x1, x2):
        if self.LPN:
            if x1 is None:
                y1 = None
            else:
                #torch.Size([150, 768, 4])
                x1 = self.model_1(x1)
                y1 = self.part_classifier(x1)

            if x2 is None:
                y2 = None
            else:
                x2 = self.model_2(x2)
                y2 = self.part_classifier(x2)
        else:
            if x1 is None:
                y1 = None
            else:
                # x1 = self.vgg1.features(x1)
                x1 = self.model_1(x1)
                y1 = self.classifier(x1)

            if x2 is None:
                y2 = None
            else:
                # x2 = self.vgg2.features(x2)
                x2 = self.model_2(x2)
                y2 = self.classifier(x2)
        return y1, y2

    def part_classifier(self, x):
        part = {}
        predict = {}
        for i in range(self.block):
            # part[i] = torch.squeeze(x[:,:,i])
            part[i] = x[:,:,i].view(x.size(0),-1)
            name = 'classifier'+str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y






#===============================================================
class ft_net_cvusa_LPN_R_swin_stage4(nn.Module):

    def __init__(self, class_num, model=None, droprate=0.5, stride=2, init_model=None, pool='avg', block=6):
        super(ft_net_cvusa_LPN_R_swin_stage4, self).__init__()

        self.pool = pool
        self.model = model
        self.block = block
        
    def forward(self, x):
        
        if self.model:
            B,C,H,W = x.size()
            x = x.reshape(B, C, -1).transpose(2, 1)
            x = self.model(x, input_dimensions=(H,H))           
            x = x[0].transpose(2, 1).reshape(B, C, H, H)
        else:
            x = x

        
        
        
        if self.pool == 'avg+max':
            x1 = self.get_part_pool(x, pool='avg')
            x2 = self.get_part_pool(x, pool='max')
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.get_part_pool(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(1), -1)
            # print(x)
        elif self.pool == 'max':
            x = self.get_part_pool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        #x = self.classifier(x)
        return x

    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1,1)) 
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        per_h, per_w = H/(2*self.block),W/(2*self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H+(self.block-c_h)*2, W+(self.block-c_w)*2
            x = nn.functional.interpolate(x, size=[new_H,new_W], mode='bilinear')
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H/2), int(W/2)
            per_h, per_w = H/(2*self.block),W/(2*self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                x_curr = x[:,:,(c_h-i*per_h):(c_h+i*per_h),(c_w-i*per_w):(c_w+i*per_w)]
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)] 
                    x_pad = F.pad(x_pre,(per_h,per_h,per_w,per_w),"constant",0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.insert(0, avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    pad_h = c_h-(i-1)*per_h
                    pad_w = c_w-(i-1)*per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2)+2*pad_h == H:
                        x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    else:
                        ep = H - (x_pre.size(2)+2*pad_h)
                        x_pad = F.pad(x_pre,(pad_h+ep,pad_h,pad_w+ep,pad_w),"constant",0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.insert(0, avgpool)
        return torch.cat(result, dim=2)
    
    
    
class ft_net_cvusa_LPN_swin_stage4(nn.Module):

    def __init__(self, class_num, model=None,droprate=0.5, stride=2, init_model=None, pool='avg', block=6, row=False):
        super(ft_net_cvusa_LPN_swin_stage4, self).__init__()
        
        self.pool = pool
        self.model = model
        self.block = block
        self.avgpool = nn.AdaptiveAvgPool2d((1,block))
        self.maxpool = nn.AdaptiveMaxPool2d((1,block))
        if row:
            self.avgpool = nn.AdaptiveAvgPool2d((block,1))
            self.maxpool = nn.AdaptiveMaxPool2d((block,1))
        

    def forward(self, x):
        if self.model:
            B,C,H,W = x.size()
            x = x.reshape(B, C, -1).transpose(2, 1)
            x = self.model(x, input_dimensions=(H,H))           
            x = x[0].transpose(2, 1).reshape(B, C, H, H)
        else:
            x = x

           
        if self.pool == 'avg+max':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.avgpool(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(1), -1)
            # print(x)
        elif self.pool == 'max':
            x = self.maxpool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        #x = self.classifier(x)
        return x
    
    
    
    

class two_view_net_swin_infonce(nn.Module):
    def __init__(self, class_num, droprate, stride = 1, pool = 'avg', share_weight = False, VGG16=False, LPN=False, block=2):
        super(two_view_net_swin_infonce, self).__init__()
        self.feature_dim = 768
        self.LPN = LPN
        self.block = block
        self.final_H = 16
        self.final_W = 16
        self.sqr = True # if the satellite image is square ring partition and the ground image is row partition, self.sqr is True. Otherwise it is False.
        configuration = SwinConfig(image_size=(256, 256), output_hidden_states=True)
        configuration_stage4 = SwinConfig(image_size=(256, 256))
        model1 = SwinModel.from_pretrained(pretrained_model_name_or_path='microsoft/swin-tiny-patch4-window7-224',
                                            config=configuration,
                                            ignore_mismatched_sizes=True,
                                            )
        model2 = SwinModel.from_pretrained(pretrained_model_name_or_path='microsoft/swin-tiny-patch4-window7-224',
                                            config=configuration,
                                            ignore_mismatched_sizes=True,
                                            )
        self.model_1 = model1
        self.model_2 = model2
        
        self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear')
        self.channel_adapter1 = nn.Sequential(nn.Conv2d(self.feature_dim, int(self.feature_dim/2), kernel_size=1, bias=False),
                                     nn.BatchNorm2d(int(self.feature_dim/2)),
                                     nn.ReLU(inplace=True))
        self.channel_adapter1.apply(weights_init_kaiming)

        self.channel_adapter2 = nn.Sequential(nn.Conv2d(self.feature_dim, int(self.feature_dim/2), kernel_size=1, bias=False),
                                nn.BatchNorm2d(int(self.feature_dim/2)),
                                nn.ReLU(inplace=True))
        self.channel_adapter2.apply(weights_init_kaiming)
        
        self.global_classifier = ClassBlock(self.feature_dim, class_num, droprate, return_f=True)
        
        resblock4_street = SwinStageLast(configuration_stage4)
        resblock4_sate = SwinStageLast(configuration_stage4)
        
        stage4_weight = self.model_1.encoder.layers[3]
        part_weight = dict()
        for name, para in stage4_weight.named_parameters(): 
            #print(name)   
            temp = name
            part_weight[temp] = para
        resblock4_street.load_state_dict(part_weight, strict=False)
        resblock4_sate.load_state_dict(part_weight, strict=False)
        
        self.part_stage4_sate = ft_net_cvusa_LPN_R_swin_stage4(class_num, model=None, stride=stride, pool=pool, block=block)
        self.part_stage4_street = ft_net_cvusa_LPN_swin_stage4(class_num, model=None,stride = stride, pool = pool, block=block, row = self.sqr)
            
        for i in range(self.block):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(768, class_num, droprate, num_bottleneck=512))
                        

    def forward(self, x1, x2):
        if self.LPN:
            if x1 is None:
                y1_global_logits=y1_embedding=y1_s4_part_logits = None
            else:
                #torch.Size([150, 768, 4])
                x1_output, x1_hidden_before_dsample = self.model_1(x1)
                x1_stage4 = x1_output.last_hidden_state
                #x1_stage3_concat = x1_stage4.transpose(2, 1).reshape(-1, self.feature_dim, int(self.final_W/2), int(self.final_H/2))
                x1_stage3 = x1_output[3][3]
                x1_stage3_before_dsample = x1_hidden_before_dsample[2]
                
                # global
                y1_global_logits, y1_embedding = self.global_classifier(x1_stage4.mean(dim=1))
                
                          
                # # part
                x1_stage3 = x1_stage3.transpose(2, 1).reshape(-1, self.feature_dim, int(self.final_W/2), int(self.final_H/2))
                x1_stage3_before_dsample = x1_stage3_before_dsample.transpose(2, 1).reshape(-1, int(self.feature_dim/2), self.final_W, self.final_H)
                x1_stage3 = self.upsample_layer(x1_stage3)
                x1_stage3 = self.channel_adapter1(x1_stage3)   
                x1_stage3_concat = torch.concat((x1_stage3, x1_stage3_before_dsample), dim=1)
                #print(x1_stage3_concat.size())
                #torch.Size([60, 768, 4])
                #x1_stage3_concat = x1_stage3
                #print(f'x1_stage3_concat {x1_stage3_concat.size()}')
                y1_s4_part = self.part_stage4_sate(x1_stage3_concat)
                y1_s4_part_logits = self.part_classifier(y1_s4_part)


            if x2 is None:
                y2_global_logits=y2_embedding=y2_s4_part_logits = None
            else:
                x2_output, x2_hidden_before_dsample = self.model_2(x2)
                x2_stage4 = x2_output.last_hidden_state
                x2_stage3_concat = x2_stage4.transpose(2, 1).reshape(-1, self.feature_dim, int(self.final_W/2), int(self.final_H/2))
                x2_stage3 = x2_output[3][3]
                x2_stage3_before_dsample = x2_hidden_before_dsample[2]
                
                # global
                y2_global_logits, y2_embedding = self.global_classifier(x2_stage4.mean(dim=1))
                
                # # part
                x2_stage3 = x2_stage3.transpose(2, 1).reshape(-1, self.feature_dim, int(self.final_W/2), int(self.final_H/2))
                x2_stage3_before_dsample = x2_stage3_before_dsample.transpose(2, 1).reshape(-1, int(self.feature_dim/2), self.final_W, self.final_H)
                x2_stage3 = self.upsample_layer(x2_stage3)
                x2_stage3 = self.channel_adapter2(x2_stage3)   
                x2_stage3_concat = torch.concat((x2_stage3, x2_stage3_before_dsample), dim=1)
                #x2_stage3_concat = x2_stage3
                y2_s4_part = self.part_stage4_street(x2_stage3_concat)
                y2_s4_part_logits = self.part_classifier(y2_s4_part)
                
            result = {'global_logits': (y1_global_logits, y2_global_logits),
                      'global_embedding':(y1_embedding, y2_embedding),
                      'part_logits': (y1_s4_part_logits, y2_s4_part_logits)
                     }
            
            # result = {
            #           'part_logits': (y1_s4_part_logits, y2_s4_part_logits)
            #          }

        
        return result

    def part_classifier(self, x):
        part = {}
        predict = {}
        for i in range(self.block):
            # part[i] = torch.squeeze(x[:,:,i])
            part[i] = x[:,:,i].view(x.size(0),-1)
            name = 'classifier'+str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y






#==============================================================================================
def init_weights_vit_timm(module: nn.Module, name: str = ''):
        """ ViT weight initialization, original timm impl (for reproducibility) """
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            print(f'Plpn linear is initialized')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, 'init_weights'):
            module.init_weights()
            
class CAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., part_num=0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.part_num = part_num

    def forward(self, lpn_q, x):
        
        # print(f"lpn_q  {lpn_q.size()}")
        # print(f"x  {x.size()}")
        # assert(0)
        # lpn_q  torch.Size([36, 4, 768])
        # x  torch.Size([36, 196, 768])
        
              
        B, N, C = x.shape
        #qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        #torch.Size([3, 8, 16, 197, 64])   其中8是bs    16是number heads
        #q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        # print(q.size())
        
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        lpn_q = lpn_q.reshape(B, self.part_num, self.num_heads, C // self.num_heads).transpose(2,1)
        # print(lpn_q.size())
        # assert(0)

        #attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (lpn_q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, self.part_num, C)
        # print(f"x  {x.size()}")
        # assert(0)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
    
class Plpn(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=2.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            part_num = 0
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, part_num=part_num)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        

    def forward(self, x, stage3):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), self.norm1(stage3))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        #x = self.norm3(x)
        return x
    
    
    
    
class Plpn_layer(nn.Module):
    def __init__(self,dim, layer_depth, num_heads=12, part_num = 0, block_fn=Plpn):
        super().__init__()
        self.depth = layer_depth
        self.pos_embed = nn.Parameter(torch.randn(1, part_num, dim) * .02)
        
        self.region_embed = nn.Embedding(part_num, dim)
        
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=dim,
                num_heads=num_heads,
                part_num=part_num            
            )
            for i in range(layer_depth)])        
               
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)
        
    
    def forward(self, lpn_query, stage3):
        bs = stage3.size(0)
        
        # query_embed 复制bs分   4 bs c
        query_embed = self.region_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        x = lpn_query + query_embed.transpose(1, 0)
        x = x + self.pos_embed
        
        #x = lpn_query
        
        for i in range(self.depth):
            x = self.blocks[i](x, stage3)
            
        return x
    
    
    
    
    
    
    
    
class two_view_net_swin_infonce_plpn(nn.Module):
    def __init__(self, class_num, droprate, stride = 1, pool = 'avg', share_weight = False, VGG16=False, LPN=False, block=2):
        super(two_view_net_swin_infonce_plpn, self).__init__()
        self.feature_dim = 768
        self.LPN = LPN
        self.block = block
        self.final_H = 16
        self.final_W = 16
        self.sqr = True # if the satellite image is square ring partition and the ground image is row partition, self.sqr is True. Otherwise it is False.
        configuration = SwinConfig(image_size=(256, 256), output_hidden_states=True)
        configuration_stage4 = SwinConfig(image_size=(256, 256))
        model1 = SwinModel.from_pretrained(pretrained_model_name_or_path='microsoft/swin-tiny-patch4-window7-224',
                                            config=configuration,
                                            ignore_mismatched_sizes=True,
                                            )
        model2 = SwinModel.from_pretrained(pretrained_model_name_or_path='microsoft/swin-tiny-patch4-window7-224',
                                            config=configuration,
                                            ignore_mismatched_sizes=True,
                                            )
        self.model_1 = model1
        self.model_2 = model2
        
        self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear')
        self.channel_adapter1 = nn.Sequential(nn.Conv2d(self.feature_dim, int(self.feature_dim/2), kernel_size=1, bias=False),
                                     nn.BatchNorm2d(int(self.feature_dim/2)),
                                     nn.ReLU(inplace=True))
        self.channel_adapter1.apply(weights_init_kaiming)

        self.channel_adapter2 = nn.Sequential(nn.Conv2d(self.feature_dim, int(self.feature_dim/2), kernel_size=1, bias=False),
                                nn.BatchNorm2d(int(self.feature_dim/2)),
                                nn.ReLU(inplace=True))
        self.channel_adapter2.apply(weights_init_kaiming)
        
        self.global_classifier = ClassBlock(self.feature_dim, class_num, droprate, return_f=True)
        
        resblock4_street = SwinStageLast(configuration_stage4)
        resblock4_sate = SwinStageLast(configuration_stage4)
        
        stage4_weight = self.model_1.encoder.layers[3]
        part_weight = dict()
        for name, para in stage4_weight.named_parameters(): 
            #print(name)   
            temp = name
            part_weight[temp] = para
        resblock4_street.load_state_dict(part_weight, strict=False)
        resblock4_sate.load_state_dict(part_weight, strict=False)
        
        self.plpn = Plpn_layer(self.feature_dim, layer_depth=4, num_heads=12, part_num = self.block)
        
        self.part_stage4_sate = ft_net_cvusa_LPN_R_swin_stage4(class_num, model=None, stride=stride, pool=pool, block=block)
        self.part_stage4_street = ft_net_cvusa_LPN_swin_stage4(class_num, model=None,stride = stride, pool = pool, block=block, row = self.sqr)
            
        for i in range(self.block):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(768, class_num, droprate, num_bottleneck=512))
                        

    def forward(self, x1, x2):
        if self.LPN:
            if x1 is None:
                y1_global_logits=y1_embedding=y1_s4_part_logits = None
            else:
                #torch.Size([150, 768, 4])
                x1_output, x1_hidden_before_dsample = self.model_1(x1)
                x1_stage4 = x1_output.last_hidden_state
                #x1_stage3_concat = x1_stage4.transpose(2, 1).reshape(-1, self.feature_dim, int(self.final_W/2), int(self.final_H/2))
                x1_stage3 = x1_output[3][3]
                x1_stage3_before_dsample = x1_hidden_before_dsample[2]
                
                # global
                y1_global_logits, y1_embedding = self.global_classifier(x1_stage4.mean(dim=1))
                
                          
                # # part
                x1_stage3 = x1_stage3.transpose(2, 1).reshape(-1, self.feature_dim, int(self.final_W/2), int(self.final_H/2))
                x1_stage3_before_dsample = x1_stage3_before_dsample.transpose(2, 1).reshape(-1, int(self.feature_dim/2), self.final_W, self.final_H)
                x1_stage3 = self.upsample_layer(x1_stage3)
                x1_stage3 = self.channel_adapter1(x1_stage3)   
                x1_stage3_concat = torch.concat((x1_stage3, x1_stage3_before_dsample), dim=1)

                #print(x1_stage3_concat.size())
                #torch.Size([60, 768, 4])
                #x1_stage3_concat = x1_stage3
                #print(f'x1_stage3_concat {x1_stage3_concat.size()}')
                y1_s4_part = self.part_stage4_sate(x1_stage3_concat)
                y1_s4_part = y1_s4_part.transpose(2,1)
                B,C,H,W = x1_stage3_concat.size()
                x1_stage3_concat = x1_stage3_concat.reshape(B,C,-1).transpose(2,1)
                y1_s4_part = self.plpn(y1_s4_part, x1_stage3_concat)
                y1_s4_part_logits = self.part_classifier(y1_s4_part.transpose(2,1))


            if x2 is None:
                y2_global_logits=y2_embedding=y2_s4_part_logits = None
            else:
                x2_output, x2_hidden_before_dsample = self.model_2(x2)
                x2_stage4 = x2_output.last_hidden_state
                x2_stage3_concat = x2_stage4.transpose(2, 1).reshape(-1, self.feature_dim, int(self.final_W/2), int(self.final_H/2))
                x2_stage3 = x2_output[3][3]
                x2_stage3_before_dsample = x2_hidden_before_dsample[2]
                
                # global
                y2_global_logits, y2_embedding = self.global_classifier(x2_stage4.mean(dim=1))
                
                # # part
                x2_stage3 = x2_stage3.transpose(2, 1).reshape(-1, self.feature_dim, int(self.final_W/2), int(self.final_H/2))
                x2_stage3_before_dsample = x2_stage3_before_dsample.transpose(2, 1).reshape(-1, int(self.feature_dim/2), self.final_W, self.final_H)
                x2_stage3 = self.upsample_layer(x2_stage3)
                x2_stage3 = self.channel_adapter2(x2_stage3)   
                x2_stage3_concat = torch.concat((x2_stage3, x2_stage3_before_dsample), dim=1)
                
                #x2_stage3_concat = x2_stage3
                y2_s4_part = self.part_stage4_street(x2_stage3_concat)
                B,C,H,W = x2_stage3_concat.size()
                x2_stage3_concat = x2_stage3_concat.reshape(B,C,-1).transpose(2,1)
                y2_s4_part = y2_s4_part.transpose(2,1)
                y2_s4_part = self.plpn(y2_s4_part, x2_stage3_concat)                
                y2_s4_part_logits = self.part_classifier(y2_s4_part.transpose(2,1))
                
            result = {'global_logits': (y1_global_logits, y2_global_logits),
                      'global_embedding':(y1_embedding, y2_embedding),
                      'part_logits': (y1_s4_part_logits, y2_s4_part_logits)
                     }
            
            # result = {
            #           'part_logits': (y1_s4_part_logits, y2_s4_part_logits)
            #          }

        
        return result

    def part_classifier(self, x):
        part = {}
        predict = {}
        for i in range(self.block):
            # part[i] = torch.squeeze(x[:,:,i])
            part[i] = x[:,:,i].view(x.size(0),-1)
            name = 'classifier'+str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y



class two_view_net_swinB_infonce(nn.Module):
    def __init__(self, class_num, droprate, stride = 1, pool = 'avg', share_weight = False, VGG16=False, LPN=False, block=2):
        super(two_view_net_swinB_infonce, self).__init__()
        self.feature_dim = 1024
        self.LPN = LPN
        self.block = block
        self.final_H = 16
        self.final_W = 16
        self.sqr = True # if the satellite image is square ring partition and the ground image is row partition, self.sqr is True. Otherwise it is False.

        configuration = SwinConfig.from_pretrained("microsoft/swin-base-patch4-window7-224", output_hidden_states=True)
        configuration_stage4 = SwinConfig.from_pretrained("microsoft/swin-base-patch4-window7-224")
        
        model1 = SwinModel.from_pretrained(pretrained_model_name_or_path="microsoft/swin-base-patch4-window7-224", config=configuration)
        model2 = SwinModel.from_pretrained(pretrained_model_name_or_path="microsoft/swin-base-patch4-window7-224", config=configuration)

        self.model_1 = model1
        self.model_2 = model2
        
        self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear')
        self.channel_adapter1 = nn.Sequential(nn.Conv2d(self.feature_dim, int(self.feature_dim/2), kernel_size=1, bias=False),
                                     nn.BatchNorm2d(int(self.feature_dim/2)),
                                     nn.ReLU(inplace=True))
        self.channel_adapter1.apply(weights_init_kaiming)

        self.channel_adapter2 = nn.Sequential(nn.Conv2d(self.feature_dim, int(self.feature_dim/2), kernel_size=1, bias=False),
                                nn.BatchNorm2d(int(self.feature_dim/2)),
                                nn.ReLU(inplace=True))
        self.channel_adapter2.apply(weights_init_kaiming)
        
        self.global_classifier = ClassBlock(self.feature_dim, class_num, droprate, return_f=True)
        
        resblock4_street = SwinStageLast(configuration_stage4,dim=self.feature_dim, num_heads=32)
        resblock4_sate = SwinStageLast(configuration_stage4,dim=self.feature_dim, num_heads=32)
        
        stage4_weight = self.model_1.encoder.layers[3]
        part_weight = dict()
        for name, para in stage4_weight.named_parameters(): 
            #print(name)   
            temp = name
            part_weight[temp] = para
        resblock4_street.load_state_dict(part_weight, strict=False)
        resblock4_sate.load_state_dict(part_weight, strict=False)
        
        self.part_stage4_sate = ft_net_cvusa_LPN_R_swin_stage4(class_num, model=None, stride=stride, pool=pool, block=block)
        self.part_stage4_street = ft_net_cvusa_LPN_swin_stage4(class_num, model=None,stride = stride, pool = pool, block=block, row = self.sqr)
            
        for i in range(self.block):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(self.feature_dim, class_num, droprate, num_bottleneck=512))
                        

    def forward(self, x1, x2):
        if self.LPN:
            if x1 is None:
                y1_global_logits=y1_embedding=y1_s4_part_logits = None
            else:
                #torch.Size([150, 768, 4])
                x1_output, x1_hidden_before_dsample = self.model_1(x1)
                x1_stage4 = x1_output.last_hidden_state
                

                #x1_stage3_concat = x1_stage4.transpose(2, 1).reshape(-1, self.feature_dim, int(self.final_W/2), int(self.final_H/2))
                x1_stage3 = x1_output[3][3] # bs 1024 8 8
                x1_stage3_before_dsample = x1_hidden_before_dsample[2]
                
                # global
                y1_global_logits, y1_embedding = self.global_classifier(x1_stage4.mean(dim=1))
                
                          
                # # part
                x1_stage3 = x1_stage3.transpose(2, 1).reshape(-1, self.feature_dim, int(self.final_W/2), int(self.final_H/2))
                x1_stage3_before_dsample = x1_stage3_before_dsample.transpose(2, 1).reshape(-1, int(self.feature_dim/2), self.final_W, self.final_H)
                x1_stage3 = self.upsample_layer(x1_stage3)
                x1_stage3 = self.channel_adapter1(x1_stage3)   
                x1_stage3_concat = torch.concat((x1_stage3, x1_stage3_before_dsample), dim=1)
                #print(x1_stage3_concat.size())
                #torch.Size([60, 768, 4])
                #x1_stage3_concat = x1_stage3
                #print(f'x1_stage3_concat {x1_stage3_concat.size()}')
                y1_s4_part = self.part_stage4_sate(x1_stage3_concat)
                y1_s4_part_logits = self.part_classifier(y1_s4_part)


            if x2 is None:
                y2_global_logits=y2_embedding=y2_s4_part_logits = None
            else:
                x2_output, x2_hidden_before_dsample = self.model_2(x2)
                x2_stage4 = x2_output.last_hidden_state
                x2_stage3_concat = x2_stage4.transpose(2, 1).reshape(-1, self.feature_dim, int(self.final_W/2), int(self.final_H/2))
                x2_stage3 = x2_output[3][3]
                x2_stage3_before_dsample = x2_hidden_before_dsample[2]
                
                # global
                y2_global_logits, y2_embedding = self.global_classifier(x2_stage4.mean(dim=1))
                
                # # part
                x2_stage3 = x2_stage3.transpose(2, 1).reshape(-1, self.feature_dim, int(self.final_W/2), int(self.final_H/2))
                x2_stage3_before_dsample = x2_stage3_before_dsample.transpose(2, 1).reshape(-1, int(self.feature_dim/2), self.final_W, self.final_H)
                x2_stage3 = self.upsample_layer(x2_stage3)
                x2_stage3 = self.channel_adapter2(x2_stage3)   
                x2_stage3_concat = torch.concat((x2_stage3, x2_stage3_before_dsample), dim=1)
                #x2_stage3_concat = x2_stage3
                y2_s4_part = self.part_stage4_street(x2_stage3_concat)
                y2_s4_part_logits = self.part_classifier(y2_s4_part)
                
            result = {'global_logits': (y1_global_logits, y2_global_logits),
                      'global_embedding':(y1_embedding, y2_embedding),
                      'part_logits': (y1_s4_part_logits, y2_s4_part_logits)
                     }
            
            # result = {
            #           'part_logits': (y1_s4_part_logits, y2_s4_part_logits)
            #          }

        
        return result

    def part_classifier(self, x):
        part = {}
        predict = {}
        for i in range(self.block):
            # part[i] = torch.squeeze(x[:,:,i])
            part[i] = x[:,:,i].view(x.size(0),-1)
            name = 'classifier'+str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y
#==========================================================================================================================================











class three_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False, VGG16=False, LPN=False, block=6):
        super(three_view_net, self).__init__()
        self.LPN = LPN
        self.block = block
        if VGG16:
            self.model_1 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
            self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
        elif LPN:
            self.model_1 =  ft_net_LPN(class_num, stride = stride, pool = pool, block = block)
            self.model_2 =  ft_net_LPN(class_num, stride = stride, pool = pool, block = block)
            # self.block = self.model_1.block
        else: 
            self.model_1 =  ft_net(class_num, stride = stride, pool = pool)
            self.model_2 =  ft_net(class_num, stride = stride, pool = pool)

        if share_weight:
            self.model_3 = self.model_1
        else:
            if VGG16:
                self.model_3 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
            elif LPN:
                self.model_3 =  ft_net_LPN(class_num, stride = stride, pool = pool, block = block)
            else:
                self.model_3 =  ft_net(class_num, stride = stride, pool = pool)
        if LPN:
            if pool == 'avg+max':
                for i in range(self.block):
                    name = 'classifier'+str(i)
                    setattr(self, name, ClassBlock(4096, class_num, droprate))
            else:
                for i in range(self.block):
                    name = 'classifier'+str(i)
                    setattr(self, name, ClassBlock(2048, class_num, droprate))
        else:    
            self.classifier = ClassBlock(2048, class_num, droprate)
            if pool =='avg+max':
                self.classifier = ClassBlock(4096, class_num, droprate)

    def forward(self, x1, x2, x3, x4 = None): # x4 is extra data
        if self.LPN:
            if x1 is None:
                y1 = None
            else:
                x1 = self.model_1(x1)
                y1 = self.part_classifier(x1)

            if x2 is None:
                y2 = None
            else:
                x2 = self.model_2(x2)
                y2 = self.part_classifier(x2)

            if x3 is None:
                y3 = None
            else:
                x3 = self.model_3(x3)
                y3 = self.part_classifier(x3)

            if x4 is None:
                return y1, y2, y3
            else:
                x4 = self.model_2(x4)
                y4 = self.part_classifier(x4)
                return y1, y2, y3, y4
        else:
            if x1 is None:
                y1 = None
            else:
                x1 = self.model_1(x1)
                y1 = self.classifier(x1)

            if x2 is None:
                y2 = None
            else:
                x2 = self.model_2(x2)
                y2 = self.classifier(x2)

            if x3 is None:
                y3 = None
            else:
                x3 = self.model_3(x3)
                y3 = self.classifier(x3)

            if x4 is None:
                return y1, y2, y3
            else:
                x4 = self.model_2(x4)
                y4 = self.classifier(x4)
                return y1, y2, y3, y4

    def part_classifier(self, x):
        part = {}
        predict = {}
        for i in range(self.block):
            part[i] = x[:,:,i].view(x.size(0),-1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = two_view_net(701, droprate=0.5, pool='avg', stride=1, VGG16=False, LPN=True, block=8)

    # net = three_view_net(701, droprate=0.5, pool='avg', stride=1, share_weight=True, LPN=True, block=2)
    # net.eval()

    # net = ft_net_VGG16_LPN_R(701)
    # net = ft_net_cvusa_LPN(701, stride=1)
    # net = ft_net(701)

    print(net)

    input = Variable(torch.FloatTensor(2, 3, 256, 256))
    output1,output2 = net(input,input)
    # output1,output2,output3 = net(input,input,input)
    # output1 = net(input)
    # print('net output size:')
    # print(output1.shape)
    # print(output.shape)
    for i in range(len(output1)):
        print(output1[i].shape)
    # x = torch.randn(2,512,8,8)
    # x_shape = x.shape
    # pool = AzimuthPool2d(x_shape, 8)
    # out = pool(x)
    # print(out.shape)
