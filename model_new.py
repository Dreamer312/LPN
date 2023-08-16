import argparse
import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
# from swin import SwinConfig, SwinModel, SwinStageLast
from transformers import SwinConfig, SwinModel
from timm.models.layers import  Mlp, DropPath, trunc_normal_
from einops import rearrange

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

def init_weights_vit_timm(module: nn.Module, name: str = ''):
        """ ViT weight initialization, original timm impl (for reproducibility) """
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, 'init_weights'):
            module.init_weights()



class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False, classifier=False):
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

        if classifier:
            classifier = []
            classifier += [nn.Linear(num_bottleneck, class_num)]
            classifier = nn.Sequential(*classifier)
            classifier.apply(weights_init_classifier)
        else:
            classifier = nn.Identity()

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
        

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    


class UPSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., part_num=0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

       
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.part_num = part_num
        self.softmax = nn.Softmax(dim=-1)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        

    def forward(self, lpn_q, x, epoch=None):
                   
        B, N, C = x.shape  
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
    
        lpn_q = lpn_q.reshape(B, self.part_num, self.num_heads, C // self.num_heads).transpose(2,1)
        attn = (F.normalize(lpn_q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))



        # mask_ratio = 0.05
        # if epoch is not None:
        #     if epoch <= 79 and self.training:
        #         m_r = torch.ones_like(attn) * mask_ratio
        #         attn = attn + torch.bernoulli(m_r) * -1e12

            
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, self.part_num, C)

        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class ssMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        H=3
        W=3
        x = self.fc1(x)
        #print(x.size())  #torch.Size([32, 9, 4096])
        #assert(0)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class UQPT(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            part_num = 0,
            conv_ffn = True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = UPSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, part_num=part_num)

        #self.attn = UPSA_dot(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, part_num=part_num)

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)


        if conv_ffn:
            self.mlp = ssMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        

    def forward(self, x, stage3, epoch):


        if epoch is not None:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), self.norm1(stage3), epoch)))
            # print(epoch)
        else:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), self.norm1(stage3))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
    

class UQPT_layer(nn.Module):
    def __init__(self,dim, layer_depth, num_heads=12, part_num = 0, block_fn=UQPT):
        super().__init__()
        self.depth = layer_depth
        self.pos_embed = nn.Parameter(torch.randn(1, part_num, dim) * .02)
        
        self.region_embed = nn.Embedding(part_num, dim)
        
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=dim,
                num_heads=num_heads,
                part_num=part_num,
                drop_path=0.1,
        
            )
            for i in range(layer_depth)])        
               
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)
        
    
    def forward(self, lpn_query, stage3, epoch=None):
        bs = stage3.size(0)
        
        # query_embed 复制bs分   4 bs c
        query_embed = self.region_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        x = lpn_query + query_embed.transpose(1, 0)
        x = x + self.pos_embed
        
        #如果不需要位置信息的话
        #x = lpn_query
        
        for i in range(self.depth):
            x = self.blocks[i](x, stage3, epoch)
            
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim  

class Cluster(nn.Module):
    def __init__(self, dim, out_dim, proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24,
                 return_center=False):
        """

        :param dim:  channel nubmer
        :param out_dim: channel nubmer
        :param proposal_w: the sqrt(proposals) value, we can also set a different value
        :param proposal_h: the sqrt(proposals) value, we can also set a different value
        :param fold_w: the sqrt(number of regions) value, we can also set a different value
        :param fold_h: the sqrt(number of regions) value, we can also set a different value
        :param heads:  heads number in context cluster
        :param head_dim: dimension of each head in context cluster
        :param return_center: if just return centers instead of dispatching back (deprecated).
        """
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for similarity
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)  # for projecting channel number
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for value
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.return_center = return_center

        



    def forward(self, x):  # [b,c,w,h]
        value = self.v(x) # value torch.Size([32, 768, 20, 20])
        x = self.f(x) # x torch.Size([32, 768, 20, 20])

        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads) # x torch.Size([384, 64, 20, 20])

        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)# value torch.Size([384, 64, 20, 20])

        
        if self.fold_w > 1 and self.fold_h > 1:
            # split the big feature maps to small local regions to reduce computations.
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                          f2=self.fold_h)  # [bs*blocks,c,ks[0],ks[1]]
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
        b, c, w, h = x.shape
        centers = self.centers_proposal(x)  # [b,c,C_W,C_H], we set M = C_W*C_H and N = w*h  centers torch.Size([384, 64, 3, 3])
        value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')  # [b,C_W,C_H,c]  value_centers torch.Size([384, 9, 64])


        b, c, ww, hh = centers.shape
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )  # [B,M,N]
        # we use mask to sololy assign each point to one center
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)  # binary #[B,M,N]
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c w h -> b (w h) c')  # [B,N,D]
        # aggregate step, out shape [B,M,D]
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                    sim.sum(dim=-1, keepdim=True) + 1.0)  # [B,M,D]

        if self.return_center:
            out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        else:
            # dispatch step, return to each point in a cluster
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
            out = rearrange(out, "b (w h) c -> b c w h", w=w)

        if self.fold_w > 1 and self.fold_h > 1:
            # recover the splited regions back to big feature maps if use the region partition.
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        return out

class RegionCenter(nn.Module):
    def __init__(self, dim, head_nums=12, num_tokens_out=4):
        super().__init__()
        window_size = int(num_tokens_out ** 0.5)
        self.cluster = Cluster(dim=dim, out_dim=dim, proposal_w=window_size, proposal_h=window_size,
                                fold_w=1, fold_h=1, heads=head_nums, head_dim=64, return_center=True)
     
    def forward(self, x):
        regions = self.cluster(x)
        # print(f"regions {regions.size()}")
        # assert(0)
        return regions



class swin_infonce_region_cluster(nn.Module):
    def __init__(self, class_num, droprate, stride = 1, pool = 'avg', share_weight = False, VGG16=False, LPN=False, block=2, model="swint", class_dim=256, dataset="vigor"):
        super(swin_infonce_region_cluster, self).__init__()

        self.feature_dim = 768
        if model == "swinb":
            self.feature_dim = 1024
        
        self.LPN = LPN
        self.block = block
             

        if dataset=="vigor":
            self.final_H = 20
            self.final_W = 20
            self.final_H_street = 20
            self.final_W_street = 40
        else:
            self.final_H = 16
            self.final_W = 16
            self.final_H_street = 8
            self.final_W_street = 32



        if model == "swinb":
            head_nums_region = 16
            #* SwinB
            configuration = SwinConfig.from_pretrained("microsoft/swin-base-patch4-window7-224", output_hidden_states=True)
        
            model1 = SwinModel.from_pretrained(pretrained_model_name_or_path="microsoft/swin-base-patch4-window7-224", config=configuration)
            model2 = SwinModel.from_pretrained(pretrained_model_name_or_path="microsoft/swin-base-patch4-window7-224", config=configuration)


        else:
            head_nums_region = 12
            if dataset=="vigor":
                configuration1 = SwinConfig(output_hidden_states=True)
                configuration2 = SwinConfig(output_hidden_states=True)
            else:
                configuration1 = SwinConfig(image_size=(128, 512), output_hidden_states=True)
                configuration2 = SwinConfig(image_size=(256, 256), output_hidden_states=True)

            
            model1 = SwinModel.from_pretrained(pretrained_model_name_or_path='microsoft/swin-tiny-patch4-window7-224',
                                                config=configuration2,
                                                ignore_mismatched_sizes=True,
                                                )
            model2 = SwinModel.from_pretrained(pretrained_model_name_or_path='microsoft/swin-tiny-patch4-window7-224',
                                                config=configuration1,
                                                ignore_mismatched_sizes=True,
                                                )




        self.model_1 = model1
        self.model_2 = model2
        
        # self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.channel_adapter1 = nn.Sequential(nn.Conv2d(self.feature_dim, int(self.feature_dim/2), kernel_size=1, bias=False),
        #                              nn.BatchNorm2d(int(self.feature_dim/2)),
        #                              nn.ReLU(inplace=True))
        # self.channel_adapter1.apply(weights_init_kaiming)

        # self.channel_adapter2 = nn.Sequential(nn.Conv2d(self.feature_dim, int(self.feature_dim/2), kernel_size=1, bias=False),
        #                         nn.BatchNorm2d(int(self.feature_dim/2)),
        #                         nn.ReLU(inplace=True))
        # self.channel_adapter2.apply(weights_init_kaiming)




        # 4.5M参数
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.feature_dim, self.feature_dim, kernel_size=2, stride=2),
            Norm2d(self.feature_dim),
            nn.GELU(),
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.feature_dim, self.feature_dim, kernel_size=2, stride=2),
            Norm2d(self.feature_dim),
            nn.GELU(),
        )
        
        self.global_classifier = ClassBlock(self.feature_dim, class_num, droprate, return_f=True, num_bottleneck=class_dim, classifier=True)
        
        # 27 M 参数
        self.plpn = UQPT_layer(self.feature_dim, layer_depth=1, num_heads=head_nums_region, part_num = self.block)
        
    
        self.parts = RegionCenter(dim=self.feature_dim, num_tokens_out=block)
        self.parts_street = RegionCenter(dim=self.feature_dim, num_tokens_out=block)
            
        for i in range(self.block):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(self.feature_dim, class_num, droprate, num_bottleneck=class_dim))
                        

    def forward(self, x1, x2, epoch=None):
        if self.LPN:
            if x1 is None:
                y1_global_logits=y1_embedding=y1_s4_part_logits = None
            else:
                x1_output = self.model_1(x1)
                x1_stage4 = x1_output.last_hidden_state #x1_stage4 torch.Size([16, 100, 768])              
                # global
                y1_global_logits, y1_embedding = self.global_classifier(x1_stage4.mean(dim=1))              
                x1_stage4 = rearrange(x1_stage4, "b (h w) c -> b c h w", h=10) 
                x1_stage4 = self.fpn1(x1_stage4) # torch.Size([16, 768, 40, 40])
                y1_s4_part = self.parts(x1_stage4) # y1_s4_part torch.Size([16, 768, 3, 3])
                y1_s4_part = rearrange(y1_s4_part, "b c h w -> b (h w) c")
                x1_stage4 = rearrange(x1_stage4, "b c h w -> b (h w) c", h=20) 
                y1_s4_part = self.plpn(y1_s4_part, x1_stage4, epoch) #torch.Size([16, 9, 768])
                y1_s4_part_logits = self.part_classifier(y1_s4_part.transpose(2,1))
                # print(y1_s4_part_logits.size())
                # assert(0)
                



            if x2 is None:
                y2_global_logits=y2_embedding=y2_s4_part_logits = None
            else:

                x2_output = self.model_2(x2)
                x2_stage4 = x2_output.last_hidden_state
                y2_global_logits, y2_embedding = self.global_classifier(x2_stage4.mean(dim=1))
                x2_stage4 = rearrange(x2_stage4, "b (h w) c -> b c h w", h=10, w=20) 
                x2_stage4 = self.fpn2(x2_stage4)


                y2_s4_part = self.parts_street(x2_stage4)
                y2_s4_part = rearrange(y2_s4_part, "b c h w -> b (h w) c")
                x2_stage4 = rearrange(x2_stage4, "b c h w -> b (h w) c", h=20, w=40)
                y2_s4_part = self.plpn(y2_s4_part, x2_stage4, epoch)
                y2_s4_part_logits = self.part_classifier(y2_s4_part.transpose(2,1))
              
            result = {'global_logits': (y1_global_logits, y2_global_logits),
                      'global_embedding':(y1_embedding, y2_embedding),
                      'part_logits': (y1_s4_part_logits, y2_s4_part_logits)
                     }
                   
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