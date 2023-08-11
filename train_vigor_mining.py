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
# os.environ["WANDB_MODE"] = "disabled"
from model import two_view_net_swin_infonce_plpn2, two_view_net_swinB_infonce, two_view_net_swin_infonce_region_cluster, swin_infonce_region_cluster_sr
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
from accelerate import Accelerator
from accelerate.utils import set_seed
# torch.cuda.empty_cache()
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from torch.utils.data.dataset import Dataset
from typing import TypeVar, Optional, Iterator
import math
T_co = TypeVar('T_co', covariant=True)
version = torch.__version__

 
######################################################################
# Options
# --------
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

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


class DistributedMiningSamplerVigor(DistributedSampler[T_co]):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = False,
                 seed: int = 0, drop_last: bool = False, batch_size: int = 128, mode = 'similarity', dim=1000, save_path=None) -> None:
        super(DistributedMiningSamplerVigor, self).__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.dim = dim
        self.batch_size = batch_size * self.num_replicas
        self.queue_length = max(dataset.train_data_size, len(dataset.train_sat_cover_list))
        self.current_size = len(self.dataset) // self.batch_size * self.batch_size
        self.current_indices = np.arange(self.current_size)
        self.queue_size = 1 # for computing moving average, not used in this implementation
        self.queue = np.zeros([self.queue_length, self.queue_size, self.dim, 2])
        self.queue_ptr = 0
        self.queue_counter = np.zeros(self.queue_length,dtype=np.int64)
        self.save_path = save_path
        self.mining_start = 1
        self.mining_pool_size = min(40000, len(dataset.train_sat_cover_list))
        self.mining_save_size = 100
        self.choice_pool = range(self.mining_save_size)
        self.mining_save = np.zeros([self.queue_length, self.mining_save_size],dtype=int)
        self.mode = mode
        # raise Exception
        print("Batch size per GPU:", batch_size)
        print("Number of GPUs (replicas):", self.num_replicas)
        print("Global batch size:", self.batch_size)
        print(f'self.queue_length {self.queue_length}')
        print(f'self.current_size {self.current_size}') 

    def update(self, data_sat, data_grd, indexes):
        data_sat_norm = data_sat / np.linalg.norm(data_sat, axis=1, keepdims=True)
        data_grd_norm = data_grd / np.linalg.norm(data_grd, axis=1, keepdims=True)
        batch_size = data_sat.shape[0]
        #print(f'这个是gather之后的data_sat.shape[0] {batch_size}')  GPU数量*每张卡的batch size

       

        # writing in distributed training style, complicated. Update the queue according to the previous index.
        for j in range(self.num_replicas):
            #print(f'self.num_replicas {self.num_replicas}') # =GPUs数量
            index_j = self.indices_out[j:self.current_size:self.num_replicas]

            # print(f'index_j {len(index_j)}')
            # print(f'index_j {index_j[:10]}')
            # print("Entering loop with batch_size:", batch_size, "and num_replicas:", self.num_replicas)
            

            # print(indexes)
            # assert(0)


            for i in range(batch_size // self.num_replicas):


                # calculated_index = indexes[i + j * (batch_size // self.num_replicas)]
                # target_list = self.dataset.train_sat_cover_dict[self.dataset.train_sat_cover_list[index]]
                # #print("Calculated index:", calculated_index)
                # print("Target list:", target_list)
                # assert calculated_index in target_list



                index = index_j[self.queue_ptr + i] %len(self.dataset.train_sat_cover_list)
                assert indexes[i + j * (batch_size // self.num_replicas)] in self.dataset.train_sat_cover_dict[self.dataset.train_sat_cover_list[index]]
                self.queue[index, self.queue_counter[index] % self.queue_size, :, 0] = \
                data_sat_norm[i + j * (batch_size // self.num_replicas)]
                self.queue[indexes[i + j * (batch_size // self.num_replicas)], self.queue_counter[index] % self.queue_size, :, 1] = \
                    data_grd_norm[i + j * (batch_size // self.num_replicas)]
                self.queue_counter[index] += 1
            self.queue_ptr = (self.queue_ptr + batch_size // self.num_replicas)

    def generate_indices_sim(self):
        self.queue_ptr = 0

        random.seed(7 + self.epoch)
        self.current_indices = np.arange(self.current_size) %len(self.dataset.train_sat_cover_list)
        random.shuffle(self.current_indices)

        if self.epoch >= self.mining_start:
            assert self.mining_pool_size <= self.queue_length
            mining_pool = np.array(random.sample(range(len(self.dataset.train_sat_cover_list)), self.mining_pool_size),dtype=int)
            product_train = np.matmul(self.queue[:,:,:,1].mean(axis=1), np.transpose(self.queue[mining_pool,:,:,0].mean(axis=1)))
            product_index = np.argsort(product_train, axis=1)
            # update mining pool
            for i in range(product_train.shape[0]):
                self.mining_save[i, :] = mining_pool[product_index[i, -self.mining_save_size:]]
            # randomly sample the first half
            ori_list = self.current_indices[:self.current_size//2]
            self.current_indices = []
            # global hard mining for the other half
            for i in range(self.current_size//self.batch_size):
                index_s = i * (self.batch_size//2)
                index_e = index_s + min(self.batch_size//2, self.current_size//2 - index_s)
                self.current_indices.extend(ori_list[index_s:index_e])
                hard_list = []
                for j in range(index_s, index_e):
                    grd_id = random.choice(self.dataset.train_sat_cover_dict[self.dataset.train_sat_cover_list[ori_list[j]]])
                    idx = int(random.choice(self.mining_save[grd_id]))
                    # keep random sampling until there is no overlap in the batch, hard coded as VIGOR is complicated
                    while True:
                        flag = False
                        for grd_idx in self.dataset.train_sat_cover_dict[self.dataset.train_sat_cover_list[idx]]:
                            if not self.dataset.check_overlap(ori_list[index_s:index_e],
                                                          grd_idx) or not self.dataset.check_overlap(hard_list, grd_idx):
                                flag = True
                        if flag:
                            idx = random.choice(self.mining_save[grd_id])
                        else:
                            break
                    hard_list.append(idx)
                self.current_indices.extend(hard_list)
        self.current_indices = np.array(self.current_indices, dtype=int)
        assert len(self.current_indices) == self.current_size
        print('sampler updated!')

    def update_epoch(self):
        # if self.epoch >= self.mining_start:
        self.generate_indices_sim()
        if self.rank == 0:
            np.save(os.path.join(self.save_path, 'queue.npy'), self.queue)
            np.save(os.path.join(self.save_path, 'queue_counter.npy'), self.queue_counter)

    def load(self, path):
        self.mining_start = 0
        self.queue_counter = np.load(os.path.join(path, 'queue_counter.npy'))
        self.queue = np.load(os.path.join(path, 'queue.npy'))

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.current_indices), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.current_indices)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.current_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.current_size]
        assert len(indices) == self.current_size

        # subsample
        self.indices_out = self.current_indices[indices].tolist()
        indices = indices[self.rank:self.current_size:self.num_replicas]
        # assert len(indices) == self.num_samples
        # print(indices)
        indices_out = self.current_indices[indices].tolist()
        # print(self.rank, len(indices), len(indices_out))

        return iter(indices_out)


class DistributedMiningSamplerVigorDelay(DistributedSampler[T_co]):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = False,
                 seed: int = 0, drop_last: bool = False, batch_size: int = 128, mode = 'similarity', dim=1000, save_path=None, accelerate=None) -> None:
        super(DistributedMiningSamplerVigorDelay, self).__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.dim = dim
        self.batch_size = batch_size * self.num_replicas
        self.queue_length = max(dataset.train_data_size, len(dataset.train_sat_cover_list))
        self.current_size = len(self.dataset) // self.batch_size * self.batch_size
        self.current_indices = np.arange(self.current_size)
        self.queue_size = 1 # for computing moving average, not used in this implementation
        self.queue = np.zeros([self.queue_length, self.queue_size, self.dim, 2])
        self.queue_ptr = 0
        self.queue_counter = np.zeros(self.queue_length,dtype=np.int64)
        self.save_path = save_path
        self.mining_start = 1
        self.mining_pool_size = min(40000, len(dataset.train_sat_cover_list))
        self.mining_save_size = 100
        self.choice_pool = range(self.mining_save_size)
        self.mining_save = np.zeros([self.queue_length, self.mining_save_size],dtype=int)
        self.mode = mode
        # raise Exception
        self.accelerate = accelerate
        self.use_custom_sampling = False

    def update(self, data_sat, data_grd, indexes):
        data_sat_norm = data_sat / np.linalg.norm(data_sat, axis=1, keepdims=True)
        data_grd_norm = data_grd / np.linalg.norm(data_grd, axis=1, keepdims=True)
        batch_size = data_sat.shape[0]
        # writing in distributed training style, complicated. Update the queue according to the previous index.
        for j in range(self.num_replicas):
            index_j = self.indices_out[j:self.current_size:self.num_replicas]
            for i in range(batch_size // self.num_replicas):
                index = index_j[self.queue_ptr + i] %len(self.dataset.train_sat_cover_list)                                                
                assert indexes[i + j * (batch_size // self.num_replicas)] in self.dataset.train_sat_cover_dict[self.dataset.train_sat_cover_list[index]]
                self.queue[index, self.queue_counter[index] % self.queue_size, :, 0] = \
                data_sat_norm[i + j * (batch_size // self.num_replicas)]
                self.queue[indexes[i + j * (batch_size // self.num_replicas)], self.queue_counter[index] % self.queue_size, :, 1] = \
                    data_grd_norm[i + j * (batch_size // self.num_replicas)]
                self.queue_counter[index] += 1
                # print("OK")

            self.queue_ptr = (self.queue_ptr + batch_size // self.num_replicas)

    def generate_indices_sim(self):
        self.queue_ptr = 0

        random.seed(7 + self.epoch)
        self.current_indices = np.arange(self.current_size) %len(self.dataset.train_sat_cover_list)
        random.shuffle(self.current_indices)

        if self.epoch >= self.mining_start:
            assert self.mining_pool_size <= self.queue_length
            mining_pool = np.array(random.sample(range(len(self.dataset.train_sat_cover_list)), self.mining_pool_size),dtype=int)
            product_train = np.matmul(self.queue[:,:,:,1].mean(axis=1), np.transpose(self.queue[mining_pool,:,:,0].mean(axis=1)))
            product_index = np.argsort(product_train, axis=1)
            # update mining pool
            for i in range(product_train.shape[0]):
                self.mining_save[i, :] = mining_pool[product_index[i, -self.mining_save_size:]]
            # randomly sample the first half
            ori_list = self.current_indices[:self.current_size//2]
            self.current_indices = []
            # global hard mining for the other half
            for i in range(self.current_size//self.batch_size):
                index_s = i * (self.batch_size//2)
                index_e = index_s + min(self.batch_size//2, self.current_size//2 - index_s)
                self.current_indices.extend(ori_list[index_s:index_e])
                hard_list = []
                for j in range(index_s, index_e):
                    grd_id = random.choice(self.dataset.train_sat_cover_dict[self.dataset.train_sat_cover_list[ori_list[j]]])
                    idx = int(random.choice(self.mining_save[grd_id]))
                    # keep random sampling until there is no overlap in the batch, hard coded as VIGOR is complicated
                    while True:
                        flag = False
                        for grd_idx in self.dataset.train_sat_cover_dict[self.dataset.train_sat_cover_list[idx]]:
                            if not self.dataset.check_overlap(ori_list[index_s:index_e],
                                                          grd_idx) or not self.dataset.check_overlap(hard_list, grd_idx):
                                flag = True
                        if flag:
                            idx = random.choice(self.mining_save[grd_id])
                        else:
                            break
                    hard_list.append(idx)
                self.current_indices.extend(hard_list)
        self.current_indices = np.array(self.current_indices, dtype=int)
        assert len(self.current_indices) == self.current_size
        print('sampler updated!')
    
    
    def enable_custom_sampling(self):
        self.use_custom_sampling = True

    def update_epoch(self):
        #print(f"self.epoch {self.epoch}")
        # assert(0)
        if self.epoch >= self.mining_start:
            self.generate_indices_sim()

        if self.rank == 0:
            np.save(os.path.join(self.save_path, 'queue.npy'), self.queue)
            np.save(os.path.join(self.save_path, 'queue_counter.npy'), self.queue_counter)

    def load(self, path):
        self.mining_start = 0
        self.queue_counter = np.load(os.path.join(path, 'queue_counter.npy'))
        self.queue = np.load(os.path.join(path, 'queue.npy'))

    def __iter__(self) -> Iterator[T_co]:
        if self.use_custom_sampling:
            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.current_indices), generator=g).tolist()  # type: ignore
            else:
                indices = list(range(len(self.current_indices)))  # type: ignore

            if not self.drop_last:
                # add extra samples to make it evenly divisible
                padding_size = self.current_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
            else:
                # remove tail of data to make it evenly divisible.
                indices = indices[:self.current_size]
            assert len(indices) == self.current_size

            # subsample
            self.indices_out = self.current_indices[indices].tolist()
            indices = indices[self.rank:self.current_size:self.num_replicas]
            # assert len(indices) == self.num_samples
            # print(indices)
            indices_out = self.current_indices[indices].tolist()
            # print(self.rank, len(indices), len(indices_out))

            return iter(indices_out)
        else:
            return super().__iter__()




class HardSampler(DistributedSampler[T_co]):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = False,
                 seed: int = 0, drop_last: bool = False, batch_size: int = 128, mode = 'similarity', dim=1000, save_path=None) -> None:
        super(HardSampler, self).__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.dim = dim
        self.batch_size = batch_size * self.num_replicas
        self.queue_length = max(dataset.train_data_size, len(dataset.train_sat_cover_list))
        self.current_size = len(self.dataset) // self.batch_size * self.batch_size
        self.current_indices = np.arange(self.current_size)
        self.queue_size = 1 # for computing moving average, not used in this implementation
        self.queue = np.zeros([self.queue_length, self.queue_size, self.dim, 2])
        self.queue_ptr = 0
        self.queue_counter = np.zeros(self.queue_length,dtype=np.int64)
        self.save_path = save_path
        self.mining_start = 1
        self.mining_pool_size = min(40000, len(dataset.train_sat_cover_list))
        self.mining_save_size = 100
        self.choice_pool = range(self.mining_save_size)
        self.mining_save = np.zeros([self.queue_length, self.mining_save_size],dtype=int)
        self.mode = mode
        self.dynamic_ratio = False
        self.use_custom_sampling = False
        # raise Exception
        # print("Batch size per GPU:", batch_size)
        # print("Number of GPUs (replicas):", self.num_replicas)
        # print("Global batch size:", self.batch_size)
        # print(f'self.queue_length {self.queue_length}')
        # print(f'self.current_size {self.current_size}') 

    def update(self, data_sat, data_grd, indexes):
        data_sat_norm = data_sat / np.linalg.norm(data_sat, axis=1, keepdims=True)
        data_grd_norm = data_grd / np.linalg.norm(data_grd, axis=1, keepdims=True)
        batch_size = data_sat.shape[0]
        #print(f'这个是gather之后的data_sat.shape[0] {batch_size}')  GPU数量*每张卡的batch size

       

        # writing in distributed training style, complicated. Update the queue according to the previous index.
        for j in range(self.num_replicas):
            #print(f'self.num_replicas {self.num_replicas}') # =GPUs数量
            index_j = self.indices_out[j:self.current_size:self.num_replicas]




            for i in range(batch_size // self.num_replicas):
                index = index_j[self.queue_ptr + i] %len(self.dataset.train_sat_cover_list)
                assert indexes[i + j * (batch_size // self.num_replicas)] in self.dataset.train_sat_cover_dict[self.dataset.train_sat_cover_list[index]]
                self.queue[index, self.queue_counter[index] % self.queue_size, :, 0] = \
                data_sat_norm[i + j * (batch_size // self.num_replicas)]
                self.queue[indexes[i + j * (batch_size // self.num_replicas)], self.queue_counter[index] % self.queue_size, :, 1] = \
                    data_grd_norm[i + j * (batch_size // self.num_replicas)]
                self.queue_counter[index] += 1
            self.queue_ptr = (self.queue_ptr + batch_size // self.num_replicas)


    def get_hard_example_ratio(self, epoch):
    # Example: Linearly increase the ratio from 0.1 to 0.5 over 100 epochs
        start_ratio = 0.1
        end_ratio = 0.5
        total_epochs = 100
        return start_ratio + (end_ratio - start_ratio) * min(epoch / total_epochs, 1)

    def generate_indices_sim(self):
        self.queue_ptr = 0

        random.seed(7 + self.epoch)
        self.current_indices = np.arange(self.current_size) % len(self.dataset.train_sat_cover_list)
        random.shuffle(self.current_indices)

        if self.epoch >= self.mining_start:
            assert self.mining_pool_size <= self.queue_length
            mining_pool = np.array(random.sample(range(len(self.dataset.train_sat_cover_list)), self.mining_pool_size), dtype=int)
            product_train = np.matmul(self.queue[:, :, :, 1].mean(axis=1), np.transpose(self.queue[mining_pool, :, :, 0].mean(axis=1)))
            product_index = np.argsort(product_train, axis=1)

            # update mining pool
            for i in range(product_train.shape[0]):
                self.mining_save[i, :] = mining_pool[product_index[i, -self.mining_save_size:]]

            # Example: Linearly increase the ratio from 0.1 to 0.5 over 100 epochs
            if self.dynamic_ratio:
                hard_example_ratio = self.get_hard_example_ratio(self.epoch)
            else:
                hard_example_ratio = 0.25

            num_hard_examples_per_batch = int(self.batch_size * hard_example_ratio)
            num_original_examples_per_batch = self.batch_size - num_hard_examples_per_batch

            ori_list = self.current_indices[:self.current_size // 2]
            self.current_indices = []

            for i in range(self.current_size // self.batch_size):
                index_s = i * (self.batch_size // 2)
                index_e = index_s + min(num_original_examples_per_batch, self.current_size // 2 - index_s)
                self.current_indices.extend(ori_list[index_s:index_e])
                hard_list = []

                for j in range(index_s, index_e):
                    grd_id = random.choice(self.dataset.train_sat_cover_dict[self.dataset.train_sat_cover_list[ori_list[j]]])
                    idx = int(random.choice(self.mining_save[grd_id]))

                    while True:
                        flag = False
                        for grd_idx in self.dataset.train_sat_cover_dict[self.dataset.train_sat_cover_list[idx]]:
                            if not self.dataset.check_overlap(ori_list[index_s:index_e],
                                                            grd_idx) or not self.dataset.check_overlap(hard_list, grd_idx):
                                flag = True
                        if flag:
                            idx = random.choice(self.mining_save[grd_id])
                        else:
                            break
                    hard_list.append(idx)

                self.current_indices.extend(hard_list)

            self.current_indices = np.array(self.current_indices, dtype=int)
            assert len(self.current_indices) == self.current_size
            print('sampler updated!')


    def update_epoch(self):
        #if self.epoch >= self.mining_start:
        self.generate_indices_sim()
        if self.rank == 0:
            np.save(os.path.join(self.save_path, 'queue.npy'), self.queue)
            np.save(os.path.join(self.save_path, 'queue_counter.npy'), self.queue_counter)


    def enable_custom_sampling(self):
        self.use_custom_sampling = True

    def load(self, path):
        self.mining_start = 0
        self.queue_counter = np.load(os.path.join(path, 'queue_counter.npy'))
        self.queue = np.load(os.path.join(path, 'queue.npy'))

    def __iter__(self) -> Iterator[T_co]:
        if self.use_custom_sampling:
            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.current_indices), generator=g).tolist()  # type: ignore
            else:
                indices = list(range(len(self.current_indices)))  # type: ignore

            if not self.drop_last:
                # add extra samples to make it evenly divisible
                padding_size = self.current_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
            else:
                # remove tail of data to make it evenly divisible.
                indices = indices[:self.current_size]
            assert len(indices) == self.current_size

            # subsample
            self.indices_out = self.current_indices[indices].tolist()
            indices = indices[self.rank:self.current_size:self.num_replicas]
            # assert len(indices) == self.num_samples
            # print(indices)
            indices_out = self.current_indices[indices].tolist()
            # print(self.rank, len(indices), len(indices_out))

            return iter(indices_out)
        else:
            return super().__iter__()
        







def train_model(opt):

    accelerate = Accelerator(mixed_precision='fp16', log_with="wandb")
    
    accelerate.init_trackers("UniQT")
    # set_seed(1)



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


    accelerate.print(transform_train_list_street)
    data_transforms = {
        'train_street': transforms.Compose(transform_train_list_street),
        'train_sate' : transforms.Compose(transform_train_list_sate),
        'val': transforms.Compose(transform_val_list)}

    accelerate.print(f"opt.same_area {opt.same_area}")
    # assert(0)

    
    image_datasets = TrainDataloader(data_dir, data_transforms['train_street'], data_transforms['train_sate'], opt.same_area)
    #image_datasets = TrainDataloaderMining(data_dir, data_transforms['train_street'], data_transforms['train_sate'], opt.same_area)

    # for data in tqdm(image_datasets):
    #     data
    #     continue
    #     None
    #     # print()
    # assert(0)
    mining_sampler = HardSampler
    train_sampler = mining_sampler(image_datasets, batch_size=opt.batchsize, dim=opt.class_dim, save_path="./")
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize, sampler = train_sampler,
                                              shuffle=False, num_workers=12, pin_memory=False, drop_last=True)


    # dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize,
    #                                           shuffle=True, num_workers=12, pin_memory=False, drop_last=True)

    # class_names = image_datasets.classes
    accelerate.print(f'there are {len(image_datasets)} IDs')
    opt.nclasses = len(image_datasets)

    if accelerate.is_main_process:
        dir_name = os.path.join('./model', opt.name)
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        # save opts
        with open('%s/opts.yaml' % dir_name, 'w') as fp:
            yaml.dump(vars(opt), fp, default_flow_style=False)


    # ============ building networks ... ============
    model = two_view_net_swin_infonce_region_cluster(len(image_datasets), droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
                                      LPN=True, block=opt.block, model=opt.backbone, class_dim=opt.class_dim, dataset=opt.dataset)

    # assert(0)
    accuracy = Accuracy(num_classes=len(image_datasets), task='multiclass').cuda()
    
    
    # accelerate.print(model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerate.print(f'number of params: {n_parameters//1000000} M')

    
    
    num_epochs = opt.epoch
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
    #scaler = torch.cuda.amp.GradScaler()

    #=================scheduler =======================
    scheduler = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
    
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[80,120,160])
    resume =False
    model, criterion_class, infonce, optimizer, dataloaders = accelerate.prepare(model, criterion_class, infonce, optimizer, dataloaders)
    if resume:
        accelerate.load_sate("/home/minghach/Data/CMH/LPN/model/vigor-swint-infonce-UniQT-accelerate-16/each_epoch")

    #########################################################
    since = time.time()
    for epoch in range(num_epochs - start_epoch):
        epoch = epoch + start_epoch
        accelerate.print('=========================================')
        accelerate.print(f'Epoch {epoch}/{(num_epochs - 1)}')
        accelerate.print('-' * 10)
        
        # train_sampler.set_epoch(epoch)
        # train_sampler.update_epoch()
        # if epoch >= 80:
        #     train_sampler.enable_custom_sampling()

        if epoch < 5:
            train_sampler.set_epoch(epoch)  # Use default behavior
        else:
            train_sampler.set_epoch(epoch)
            train_sampler.update_epoch()
            train_sampler.enable_custom_sampling()

        train_one_epoch(accelerate, model, epoch, criterion_class, infonce, 
                        optimizer, accuracy, dataloaders, scheduler, num_epochs ,opt, train_sampler)
                        
        time_elapsed = time.time() - since
        accelerate.print('Training complete in {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))
    accelerate.end_training()




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


def train_one_epoch(accelerate, model, epoch, criterion_class, infonce, optimizer, accuracy, dataloaders, scheduler, num_epochs, opt, train_sampler):

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



    for data in (tqdm(dataloaders) if accelerate.is_local_main_process else dataloaders):
        step_corrects = 0.0
        step_corrects2 = 0.0
        step_corrects3 = 0.0
        step_lpn_corrects = 0.0
        step_lpn_corrects2 = 0.0
        step_lpn_corrects3 = 0.0
        loss_main = 0.0
        

        sate_data, street_data,  all_label, idx = data
        #print(all_label)

        # sate_all = concat_all_gather(sate_data)
        # print(f'sate_all {sate_all.size()}')

        # sate_all2 = accelerate.gather(sate_data)
        # print(f'sate_all {sate_all2.size()}')

        # print(sate_all.equal(sate_all2))



        # assert(0)
        #train_sampler.update(concat_all_gather(embed_k).detach().cpu().numpy(),concat_all_gather(embed_q).detach().cpu().numpy(),concat_all_gather(indexes).detach().cpu().numpy())
        # sate_data = sate_data.cuda(non_blocking=True)
        # street_data = street_data.cuda(non_blocking=True)
        # all_label = all_label.cuda(non_blocking=True)

        # print(all_label)
        # assert(0)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        
        result = model(sate_data, street_data, epoch)
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

        sate_embd_all = accelerate.gather(sate_embd)
        street_embd_all = accelerate.gather(street_embd)
        idx_all = accelerate.gather(idx)

        #query = street


        if epoch >= 5:
            train_sampler.update(sate_embd_all.detach().cpu().numpy(),street_embd_all.detach().cpu().numpy(),idx_all.detach().cpu().numpy())
            # if epoch == 80:
            #     accelerate.print(f"Hard example mining starts!")
        
        
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

        accelerate.backward(loss)
        optimizer.step()

        accelerate.log({"step loss": loss.item()})

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

    accelerate.print(f'Loss: {epoch_loss:.4f}')
    accelerate.print(f"epoch_loss_global: {epoch_loss_global:.4f},       epoch_loss_branch4: {epoch_loss_branch4:.4f}     epoch_infonce_sup: {epoch_infonce_loss} ")
    accelerate.print(f"Satellite_Acc:{100*epoch_acc_sate:.4f}%    Street_Acc:{100*epoch_acc_street:.4f}%")
    accelerate.print(f"Satellite_LPN_Acc:{100*epoch_lpn_acc_sate:.4f}%    Street_LPN_Acc:{100*epoch_lpn_acc_street:.4f}%")

    
    scheduler.step()

    # if (epoch+1) % 20 == 0:
    #     save_network(model, opt.name, epoch)

    if accelerate.is_main_process:  
        if (epoch+1) == num_epochs: 
            unwrp_model = accelerate.unwrap_model(model)        
            save_network(unwrp_model, opt.name, epoch)
        
        # if (epoch+1) % 10 == 0:
        #     inter_epoch_path = os.path.join('./model',opt.name, f'epoch_{epoch}')
        #     if not os.path.isdir(inter_epoch_path):
        #         os.mkdir(inter_epoch_path)
        #     accelerate.save_state(inter_epoch_path)

        if epoch+1 in [80, 85, 90]:
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