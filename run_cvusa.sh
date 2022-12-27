# # lpn vgg16
# python train_cvusa.py \
# --name='usa_vgg_noshare_warm5_8LPN-s-r_lr0.1' \
# --data_dir='/home/wangtyu/datasets/CVUSA/train' \
# --warm_epoch=5 \
# --batchsize=16 \
# --h=256 \
# --w=256 \
# --use_vgg16 \
# --fp16 \
# --LPN \
# --lr=0.1 \
# --block=8 \
# --gpu_ids='3'

# python test_cvusa.py \
# --name='usa_vgg_noshare_warm5_8LPN-s-r_lr0.1' \
# --test_dir='/home/wangtyu/datasets/CVUSA/val' \
# --gpu_ids='3' \

# lpn resnet50
# python train_cvusa_fp16.py \
# --name='usa_res50_noshare_warm5_8LPN-s-r_lr0.4' \
# --data_dir='../CVUSA_ori/train' \

# --batchsize=150 \
# --h=256 \
# --w=256 \
# --LPN \
# --lr=0.4 \
# --block=8 \
# --stride=1 \
# --gpu_ids='0'

# python test_cvusa.py \
# --name='usa_res50_noshare_warm5_8LPN-s-r_lr0.4' \
# --test_dir='../CVUSA_ori/val' \
# --gpu_ids='0' \






# python cmh_test_cvusa.py \
# --name='usa_res50_noshare_warm5_8LPN-s-r_lr0.05' \
# --test_dir='../CVUSA_ori/val' \
# --gpu_ids='0' \


# 177
#=======================================================================
# python train_cvusa_fp16.py \
# --name='177-usa_swin_noshare_4LPN-s-r_lr0.1' \
# --data_dir='../CVUSA_ori/train' \
# --batchsize=64 \
# --h=256 \
# --w=256 \
# --LPN \
# --lr=0.03 \
# --block=4 \
# --stride=1 \
# --gpu_ids='0'

# python test_cvusa.py \
# --name='177-usa_swin_noshare_4LPN-s-r_lr0.1' \
# --test_dir='../CVUSA_ori/val' \
# --gpu_ids='0' \
#=======================================================================







# 178
# # # #########################################################################
#  python train_cvusa_fp16_infonce.py \
#  --name='179-1-3090-swint-infonce-lpn' \
#  --data_dir='../CVUSA_ori/train' \
#  --erasing_p=0.5 \
#  --block=4 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.03 \
#  --optimizer='SGD' \
#  --batchsize=64 \
#  --gpu_ids='0'



#  python test_cvusa_infonce.py \
# --name='179-1-3090-swint-infonce-lpn' \
# --test_dir='../CVUSA_ori/val' \
# --gpu_ids='1' \
 # # #########################################################################







# 180
############################################################################
# python train_cvusa_fp16_baseline_info.py \
# --name='180-1-3090-swint-lpn' \
# --data_dir='../CVUSA_ori/train' \
# --batchsize=64 \
# --h=256 \
# --w=256 \
# --LPN \
# --lr=0.03 \
# --block=4 \
# --stride=1 \
# --gpu_ids='0'


#  python test_cvusa_infonce.py \
# --name='180-1-3090-swint-lpn' \
# --test_dir='../CVUSA_ori/val' \
# --gpu_ids='0' \
 # # #########################################################################




 # 181
############################################################################
# python train_cvusa_fp16_infonce.py \
# --name='181-3090-swint-lpn' \
# --data_dir='../CVUSA_ori/train' \
# --batchsize=64 \
# --h=256 \
# --w=256 \
# --lr=0.03 \
# --block=4 \
# --stride=1 \
# --gpu_ids='0'


# python test_cvusa_infonce.py \
# --name='181-3090-swint-lpn' \
# --test_dir='../CVUSA_ori/val' \
# --gpu_ids='0' \
 # #########################################################################




 # 182
############################################################################
# python train_cvusa_fp16_baseline_info.py \
# --name='182-3090-swint-lpn' \
# --data_dir='../CVUSA_ori/train' \
# --batchsize=64 \
# --h=256 \
# --w=256 \
# --LPN \
# --lr=0.03 \
# --block=8 \
# --stride=1 \
# --gpu_ids='0'


#  python test_cvusa_infonce.py \
# --name='182-3090-swint-lpn' \
# --test_dir='../CVUSA_ori/val' \
# --gpu_ids='0' \
 # # #########################################################################






 #183
# python train_cvusa_fp16.py \
# --name='swint_lpn' \
# --data_dir='../CVUSA_ori/train' \
# --batchsize=72 \
# --h=256 \
# --w=256 \
# --LPN \
# --lr=0.04 \
# --block=8 \
# --stride=1 \
# --gpu_ids='0'

# python test_cvusa.py \
# --name='swint_lpn' \
# --test_dir='../CVUSA_ori/val' \
# --gpu_ids='0' \







#186
# # #########################################################################
#  python train_cvusa_fp16_infonce.py \
#  --name='186-6-3090-swint-infonce-lpn' \
#  --data_dir='../CVUSA_ori/train' \
#  --erasing_p=0.5 \
#  --block=12 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.01 \
#  --optimizer='SGD' \
#  --batchsize=16 \
#  --gpu_ids='0'



#  python test_cvusa_infonce.py \
# --name='186-6-3090-swint-infonce-lpn' \
# --test_dir='../CVUSA_ori/val' \
# --gpu_ids='0' \
 # #########################################################################




# #187
# # # #########################################################################
#  python train_cvusa_fp16_infonce.py \
#  --name='187-2-3090-swint-infonce-lpn' \
#  --data_dir='../CVUSA_ori/train' \
#  --erasing_p=0.5 \
#  --block=12 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.01 \
#  --optimizer='SGD' \
#  --batchsize=16 \
#  --gpu_ids='0'



#  python test_cvusa_infonce.py \
# --name='187-2-3090-swint-infonce-lpn' \
# --test_dir='../CVUSA_ori/val' \
# --gpu_ids='0' \
#  # #########################################################################




 # lpn resnet50
# python train_cvusa_fp16.py \
# --name='res50_noshare_warm5_8LPN-s-r_lr0.4' \
# --data_dir='../CVUSA_ori/train' \
# --warm_epoch=5 \
# --batchsize=140 \
# --h=256 \
# --w=256 \
# --LPN \
# --lr=0.4 \
# --block=8 \
# --stride=1 \
# --gpu_ids='0'

# python test_cvusa.py \
# --name='res50_noshare_warm5_8LPN-s-r_lr0.4' \
# --test_dir='../CVUSA_ori/val' \
# --gpu_ids='0' \




#189
# # # #########################################################################
#  python train_cvusa_fp16_infonce.py \
#  --name='189-3090-swint-infonce-lpn' \
#  --data_dir='../CVUSA_ori/train' \
#  --erasing_p=0.5 \
#  --block=12 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.04 \
#  --optimizer='SGD' \
#  --batchsize=32 \
#  --gpu_ids='0'



 python test_cvusa_infonce.py \
--name='189-3090-swint-infonce-lpn' \
--test_dir='../CVUSA_ori/val' \
--gpu_ids='0' \
#  # #########################################################################