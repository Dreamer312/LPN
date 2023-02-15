# # LPN vgg16
# python train_cvact.py \
# --name='act_vgg_noshare_warm5_8LPN-s-r_lr0.1' \
# --data_dir='/home/wangtyu/datasets/CVACT/train' \
# --warm_epoch=5 \
# --batchsize=16 \
# --use_vgg16 \
# --h=256 \
# --w=256 \
# --fp16 \
# --LPN \
# --lr=0.1 \
# --block=8 \
# --gpu_ids='2'

# python test_cvact.py \
# --name='act_vgg_noshare_warm5_8LPN-s-r_lr0.1' \
# --test_dir='/home/wangtyu/datasets/CVACT/val' \
# --gpu_ids='2'

# LPN resnet50
# python train_cvact.py \
# --name='act_res50_noshare_warm5_8LPN-s-r_lr0.05' \
# --data_dir='/home/wangtyu/datasets/CVACT/train' \
# --warm_epoch=5 \
# --batchsize=16 \
# --h=256 \
# --w=256 \
# --fp16 \
# --LPN \
# --lr=0.05 \
# --block=8 \
# --stride=1 \
# --gpu_ids='2'

# python test_cvact.py \
# --name='act_res50_noshare_warm5_8LPN-s-r_lr0.05' \
# --test_dir='/home/wangtyu/datasets/CVACT/val' \
# --gpu_ids='2'





# #199
# #########################################################################
#  python train_cvact_fp16_infonce_plpn.py \
#  --name='199-3090-swint-infonce-lpn' \
#  --data_dir='/home/ttq/cmh/CVACT/cvact_train' \
#  --erasing_p=0.5 \
#  --block=12 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.04 \
#  --optimizer='SGD' \
#  --batchsize=32 \
#  --gpu_ids='0'



#  python test_cvact_infonce.py \
# --name='199-3090-swint-infonce-lpn' \
# --test_dir='/home/ttq/cmh/CVACT/cvact_val' \
# --gpu_ids='0' \
# #########################################################################



#199
#########################################################################
#  python train_cvact_fp16_infonce_plpn.py \
#  --name='199-1-3090-swint-infonce-lpn' \
#  --data_dir='/home/ttq/cmh/CVACT/cvact_train' \
#  --erasing_p=0.5 \
#  --block=12 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.03 \
#  --optimizer='SGD' \
#  --batchsize=32 \
#  --gpu_ids='0'



#  python test_cvact_infonce.py \
# --name='199-1-3090-swint-infonce-lpn' \
# --test_dir='/home/ttq/cmh/CVACT/cvact_val' \
# --gpu_ids='0' \
#########################################################################




#199
#########################################################################
#  python train_cvact_fp16_infonce_plpn.py \
#  --name='199-2-3090-swint-infonce-lpn' \
#  --data_dir='/home/ttq/cmh/CVACT/cvact_train' \
#  --erasing_p=0.5 \
#  --block=12 \
#  --droprate=0.5 \
#  --h=256 \
#  --w=256 \
#  --lr=0.03 \
#  --optimizer='SGD' \
#  --batchsize=32 \
#  --gpu_ids='0'



#  python test_cvact_infonce.py \
# --name='199-2-3090-swint-infonce-lpn' \
# --test_dir='/home/ttq/cmh/CVACT/cvact_val' \
# --gpu_ids='0' \
#########################################################################



# #########################################################################
#  python train_cvact_fp16_infonce_plpn.py \
#  --name='199-3-3090-swint-infonce-lpn' \
#  --data_dir='/home/ttq/cmh/CVACT/cvact_train' \
#  --erasing_p=0.5 \
#  --block=8 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.03 \
#  --optimizer='SGD' \
#  --batchsize=32 \
#  --gpu_ids='0'



#  python test_cvact_infonce.py \
# --name='199-3-3090-swint-infonce-lpn' \
# --test_dir='/home/ttq/cmh/CVACT/cvact_val' \
# --gpu_ids='0' \
# #########################################################################




# #########################################################################
#  python train_cvact_fp16_infonce_plpn.py \
#  --name='199-5-3090-swint-infonce-lpn' \
#  --data_dir='/root/autodl-tmp/cvact_train' \
#  --erasing_p=0.5 \
#  --block=16 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.03 \
#  --optimizer='SGD' \
#  --batchsize=32 \
#  --gpu_ids='0'



#  python test_cvact_infonce.py \
# --name='199-5-3090-swint-infonce-lpn' \
# --test_dir='/root/autodl-tmp/cvact_val' \
# --gpu_ids='0' \
# #########################################################################




#199
#########################################################################
#  python train_cvact_fp16_infonce_plpn.py \
#  --name='199-6-3090-swint-infonce-lpn' \
#  --data_dir='/root/autodl-tmp/cvact_train' \
#  --erasing_p=0.5 \
#  --block=12 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.03 \
#  --optimizer='SGD' \
#  --batchsize=32 \
#  --gpu_ids='1'



#  python test_cvact_infonce.py \
# --name='199-6-3090-swint-infonce-lpn' \
# --test_dir='/root/autodl-tmp/cvact_val' \
# --gpu_ids='1' \
#########################################################################






# # #########################################################################
#  python train_cvact_fp16_infonce_plpn.py \
#  --name='199-9-3090-swint-infonce-lpn' \
#  --data_dir='/root/autodl-tmp/cvact_train' \
#  --erasing_p=0.5 \
#  --block=12 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.03 \
#  --optimizer='SGD' \
#  --batchsize=32 \
#  --gpu_ids='0'



#  python test_cvact_infonce.py \
# --name='199-9-3090-swint-infonce-lpn' \
# --test_dir='/root/autodl-tmp/cvact_val' \
# --gpu_ids='0' \
# # #########################################################################



# #########################################################################
#  python train_cvact_fp16_infonce_plpn.py \
#  --name='199-10-3090-swint-infonce-lpn' \
#  --data_dir='/root/autodl-tmp/cvact_train' \
#  --erasing_p=0.5 \
#  --block=9 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.03 \
#  --optimizer='SGD' \
#  --batchsize=32 \
#  --gpu_ids='1'



#  python test_cvact_infonce.py \
# --name='199-10-3090-swint-infonce-lpn' \
# --test_dir='/root/autodl-tmp/cvact_val' \
# --gpu_ids='1' \
# #########################################################################





# # #########################################################################
#  python train_cvact_fp16_infonce_plpn.py \
#  --name='199-11-3090-swint-infonce-lpn' \
#  --data_dir='/root/autodl-tmp/cvact_train' \
#  --erasing_p=0.5 \
#  --block=12 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.04 \
#  --optimizer='SGD' \
#  --batchsize=32 \
#  --gpu_ids='0'



#  python test_cvact_infonce.py \
# --name='199-11-3090-swint-infonce-lpn' \
# --test_dir='/root/autodl-tmp/cvact_val' \
# --gpu_ids='0' \
# # #########################################################################



# #########################################################################
#  python train_cvact_fp16_infonce_plpn.py \
#  --name='199-12-3090-swint-infonce-lpn' \
#  --data_dir='/root/autodl-tmp/cvact_train' \
#  --erasing_p=0.3 \
#  --block=12 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.04 \
#  --optimizer='SGD' \
#  --batchsize=32 \
#  --gpu_ids='1'



#  python test_cvact_infonce.py \
# --name='199-12-3090-swint-infonce-lpn' \
# --test_dir='/root/autodl-tmp/cvact_val' \
# --gpu_ids='1' \
# #########################################################################





# # #########################################################################
#  python train_cvact_fp16_infonce_plpn.py \
#  --name='199-14-3090-swint-infonce-lpn' \
#  --data_dir='/root/autodl-tmp/cvact_train' \
#  --erasing_p=0.3 \
#  --block=12 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.04 \
#  --optimizer='SGD' \
#  --batchsize=32 \
#  --gpu_ids='1'


#  python test_cvact_infonce.py \
# --name='199-14-3090-swint-infonce-lpn' \
# --test_dir='/root/autodl-tmp/cvact_val' \
# --gpu_ids='1' \
# # #########################################################################



# #########################################################################
#  python train_cvact_fp16_infonce_plpn.py \
#  --name='199-15-3090-swint-infonce-lpn' \
#  --data_dir='/root/autodl-tmp/cvact_train' \
#  --erasing_p=0.3 \
#  --block=12 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.04 \
#  --optimizer='SGD' \
#  --batchsize=32 \
#  --gpu_ids='0'



#  python test_cvact_infonce.py \
# --name='199-15-3090-swint-infonce-lpn' \
# --test_dir='/root/autodl-tmp/cvact_val' \
# --gpu_ids='0' \
# #########################################################################