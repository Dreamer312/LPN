########################################################################
# accelerate launch train_cvusa_accelerate.py \
#  --name='cvusa-swint-infonce-UQPT-accelerate-1' \
#  --data_dir='/home/minghach/Data/CMH/LPN/dataset/cvusa/CVUSA_ori/train' \
#  --erasing_p=0.5 \
#  --block=12 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.15 \
#  --optimizer='SGD' \
#  --batchsize=64 \




#  python test_cvusa_infonce.py \
# --name='cvusa-swint-infonce-UQPT-accelerate-1' \
# --test_dir='/home/minghach/Data/CMH/LPN/dataset/cvusa/CVUSA_ori/val' \
# --batchsize=512 \
# --gpu_ids='1' \
#########################################################################



########################################################################
# accelerate launch train_cvusa_accelerate.py \
#  --name='cvusa-swint-infonce-UQPT-accelerate-2' \
#  --data_dir='/home/minghach/Data/CMH/LPN/dataset/cvusa/CVUSA_ori/train' \
#  --erasing_p=0.5 \
#  --block=12 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.2 \
#  --optimizer='SGD' \
#  --batchsize=64 \


#  python test_cvusa_infonce.py \
# --name='cvusa-swint-infonce-UQPT-accelerate-2' \
# --test_dir='/home/minghach/Data/CMH/LPN/dataset/cvusa/CVUSA_ori/val' \
# --batchsize=512 \
# --gpu_ids='1' \
#########################################################################





########################################################################
# accelerate launch train_cvusa_accelerate.py \
#  --name='cvusa-swint-infonce-UQPT-accelerate-3' \
#  --data_dir='/home/minghach/Data/CMH/LPN/dataset/cvusa/CVUSA_ori/train' \
#  --erasing_p=0.5 \
#  --block=9 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.2 \
#  --optimizer='SGD' \
#  --batchsize=64 \


#  python test_cvusa_infonce.py \
# --name='cvusa-swint-infonce-UQPT-accelerate-3' \
# --test_dir='/home/minghach/Data/CMH/LPN/dataset/cvusa/CVUSA_ori/val' \
# --batchsize=512 \
# --gpu_ids='1' \
#########################################################################


########################################################################
# accelerate launch train_cvusa_accelerate.py \
#  --name='cvusa-swint-infonce-UQPT-accelerate-4' \
#  --data_dir='/home/minghach/Data/CMH/LPN/dataset/cvusa/CVUSA_ori/train' \
#  --erasing_p=0.5 \
#  --block=9 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.2 \
#  --optimizer='SGD' \
#  --batchsize=62 \
#  --backbone="swint" \
#  --dataset="cvusa"

#  python test_cvusa_infonce.py \
# --name='cvusa-swint-infonce-UQPT-accelerate-4' \
# --test_dir='/home/minghach/Data/CMH/LPN/dataset/cvusa/CVUSA_ori/val' \
# --batchsize=512 \
# --gpu_ids='1' \
#########################################################################



# ConvFFN+no duplicate
########################################################################
# accelerate launch train_cvusa_accelerate.py \
#  --name='cvusa-swint-infonce-UQPT-accelerate-5' \
#  --data_dir='/home/minghach/Data/CMH/LPN/dataset/cvusa/CVUSA_ori/train' \
#  --erasing_p=0.5 \
#  --block=9 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.2 \
#  --optimizer='SGD' \
#  --batchsize=62 \
#  --backbone="swint" \
#  --dataset="cvusa"

#  python test_cvusa_infonce.py \
# --name='cvusa-swint-infonce-UQPT-accelerate-5' \
# --test_dir='/home/minghach/Data/CMH/LPN/dataset/cvusa/CVUSA_ori/val' \
# --batchsize=512 \
# --gpu_ids='1' \
#########################################################################

# ConvFFN+no duplicate
########################################################################
# accelerate launch train_cvusa_accelerate.py \
#  --name='cvusa-swint-infonce-UQPT-accelerate-6' \
#  --data_dir='/home/minghach/Data/CMH/LPN/dataset/cvusa/CVUSA_ori/train' \
#  --erasing_p=0.5 \
#  --block=9 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.2 \
#  --optimizer='SGD' \
#  --batchsize=62 \
#  --backbone="swint" \
#  --dataset="cvusa" \
#  --epoch=200

#  python test_cvusa_infonce.py \
# --name='cvusa-swint-infonce-UQPT-accelerate-6' \
# --test_dir='/home/minghach/Data/CMH/LPN/dataset/cvusa/CVUSA_ori/val' \
# --batchsize=512 \
# --gpu_ids='1' \
#########################################################################





# ConvFFN+no duplicate+dropkey+cosine atten
########################################################################
accelerate launch train_cvusa_accelerate.py \
 --name='cvusa-swint-infonce-UQPT-accelerate-7' \
 --data_dir='/home/minghach/Data/CMH/LPN/dataset/cvusa/CVUSA_ori/train' \
 --erasing_p=0.5 \
 --block=9 \
 --droprate=0.75 \
 --h=256 \
 --w=256 \
 --lr=0.2 \
 --optimizer='SGD' \
 --batchsize=62 \
 --backbone="swint" \
 --dataset="cvusa" \
 --epoch=200

 python test_cvusa_infonce.py \
--name='cvusa-swint-infonce-UQPT-accelerate-7' \
--test_dir='/home/minghach/Data/CMH/LPN/dataset/cvusa/CVUSA_ori/val' \
--batchsize=512 \
--gpu_ids='1' \
#########################################################################