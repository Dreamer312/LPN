########################################################################
# accelerate launch train_cvact_accelerate.py \
#  --name='cvact-swint-infonce-UniQT-accelerate-1' \
#  --data_dir='/home/minghach/Data/CMH/LPN/dataset/cvact/cvact_train' \
#  --erasing_p=0.3 \
#  --block=12 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.2 \
#  --optimizer='SGD' \
#  --batchsize=64 \


#  python test_cvact_infonce2.py \
# --name='cvact-swint-infonce-UniQT-accelerate-1' \
# --test_dir='/home/minghach/Data/CMH/LPN/dataset/cvact/cvact_val' \
# --batchsize=512 \
# --gpu_ids='1' \
#########################################################################


########################################################################
# accelerate launch train_cvact_accelerate.py \
#  --name='cvact-swint-infonce-UniQT-accelerate-2' \
#  --data_dir='/home/minghach/Data/CMH/LPN/dataset/cvact/cvact_train' \
#  --erasing_p=0.3 \
#  --block=12 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.2 \
#  --optimizer='SGD' \
#  --batchsize=64 \


#  python test_cvact_infonce2.py \
# --name='cvact-swint-infonce-UniQT-accelerate-2' \
# --test_dir='/home/minghach/Data/CMH/LPN/dataset/cvact/cvact_val' \
# --batchsize=512 \
# --gpu_ids='1' \
#########################################################################




# ########################################################################
# accelerate launch train_cvact_accelerate.py \
#  --name='cvact-swint-infonce-UniQT-accelerate-3' \
#  --data_dir='/home/minghach/Data/CMH/LPN/dataset/cvact/cvact_train' \
#  --erasing_p=0.3 \
#  --block=12 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.2 \
#  --optimizer='SGD' \
#  --batchsize=64 \


#  python test_cvact_infonce2.py \
# --name='cvact-swint-infonce-UniQT-accelerate-3' \
# --test_dir='/home/minghach/Data/CMH/LPN/dataset/cvact/cvact_val' \
# --batchsize=512 \
# --gpu_ids='1' \
# #########################################################################




# ########################################################################
# accelerate launch train_cvact_accelerate.py \
#  --name='cvact-swint-infonce-UniQT-accelerate-4' \
#  --data_dir='/home/minghach/Data/CMH/LPN/dataset/cvact/cvact_train' \
#  --erasing_p=0.3 \
#  --block=12 \
#  --droprate=0.75 \
#  --h=256 \
#  --w=256 \
#  --lr=0.2 \
#  --optimizer='SGD' \
#  --batchsize=64 \


#  python test_cvact_infonce2.py \
# --name='cvact-swint-infonce-UniQT-accelerate-4' \
# --test_dir='/home/minghach/Data/CMH/LPN/dataset/cvact/cvact_val' \
# --batchsize=512 \
# --gpu_ids='1' \
# #########################################################################


# ########################################################################
CUDA_VISIBLE_DEVICES='0' accelerate launch train_cvact_accelerate.py \
 --name='cvact-swint-infonce-UniQT-accelerate-5' \
 --data_dir='/home/minghach/Data/CMH/LPN/dataset/cvact/cvact_train' \
 --erasing_p=0.3 \
 --block=12 \
 --droprate=0.75 \
 --h=256 \
 --w=256 \
 --lr=0.2 \
 --optimizer='SGD' \
 --batchsize=64 \


#  python test_cvact_infonce2.py \
# --name='cvact-swint-infonce-UniQT-accelerate-4' \
# --test_dir='/home/minghach/Data/CMH/LPN/dataset/cvact/cvact_val' \
# --batchsize=512 \
# --gpu_ids='1' \
# #########################################################################