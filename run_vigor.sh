# ########################################################################
# CUDA_VISIBLE_DEVICES='0' \
accelerate launch train_vigor.py \
 --name='vigor-swint-infonce-UniQT-accelerate-1' \
 --data_dir='/home/minghach/Data/CMH/LPN/dataset/vigor' \
 --erasing_p=0.3 \
 --block=3 \
 --droprate=0.75 \
 --h=256 \
 --w=256 \
 --lr=0.2 \
 --optimizer='SGD' \
 --batchsize=2 \


#  python test_cvact_infonce2.py \
# --name='cvact-swint-infonce-UniQT-accelerate-4' \
# --test_dir='/home/minghach/Data/CMH/LPN/dataset/cvact/cvact_val' \
# --batchsize=512 \
# --gpu_ids='1' \
# #########################################################################

