# ########################################################################
# export CUDA_VISIBLE_DEVICES='0' 
# accelerate launch train_vigor.py \
#  --name='vigor-swint-infonce-UniQT-accelerate-1' \
#  --data_dir='/home/minghach/Data/CMH/LPN/dataset/vigor' \
#  --erasing_p=0.3 \
#  --block=12 \
#  --h=320 \
#  --w=320 \
#  --droprate=0.75 \
#  --lr=0.04 \
#  --optimizer='SGD' \
#  --batchsize=16


#  python test_vigor.py \
# --name='vigor-swint-infonce-UniQT-accelerate-1' \
# --test_dir='/home/minghach/Data/CMH/LPN/dataset/vigor' \
# --gpu_ids='0' \
# #########################################################################





# ########################################################################
#export CUDA_VISIBLE_DEVICES='0' 
accelerate launch train_vigor.py \
 --name='vigor-swint-infonce-UniQT-accelerate-2' \
 --data_dir='/home/minghach/Data/CMH/LPN/dataset/vigor' \
 --erasing_p=0.3 \
 --block=12 \
 --h=320 \
 --w=320 \
 --droprate=0.75 \
 --lr=0.04 \
 --optimizer='SGD' \
 --batchsize=16


#  python test_vigor.py \
# --name='vigor-swint-infonce-UniQT-accelerate-2' \
# --test_dir='/home/minghach/Data/CMH/LPN/dataset/vigor' \
# --gpu_ids='0' \
# #########################################################################







