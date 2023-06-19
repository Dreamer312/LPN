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


 python test_cvusa_infonce.py \
--name='cvusa-swint-infonce-UQPT-accelerate-2' \
--test_dir='/home/minghach/Data/CMH/LPN/dataset/cvusa/CVUSA_ori/val' \
--batchsize=512 \
--gpu_ids='1' \
#########################################################################