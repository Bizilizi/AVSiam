#!/bin/bash
#SBATCH --job-name="AVSiam(AV)"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --partition=DEADLINE
#SBATCH --comment="ICCV"
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/zverev/logs/avsiam/slurm-%j.out
#SBATCH --error=/storage/slurm/zverev/logs/avsiam/slurm-%j.out
#SBATCH --nodelist=node21

ngpu=1

source activate avsiam
set -x

# mount squahshfs
METADATA_DIR=/tmp/$UID
MOUNTPOINT=$METADATA_DIR/$SLURM_JOBID/cav-mae
DATASET_DIR=$MOUNTPOINT/vggsound

trap cleanup EXIT

cleanup () {
    fusermount -u $MOUNTPOINT
    rmdir $MOUNTPOINT
    rmdir $METADATA_DIR/$SLURM_JOBID
    rmdir $METADATA_DIR || true
}

#create directories in /tmp
mkdir -p $MOUNTPOINT 
mkdir -p /storage/local/zverev

# copy the squashfs file to the local storage
rsync -av --progress --ignore-existing /home/wiss/zverev/datasets/cav-mae.squashfs /storage/local/zverev/cav-mae.squashfs

#mount the squashfs file
squashfuse /storage/local/zverev/cav-mae.squashfs $MOUNTPOINT 



model=cav-mae-ft
ftmode=mm_grad
pretrain_path=/home/wiss/zverev/AVSiam/artefacts/as+vgg+acav_pretrained.pth
freeze_base=False


bal=bal



head_lr=10 # newly initialized ft layers uses 100 times larger than the base lr
lr=5e-05

epoch=20
lrscheduler_start=2
lrscheduler_decay=0.75
lrscheduler_step=1


batch_size=32
accumulate_grad_batches=2
wa=True
wa_start=10
wa_end=15
lr_adapt=False
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
freqm=48
timem=192
mixup=0.5
label_smooth=0.1

dataset=vggsound

# prepare train jsons
tr_data=/storage/slurm/zverev/datasets/cav-mae/vggsound/metadata/vgg_train_cleaned.json
te_data=/storage/slurm/zverev/datasets/cav-mae/vggsound/metadata/vgg_test_cleaned.json
w_data=/storage/slurm/zverev/datasets/cav-mae/vggsound/metadata/vgg_train_cleaned_weight.csv

# copy train and test files and repalce the path to the mountpoint
cp $tr_data $METADATA_DIR/
cp $te_data $METADATA_DIR/
cp $w_data $METADATA_DIR/

sed -i 's|/storage/slurm/zverev/datasets/cav-mae|'$DATASET_DIR'|g' $METADATA_DIR/vgg_train_cleaned.json
sed -i 's|/storage/slurm/zverev/datasets/cav-mae|'$DATASET_DIR'|g' $METADATA_DIR/vgg_test_cleaned.json

tr_data=$METADATA_DIR/vgg_train_cleaned.json
te_data=$METADATA_DIR/vgg_test_cleaned.json


label_csv=/storage/slurm/zverev/datasets/cav-mae/vggsound/metadata/class_labels_indices_vgg.csv



export PYTHONWARNINGS="ignore"

exp_dir=./exp/CL-test2-${model}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-${lrscheduler_step}-bs${batch_size}-lda${lr_adapt}-${ftmode}-fz${freeze_base}-h${head_lr}-a5
# exp_dir=./ft_base

mkdir -p $exp_dir



#  torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nproc_per_node=1 \
#     ../../src/run_retrieval.py --model ${model} --dataset ${dataset} \
#     --data_train ${tr_data} --data_val ${te_data} --exp_dir $exp_dir \
#     --label_csv ${label_csv} --n_class 309 \
#     --lr $lr --n_epochs ${epoch} --batch_size $batch_size --save_model True \
#     --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
#     --label_smooth ${label_smooth} \
#     --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
#     --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
#     --loss CE --metrics acc --warmup True \
#     --wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
#     --pretrain_path ${pretrain_path} --ftmode ${ftmode} \
#     --freeze_base ${freeze_base} --head_lr ${head_lr} \
#     --mm_lr 1 \
#     --num_workers 6 --skip_frame_agg False --wandb 0 --model_name vgg-b_robust_2mean --dis_w 0.0 --dis_w_2 0.0



torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nproc_per_node=${ngpu} \
    ../../src/run_cavmae_ft_base.py --model ${model} --dataset ${dataset} \
    --data_train ${tr_data} --data_val ${te_data} --exp_dir $exp_dir \
    --label_csv ${label_csv} --n_class 309 \
    --lr $lr --n_epochs ${epoch} --batch_size $batch_size --accumulate_grad_batches ${accumulate_grad_batches} --save_model True \
    --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
    --label_smooth ${label_smooth} \
    --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
    --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
    --loss CE --metrics acc --warmup True \
    --wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
    --pretrain_path ${pretrain_path} --ftmode ${ftmode} \
    --freeze_base ${freeze_base} --head_lr ${head_lr} \
    --mm_lr 1 \
    --num_workers 6 --skip_frame_agg False --wandb 0 --model_name vgg-b_robust_2mean_acav --dis_w 0.0 --dis_w_2 0.0 \
    --sql_path /home/wiss/zverev/AVSiam/artefacts/sql \
    --video_path_prefix ${DATASET_DIR}/sample_audio \
    --image_path_prefix ${DATASET_DIR}/sample_frames


