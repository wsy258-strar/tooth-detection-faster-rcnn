#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}

# export CUBLAS_WORKSPACE_CONFIG=:4096:8
# bash ./tools/dist_train.sh /data/wangshunyi/mmdetection/configs/DETEX/faster-rcnn_Baseline.py 4
# bash ./tools/dist_train.sh /data/wangshunyi/mmdetection/configs/DETEX/faster-rcnn_r101_fpn_DENTEX.py 4
# bash ./tools/dist_train.sh /data/wangshunyi/mmdetection/configs/DETEX/faster-rcnnn_swin_fpn_1x_DENTEX.py 4
# bash ./tools/dist_train.sh /data/wangshunyi/mmdetection/configs/DETEX/retinanet_r50_fpn_DENTEX.py 4
# bash ./tools/dist_train.sh /data/wangshunyi/mmdetection/configs/DETEX/ssd_300_DENTEX.py 4
# bash ./tools/dist_train.sh /data/wangshunyi/mmdetection/configs/DETEX/faster-rcnn_data_augmentation.py 4
# bash ./tools/dist_train.sh /data/wangshunyi/mmdetection/configs/DETEX/faster-rcnn_set_anchors.py 4
# bash ./tools/dist_train.sh /data/wangshunyi/mmdetection/configs/DETEX/faster-rcnn_iou.py 4
# bash ./tools/dist_train.sh /data/wangshunyi/mmdetection/configs/DETEX/faster-rcnn_final.py 4
# bash ./tools/dist_train.sh /data/wangshunyi/mmdetection/projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py 4
# bash ./tools/dist_train.sh /data/wangshunyi/mmdetection/configs/DETEX/dino_swin_l_DENTEX.py 4
# bash ./tools/dist_train.sh /data/wangshunyi/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py 4