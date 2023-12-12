#!/bin/bash

set -xue

NUM_GPUS=2
MODEL="base"
SEQ_LENGTH=2048
BATCH_SIZE=8
LR=0.000015
PP_SIZE=1
DP_SIZE=2                                       
EP_SIZE=2

export CUDA_VISIBLE_DEVICES=6,7

OUTPUT_BASEPATH='./outputs'
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
LOG_PATH=${OUTPUT_BASEPATH}/${current_time}
mkdir -p ${LOG_PATH}


# ep zero
plugin="ep_zero" # ep/ep_zero/hybrid
NAME="gpt-${MODEL}-lr${LR}-bs${BATCH_SIZE}-gpus${NUM_GPUS}-plugin${plugin}-pp${PP_SIZE}-dp${DP_SIZE}-ep${EP_SIZE}"

colossalai run --master_port 6689 --nproc_per_node $NUM_GPUS train.py \
    --num_epoch 50 \
    --model_name $MODEL \
    --plugin ${plugin} \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --zero_stage 2 \
    --extra_dp_size 2 \
    --pp_size $PP_SIZE \
    --dp_size $DP_SIZE \
    --ep_size $EP_SIZE \
    --output_path ${LOG_PATH} \
    $@ \
    2>&1 | tee ${LOG_PATH}/${NAME}.log

#  --dataset wikitext --task_name wikitext-2-v1
# ep
# plugin="ep" # ep/ep_zero/hybrid
# NAME="gpt-${MODEL}-lr${LR}-bs${BATCH_SIZE}-gpus${NUM_GPUS}-plugin${plugin}-pp${PP_SIZE}-dp${DP_SIZE}-ep${EP_SIZE}"

# colossalai run --nproc_per_node $NUM_GPUS train.py \
#     --num_epoch 1 \
#     --model_name $MODEL \
#     --plugin ${plugin} \
#     --batch_size $BATCH_SIZE \
#     --lr $LR \
#     --zero_stage 1 \
#     --pp_size $PP_SIZE \
#     --dp_size $DP_SIZE \
#     --ep_size $EP_SIZE \
#     --output_path ${LOG_PATH } \
#     $@ \
#     2>&1 | tee ${LOG_PATH}/${NAME}.log

# hybrid
# plugin="hybrid" # ep/ep_zero/hybrid
# NAME="gpt-${MODEL}-lr${LR}-bs${BATCH_SIZE}-gpus${NUM_GPUS}-plugin${plugin}-pp${PP_SIZE}-dp${DP_SIZE}-ep${EP_SIZE}"

# colossalai run --nproc_per_node $NUM_GPUS train.py \
#     --num_epoch 1 \
#     --model_name $MODEL \
#     --plugin ${plugin} \
#     --batch_size $BATCH_SIZE \
#     --lr $LR \
#     --zero_stage 1 \
#     --pp_size $PP_SIZE \
#     --dp_size $DP_SIZE \
#     --ep_size $EP_SIZE \
#     --output_path ${LOG_PATH } \
#     $@ \
#     2>&1 | tee ${LOG_PATH}/${NAME}.log
