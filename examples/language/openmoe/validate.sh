#!/bin/bash

set -xue

NUM_GPUS=2
MODEL="base"
SEQ_LENGTH=2048
BATCH_SIZE=4
LR=0.000015
PP_SIZE=1
DP_SIZE=2
EP_SIZE=2

export CUDA_VISIBLE_DEVICES=1,3

OUTPUT_BASEPATH='./outputs'
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
LOG_PATH=${OUTPUT_BASEPATH}/${current_time}
mkdir -p ${LOG_PATH}

if echo "$@" | grep -q -- "--valid_ckpt_path"; then
    echo "Found --valid_ckpt_path"
else
    echo "please specify --valid_ckpt_path"
    exit 1
fi

# ep zero
plugin="ep_zero" # ep/ep_zero/hybrid
NAME="valid-gpt-${MODEL}-lr${LR}-bs${BATCH_SIZE}-gpus${NUM_GPUS}-plugin${plugin}-pp${PP_SIZE}-dp${DP_SIZE}-ep${EP_SIZE}"

# 检查参数中是否包含 --comment
if [[ "$@" =~ "--comment" ]]; then
    # 循环遍历参数列表
    for arg in "$@"; do
        # 检查是否为 --comment 参数
        if [[ "$arg" == "--comment" ]]; then
            # 获取下一个参数作为评论内容
            index=$(( $OPTIND + 1 ))
            COMMENT="${!index}"
            # 合并 COMMENT 和 NAME
            NAME="$NAME-$COMMENT"
            echo "合并后的结果为: $NAME"
            break
        fi
        ((OPTIND++))
    done
fi
echo $NAME


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
