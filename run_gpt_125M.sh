export NCCL_BUFFSIZE=33554432
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

BATCH_SIZE=1
GLOBAL_BATCH_SIZE=4

## GPT-3 models use 2K sequence length/context window
MODEL="simplegpt"
SEQ_LEN=1024

## GPT-3 Small 125M
MODEL_SIZE=0.125
NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTN_HEADS=12
LR=6.0e-4
MIN_LR=6.0e-5

## GPT-3 Medium 350M
# MODEL_SIZE=0.35
# NUM_LAYERS=24
# HIDDEN_SIZE=1024
# NUM_ATTN_HEADS=16
# LR=3.0e-4
# MIN_LR=3.0e-5

## GPT-3 Large 760M
# MODEL_SIZE=0.76
# NUM_LAYERS=24
# HIDDEN_SIZE=1536
# NUM_ATTN_HEADS=16
# LR=2.5e-4
# MIN_LR=2.5e-5

## GPT-3 XL 1.3B
# MODEL_SIZE=1.3
# NUM_LAYERS=24
# HIDDEN_SIZE=2048
# NUM_ATTN_HEADS=16
# LR=2.0e-4
# MIN_LR=2.0e-5

DP_SIZE=1
PP_SIZE=1
NUM_GPUS=1
STEPS=30
PARTITIONS="-"

# If performance degrades, increase this value
RECONFIGURE_THRESHOLD_SCALE=4

train_options=" \
    --steps ${STEPS} \
    --backend nccl \
    --dp ${DP_SIZE} \
    --pp ${PP_SIZE} \
    -N ${NUM_LAYERS} \
    -dm ${HIDDEN_SIZE} \
    -H ${NUM_ATTN_HEADS} \
    --seq ${SEQ_LEN} \
    --parts ${PARTITIONS}"

template_json="template.json"
config_json="config_${MODEL}.json"
sed "s/CONFIG_BATCH_SIZE/${GLOBAL_BATCH_SIZE}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
    | sed "s/ENVPIPE_TYPE/${ENVPIPE_TYPE}/" \
    | sed "s/ENVPIPE_SCHEDULING/${ENVPIPE_SCHEDULING}/" \
    | sed "s/ENVPIPE_RECONFIGURE/${ENVPIPE_RECONFIGURE}/" \
    | sed "s/ENVPIPE_RECONFIG_THRESHOLD_SCALE/${RECONFIGURE_THRESHOLD_SCALE}/" \
    | sed "s/ENVPIPE_GPU/${ENVPIPE_GPU}/" \
	  > ${config_json}

deepspeed ./${MODEL}.py ${train_options} --deepspeed_config ${config_json}