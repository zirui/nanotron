#!/bin/bash

# Simple script to create a tiny llama model and train it
set -e -x

. "/ML-A100/team/infra/zirui/miniconda3/etc/profile.d/conda.sh"
conda activate /ML-A100/team/infra/zirui/miniconda3/envs/nano_env


# Create the YAML config file
EXAMPLE_PATH=$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)
# REPO_PATH=$(dirname $EXAMPLE_PATH)

REPO_PATH=/ML-A100/team/infra/zirui/code/minicpm-nanotron

# Setup from environment variables
# Volcengine multi-node settings
export GPUS_PER_NODE=${MLP_WORKER_GPU:-${GPUS_PER_NODE:-1}}
export NNODES=${MLP_WORKER_NUM:-1}
export NODE_RANK=${MLP_ROLE_INDEX:-0}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-127.0.0.1}
export MASTER_PORT=${MLP_WORKER_0_PORT:-1234}
export TASK_ID=${MLP_TASK_ID:-$(date "+%Y-%m-%d-%H-%M")}
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))


export CUDA_DEVICE_MAX_CONNECTIONS=1
export FI_PROVIDER="efa"
export PYTHONPATH=/ML-A100/team/infra/zirui/code/nanotron/src:$PYTHONPATH
export LD_LIBRARY_PATH=/ML-A100/team/infra/zirui/miniconda3/envs/nano_env/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH


# config_file=$EXAMPLE_PATH/config_tiny_llama_new_vocab_50k.yaml

# config_file=$EXAMPLE_PATH/config_tiny_llama.yaml
# config_file=/ML-A100/team/infra/zirui/code/minicpm-nanotron/config_minicpm.yaml
config_file=/ML-A100/team/infra/zirui/code/minicpm-nanotron/training_config_minicpm.yaml
# train_py=$REPO_PATH/run_train.py.py
train_py=$REPO_PATH/run_train_minicpm.py

#torchrun --nproc_per_node=8 run_train.py --config-file examples/debug_run_train.yaml
# python -u -m torch.distributed.run \
torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    $train_py --config-file $config_file


