#!/bin/sh

mt="mlp"
output="./checkpoints"
if [ $1 -eq 1 ]; then
    mt="transformer"
    output="./checkpoints_trm"
fi

python train.py --mapping_type=$mt --out_dir=$output --wandb