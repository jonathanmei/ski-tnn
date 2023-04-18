#! /usr/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

#arch=laxtnn_decay_99_pre
#arch=laxtnn_sns_laplace
#arch=laxtnn_sns_laplace_small
arch=laxtnn_sns_tiny
wandb_proj=spikes-n-sines-causal

arch=ski_alm_tiny
wandb_proj=ski-causal

# change to your data dir
data_dir=${SCRIPT_DIR}/../../data-bin/wikitext-103

n_gpu=1

profile=false
#profile=true

bash ${SCRIPT_DIR}/train_alm.sh $n_gpu $arch $data_dir $wandb_proj $profile
