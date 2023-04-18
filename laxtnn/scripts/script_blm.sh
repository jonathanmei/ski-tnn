#! /usr/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

#arch=laxtnn_blm_decay_99
#arch=laxtnn_blm_sns_tiny
#wandb="spikes-n-sines-bidirectional"

arch=ski_blm_tiny
wandb="ski-bidirectional"

# change to your data dir
data_dir=data-bin/wikitext-103

bash ${SCRIPT_DIR}/train_blm.sh 1 $arch $data_dir $wandb
