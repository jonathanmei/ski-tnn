#! /usr/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

arch=laxtnn_decay_99_pre
# change to your data dir
data_dir=data-bin/wikitext-103

bash ${SCRIPT_DIR}/train_alm.sh 1 $arch $data_dir
