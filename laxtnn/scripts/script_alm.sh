#! /usr/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# arch=laxtnn_alm_baseline
arch=laxtnn_alm_tno_fd
# change to your data dir
data_dir=data-bin/wikitext-103

bash ${SCRIPT_DIR}/train_alm.sh 1 $arch $data_dir
