#! /usr/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

<<<<<<< HEAD
# arch=laxtnn_blm_decay_99
arch=laxtnn_blm_tno_fd
=======
arch=laxtnn_blm_decay_99
>>>>>>> 4d58c6bfcfaf5b1762a920c7db0f387ecd1d081d
# change to your data dir
data_dir=data-bin/wikitext-103

bash ${SCRIPT_DIR}/train_blm.sh 1 $arch $data_dir
