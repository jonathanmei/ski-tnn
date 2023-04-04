#! /usr/bin/bash

# arch=laxtnn_blm_decay_99
arch=laxtnn_blm_tno_fd
# change to your data dir
data_dir=

bash train_blm.sh 1 $arch $data_dir
