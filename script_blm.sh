#! /usr/bin/bash

arch=laxtnn_blm_decay_99
# change to your data dir
data_dir=path_to_bin_data

bash train_blm.sh 1 $arch $data_dir
