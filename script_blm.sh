#! /usr/bin/bash

arch=roberta_tnn_v2_decay_99
# change to your data dir
data_dir=data-bin/wikitext-103/

bash train_blm.sh 1 $arch $data_dir
