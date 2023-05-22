
# *** assumes our fork of tnn is on machine and current directory is ./tnn/ ***
# run as:  source setup.sh

#install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -f
source ~/miniconda3/bin/activate
conda init
source ~/.bashrc

#create envs
conda env create --file tnn.yaml
conda env create -f lra.yaml && conda env update -f lra2.yaml
# autoregressive and bi-directional LM on wikitext
# preprocess data
source setup_wikitext_data.sh
bash setup_lra_data.sh