
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
conda env create --file lra.yaml
# autoregressive LM
# preprocess data
source setup_data.sh