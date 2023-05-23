# SKI-TNN and FD-TNN

This is the the official repository for:

[_Moreno, Alexander*, Jonathan Mei* and Luke Walters*. "SKI to go Faster: Accelerating Toeplitz Neural Networks via Asymmetric Kernels." arXiv preprint arXiv:2305.09028 (2023)._](https://arxiv.org/abs/2305.09028)

Also contains a baseline implementation of TNN.
- [Quick Start](#quick-start)
- [Experiments](#experiments)
  - [Autoregressive language model](#autoregressive-language-model)
    - [1) Preprocess the data](#1-preprocess-the-data)
    - [2) Train the autoregressive language model](#2-train-the-autoregressive-language-model)
    - [3) Length extrapolation](#3-length-extrapolation)
  - [Bidirectional language model](#bidirectional-language-model)
    - [1) Preprocess the data](#1-preprocess-the-data-1)
    - [2) Train the bidirectional language model](#2-train-the-bidirectional-language-model)
  - [LRA](#lra)
    - [1) Preparation](#1-preparation-1)
    - [2) Training](#2-training-1)
- [Citation](#citation)

## Quick Start

Clone using 
```
git clone --recursive https://github.com/jonathanmei/ski-tnn.git
```

If you miss the recursive flag, you can update after the fact
```
cd fairseq-tnn
git submodule update --init
```

On a fresh linux install:
```
bash setup.sh
```

OR if conda is already installed, set up environments:
```
conda env create --file tnn.yaml
conda env create -f lra.yaml && conda env update -f lra2.yaml
bash setup_wikitext_data.sh
bash setup_lra_data.sh
```

In environment `tnn`:
SKI-TNN and FD-TNN are be found in `laxtnn/`. Run via
```
bash laxtnn/scripts/script_alm.sh
bash laxtnn/scripts/script_blm.sh
```

In environment `lra`:
Running `setup_lra_data.sh` should have created `lra_release/` in same dir as `ski-tnn/`.
Run via
```
python script_lra.py
```
The architectures and tasks are matched according to the data type. Batch size and number of GPU's can be modified to fit hardware configuration.

## Experiments

### Autoregressive language model
#### 1) Preprocess data

This is performed by `setup_wikitext_data.sh` and comes from [fairseq](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.pretraining.md).


#### 2) Train the autoregressive language model

Use the following command to train the autoregressive language model:

```
bash script_alm.sh
```

You should change data_dir to preprocessed data.



#### 3) Length extrapolation

After training, you can do a length extrapolation test by the following command, where length is the test length, e.g. 512, 1024,....:

```
bash length_extrapolation.sh tnn_v2_decay_99_pre length
```





### Bidirectional language model

#### 1) Preprocess the data

The same as the autoregressive language model part.



#### 2) Train the bidirectional language model

Use the following command to train the bidirectional language model:

```
bash script_blm.sh
```

You should change data_dir to preprocessed data.





### LRA

#### 1) Preparation
We provide the `setup_lra_data.sh` script.

#### 2) Extending
The `main` branch of this repository points to the `main` branch of the `lra-tnn`, which is a minified version of the code that reproduces the paper. The `dev` branch of this repository points to the `dev` branch of the `lra-tnn` repository, which is a pinned version of [lra]()

## Citation

```
@inproceedings{
    moreno2023ski,
    title={{SKI to go Faster: Accelerating Toeplitz Neural Networks via Asymmetric Kernels}},
    author={Alexander Moreno and Jonathan Mei and Luke Walters},
    booktitle={arXiv:2305.09028},
    year={2023},
    url={https://arxiv.org/abs/2305.09028}
}
```

