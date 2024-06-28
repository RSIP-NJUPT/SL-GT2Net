# SAR Image Time Series for Land Cover Mapping via Sparse Local-Global Temporal Transformer Network (Under review)

## Introduction
<!-- Due to the effective embedding of temporal information, time-series SAR images can acquire more abundant land covers information, thereby effectively enhancing the accuracy of land cover mapping. However, the significant introduction of temporal features greatly increases the redundancy of deep features. Simultaneously, capturing the multi-scale global-local features of time-series SAR images effectively while considering temporal features has a significant influence on time-series SAR land cover mapping. Motivated by these, a sparse local-global temporal transformer network (SL-GT2Net) is proposed for time-series SAR land cover mapping. Specifically, to address redundant features and extract multiscale discriminative spatial features, we propose the novel and effective sparse multi-scale local window multi-head self-attention block (SMLW-MHSA) and sparse multi-scale global window multi-head self-attention block (SMGW-MHSA). In SMGW-MHSA, each token within each window in the query can interact with globally shared keys and values, achieving linear computational complexity with respect to the number of tokens. Additionally, to further refine spatial-temporal features, we propose the spatial-temporal channel refinement block (STCRB). The experimental results on three publicly available benchmark time-series SAR datasets demonstrate that our SL-GT2Net exhibits outstanding competitiveness compared to other state-of-the-art methods. -->

If this codebase is helpful for you, please consider give me a star ⭐ 😊.


## Usage

### Installation

* Step 1: Create a conda environment

```shell
conda create --name slgt2 python=3.7 -y
conda activate slgt2
```

* Step 2: Install PyTorch

```shell
# CUDA 11.6, If conda cannot install, use pip
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit==11.6 -c pytorch -c conda-forge
# pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

* Step 3: Install OpenMMLab 2.x Codebases and Other Dependencies

```shell
# openmmlab codebases
pip install -U openmim
mim install mmcv-full==1.7.2
pip install mmsegmentation==0.30.0 # 0.30.0

# other dependencies
pip install scipy
pip install numpy==1.23.5
pip install terminaltables
pip install timm # 0.9.16
pip install einops # 0.7.0
pip install monai # 1.3.0
pip install ml_collections # 0.1.1
pip install yapf==0.40.0
pip install matplotlib
pip install hdf5storage
pip install cupy # 13.0.0 very slowly
```
<!-- * Install mmseg
  * Please refer to [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) to get installation guide. 
  * This repository is based on mmseg-0.30.0 and pytorch-1.12.0. -->
<!-- * Clone the repository -->
  <!-- ```shell
  git clone https://github.com/wanghao9610/TMANet.git
  cd TMANet
  pip install -e .
  ``` -->
### Prepare the datasets
  * Download [Slovenia (MTS12) dataset](http://gpcv.whu.edu.cn/data/dataset12/dataset12.html), thanks [Linying Zhao & Shunping Ji, CNN, RNN, or ViT? An evaluation of different deep learning architectures for spatio-temporal representation of Sentinel time series, JSTARS, 2022](https://ieeexplore.ieee.org/document/9940533). 
  * Download [Brandenburg Sentinel-1 time-series dataset](https://github.com/hanzhu97702/ISPRS_STMA), thanks [Spatio-temporal multi-level attention crop mapping method using time-series SAR imagery](https://www.sciencedirect.com/science/article/pii/S0924271623003210).
  * For Slovenia dataset, we need to create a data folder, and put the Slovenia dataset in the data folder. The file structure of Slovenia dataset is as followed: 
    ```none
    ├── configs
    ├── data                                                
    │   ├── slovenia                                      
    │   │   ├── label                                      
    │   │   │   ├── test                                     
    │   │   │   ├── train                                   
    │   │   │   ├── val                                     
    │   │   ├── s1 (only use s1)                                
    │   │   │   ├── test (297 files)                                     
    │   │   │   ├── train (509 files)                                   
    │   │   │   │   ├── eopatch_id_0_col_0_row_19.mat                 
    │   │   │   │   ├── ......                 
    │   │   │   │   ├── eopatch_id_939_col_49_row_26.mat                 
    │   │   │   ├── val (130 files)                                     
    │   │   ├── s2 (not use)                        
    │   │   │   ├── test                                     
    │   │   │   ├── train                                   
    │   │   │   ├── val                                     
    ```


### Training



- single gpu training

```shell
python tools/train.py ${CONFIG_FILE}  
# for example:
python tools/train.py configs/slgt2/slgt2_224x224_40k_slovenia.py
```

- multiple gpus training (assume there are four GPUs)


```shell
CUDA_VISIABLE_DEVICES=0,1,2,3 tools/dist_train.sh ${CONFIG_FILE} 4  
# for example:
CUDA_VISIABLE_DEVICES=0,1,2,3 tools/dist_train.sh configs/slgt2/slgt2_224x224_40k_slovenia.py 4
```


## Testing


- pretrained model and training logs

You can download by [Baidu Netdisk](https://pan.baidu.com/s/1kmdtT97en4wfaSRQLYYNlw) (access code: 1234) or [Google Drive](https://drive.google.com/drive/folders/1lqT1fFq_8w6FZH4e-BvXIY_EqFvd7iWI?usp=drive_link).

- single gpu testing

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
# for example:
python tools/test.py configs/slgt2/slgt2_224x224_40k_slovenia.py work_dirs/slgt2_224x224_40k_slovenia/train_20240530_032438/iter_40000.pth
```

- multiple gpus testing (assume there are four GPUs)

```shell
CUDA_VISIABLE_DEVICES=0,1,2,3 tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} 4
# for example:
CUDA_VISIABLE_DEVICES=0,1,2,3 tools/dist_test.sh configs/slgt2/slgt2_224x224_40k_slovenia.py work_dirs/slgt2_224x224_40k_slovenia/train_20240530_032438/iter_40000.pth 4
```

<!-- ## Citation
  If you find TMANet is useful in your research, please consider citing:
  ```shell
  @inproceedings{wang2021temporal,
  title={Temporal memory attention for video semantic segmentation},
  author={Wang, Hao and Wang, Weining and Liu, Jing},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={2254--2258},
  year={2021},
  organization={IEEE}
  }
  ``` -->
## Acknowledgement
Thanks [mmsegmentation](https://mmsegmentation.readthedocs.io/zh-cn/0.x/index.html) contribution to the community!

<!-- 
# SAR Image Time Series for Land Cover Mapping via Sparse Local-Global Temporal Transformer Network

## Dataset 1 （Semantic Segmentation Based on Temporal Features: Learning of Temporal–Spatial Information From Time-Series SAR Images for Paddy Rice Mapping，TGRS，2022）

### 1. The training dataset are shared by google drive: https://drive.google.com/drive/folders/120X2tLv4-6pxIREOMFFGILId4R98gdWK?usp=sharing
### The dataset is generated from time-series Sentinel-1 SAR images in 2019 in AR,MS, MO, TN of the United States, and Cropland Data Layer (CDL) is used as the label data.
### 2. The time-series Sentinel-1 SAR images is preprocessed and downloaded by Google Earth Engine and the linke of the code can be found below: https://code.earthengine.google.com/49f8e2532075272a79883ad8fbf41ccb
### 3. Download two compressed files named 'src' and 'label' to your local computer and unzip them to the same directory.

## Dataset 2 (Multi-Temporal Sentinel-1/2 (MTS12) Dataset for Land Cover Classification, JSTARS, 2022) 
### - Download websites: http://gpcv.whu.edu.cn/data/dataset12/dataset12.html -->
