# Distributed Arcface Training in Pytorch

Our modifications are based on the official release of the arcface_torch framework, which implements the ArcFace algorithm with comprehensive support for distributed and sparse training paradigms. The baseline repository provides:

Multiple distributed training configurations demonstrating memory-optimized techniques including mixed-precision training and gradient checkpointing

Native compatibility with Vision Transformer (ViT) architectures and custom datasets (e.g., our cat face dataset)

Integrated ONNX conversion utilities for seamless deployment to MFR evaluation systems

Key enhancements introduced in our implementation:

Architectural Extension in vit.py
A Fourier Transform Layer was incorporated into the ViT backbone to enhance feature representation through frequency domain processing.

Quantitative Evaluation Pipeline (testarcface.py)
A dedicated testing module was developed to systematically evaluate the accuracy metrics of trained weight files across standardized test sets, enabling performance benchmarking and model selection.

This modified framework maintains backward compatibility with all original functionalities while extending its analytical capabilities for both architectural experimentation and empirical validation
## Requirements

To avail the latest features of PyTorch, we have upgraded to version 1.12.0.

- Install [PyTorch](https://pytorch.org/get-started/previous-versions/) (torch>=1.12.0).
- (Optional) Install [DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/), our doc for [install_dali.md](docs/install_dali.md).
- `pip install -r requirement.txt`.
  
## How to Training

To train a model, execute the `train_v2.py` script with the path to the configuration files. The sample commands provided below demonstrate the process of conducting distributed training.

### 1. To run on one GPU:

```shell
python train_v2.py configs/ms1mv3_r50_onegpu
```

Note:   
It is not recommended to use a single GPU for training, as this may result in longer training times and suboptimal performance. For best results, we suggest using multiple GPUs or a GPU cluster.  


### 2. To run on a machine with 8 GPUs:

```shell
torchrun --nproc_per_node=8 train_v2.py configs/ms1mv3_r50
```

### 3. To run on 2 machines with 8 GPUs each:

Node 0:

```shell
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="ip1" --master_port=12581 train_v2.py configs/wf42m_pfc02_16gpus_r100
```

Node 1:
  
```shell
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="ip1" --master_port=12581 train_v2.py configs/wf42m_pfc02_16gpus_r100
```

### 4. Run ViT-B on a machine with 24k batchsize:

```shell
torchrun --nproc_per_node=8 train_v2.py configs/wf42m_pfc03_40epoch_8gpu_vit_b
```


## Download Datasets or Prepare Datasets  
- [https://github.com/qxzheng/CatFace/wiki] (610 IDs, 7112 images)
- [Your Dataset, Click Here!](docs/prepare_custom_dataset.md)

Note: 
If you want to use DALI for data reading, please use the script 'scripts/shuffle_rec.py' to shuffle the InsightFace style rec before using it.  
Example:

`python scripts/shuffle_rec.py ms1m-retinaface-t1`

You will get the "shuffled_ms1m-retinaface-t1" folder, where the samples in the "train.rec" file are shuffled.



### Performance on IJB-C and [**ICCV2021-MFR**](https://github.com/deepinsight/insightface/blob/master/challenges/mfr/README.md)

ICCV2021-MFR testset consists of non-celebrities so we can ensure that it has very few overlap with public available face 
recognition training set, such as MS1M and CASIA as they mostly collected from online celebrities. 
As the result, we can evaluate the FAIR performance for different algorithms.  

For **ICCV2021-MFR-ALL** set, TAR is measured on all-to-all 1:1 protocal, with FAR less than 0.000001(e-6). The 
globalised multi-racial testset contains 242,143 identities and 1,624,305 images. 



## Welcome!  
