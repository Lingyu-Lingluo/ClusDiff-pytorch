# ClusDiff-pytorch
This repo contains the PyTorch implementation of the paper:"Diffusion Model with Clustering-based Conditioning for Food Image Generation" > Authors: Yue Han, Jiangpeng He, Mridul Gupta, Edward J. Delp, and Fengqing Zhu (Purdue University).Reproduced by: [Lingyu-Lingluo](https://github.com/Lingyu-Lingluo) 

ClusDiff is a clustering-based training framework for conditional diffusion models. It is designed to generate high-quality, representative food images, specifically addressing the challenges of high intra-class variance and data scarcity in image-based dietary assessment.ClusDiff operates in a two-stage process to enhance generative quality:
* Stage 1: Sub-label Generation >     The framework employs a clustering algorithm (Affinity Propagation) to partition food categories into fine-grained sub-labels. These sub-labels capture the distinct visual modes (e.g., different cooking styles or ingredients) within a single class.
* Stage 2: Conditional Injection >     These discovered sub-labels are injected as conditioning signals into the diffusion model. This allows the generative process to be guided by specific sub-group characteristics, leading to the synthesis of highly representative sub-population images.

## About this repository
The Stable Diffusion and LoRA implementations in this repo are built upon the [diffusers](https://github.com/huggingface/diffusers) library.

## Dataset
This repo utilizes the Food-101 dataset as described in the original paper.You can download the dataset manually from the [official website](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz), or through the `torchvision` downloader pre-integrated into our scripts.

## Running the Experiments
All experiments are configured using YAML files located in the configs/ directory. These files manage both hyperparameters and file paths.By default, file paths are set as relative paths.If you need to modify the directory settings, please update the path in the corresponding YAML file to the absolute path of this repository on your local machine.

## Repo Structure
```text
clustering_food/
├── config/
│   ├── fine_tune.yaml      # Config for stable diffusion training
│   └── generate.yaml       # Config for image generation
├── data/
│   └── food-101.tar.gz     
├── clustering.py           # Stage 1: Affinity Propagation clustering
├── train_lora.py           # Stage 2: Training script for LoRA-based diffusion
├── generate.py             # Stage 2*: Inference and image generation
└── fid.py                  # Evaluation script for FID score       
