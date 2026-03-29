# ClusDiff-pytorch
This repository contains the PyTorch implementation of the paper:"Diffusion Model with Clustering-based Conditioning for Food Image Generation" > Authors: Yue Han, Jiangpeng He, Mridul Gupta, Edward J. Delp, and Fengqing Zhu (Purdue University).Reproduced by: [Lingyu-Lingluo](https://github.com/Lingyu-Lingluo) 

ClusDiff is a clustering-based training framework for conditional diffusion models. It is designed to generate high-quality, representative food images, specifically addressing the challenges of high intra-class variance and data scarcity in image-based dietary assessment.ClusDiff operates in a two-stage process to enhance generative quality:
Stage 1: Sub-label Generation >     The framework employs a clustering algorithm (such as K-Means or Nearest-Neighbor) to partition food categories into fine-grained sub-labels. These sub-labels capture the distinct visual modes (e.g., different cooking styles or ingredients) within a single class.
Stage 2: Conditional Injection >     These discovered sub-labels are injected as conditioning signals into the diffusion model. This allows the generative process to be guided by specific sub-group characteristics, leading to the synthesis of highly representative sub-population images.

## About this repository
The Stable Diffusion and LoRA implementations in this repository are built upon the [diffusers](https://github.com/huggingface/diffusers) library.
