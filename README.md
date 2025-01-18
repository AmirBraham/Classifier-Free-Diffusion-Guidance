# Classifier-Free Diffusion Guidance - Paper Review

This repository contains the source code for a **paper review** of [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). The implementation is inspired by the [classifier_free_ddim](https://github.com/tatakai1/classifier_free_ddim) GitHub repository. We have made several modifications to tailor the code for our specific experiments and to incorporate additional evaluation metrics.

## Table of Contents

- [Introduction](#introduction)
- [Modifications](#modifications)
- [Features](#features)
- [Experimental Results](#experimental-results)
  - [Generated Images](#generated-images)
  - [Performance Metrics](#performance-metrics)
- [Code Structure](#code-structure)
  - [Directory Overview](#directory-overview)
  - [File Descriptions](#file-descriptions)
- [Usage](#usage)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction

The **Classifier-Free Diffusion Guidance** technique enhances image generation models by eliminating the need for an explicit classifier during the diffusion process. This approach simplifies the training pipeline and has shown promising results in generating high-quality images. Our implementation focuses on applying this technique to the **CIFAR-10** dataset, with the flexibility to extend to other datasets such as **MNIST**.

## Modifications

We have adapted the original [classifier_free_ddim](https://github.com/tatakai1/classifier_free_ddim) repository with the following modifications:

- **Removed DDIM Integration**: Focused solely on the classifier-free diffusion technique by dropping DDIM (Denoising Diffusion Implicit Models) to streamline the implementation.
- **Added Evaluation Metrics**: Incorporated **Fréchet Inception Distance (FID)** and **Inception Score (IS)** to quantitatively assess the quality of generated images, aligning with the metrics used in the original paper.
- **Dataset Flexibility**: Enhanced the codebase to support multiple datasets. Our primary experimental results are based on the **CIFAR-10** dataset, but the structure allows easy adaptation to other datasets like **MNIST**.

## Features

- **UNet Architecture**: Utilizes a robust UNet model tailored for image generation tasks.
- **Gaussian Diffusion Process**: Implements the diffusion mechanism for generating images without relying on an external classifier.
- **Parameter Variation**: Allows training the model with different `p_uncound` values to study their impact on performance.
- **Comprehensive Metrics**: Calculates both IS and FID scores to evaluate the diversity and quality of generated images.
- **Image Generation**: Generates and saves images in an organized grid format for easy visualization.
- **Modular Codebase**: Structured to support multiple datasets with minimal modifications.

## Experimental Results

### Generated Images

Below are some images generated from the CIFAR-10 dataset with `p_uncound` set to **0.2** after **2 epochs** of training.

![generated](https://github.com/user-attachments/assets/744d7b86-bb4e-42dc-acc8-2313efc472bf)


## Code Structure

The repository is organized into two main directories: **CIFAR** and **MNIST**. Both directories contain similar structures with slight modifications to accommodate the specific requirements of each dataset, such as the number of input channels and preprocessing techniques.

### Directory Overview

```
.
├── CIFAR
│   ├── blocks.py
│   ├── gaussian_diffusion.py
│   ├── generate.py
│   ├── noise.py
│   ├── train.py
│   ├── unetmodel.py
│   └── utils.py
├── MNIST
│   ├── blocks.py
│   ├── gaussian_diffusion.py
│   ├── generate.py
│   ├── noise.py
│   ├── train.py
│   ├── unetmodel.py
│   └── utils.py
├── plots
│   ├── fid_score_vs_p_uncound.png
│   └── inception_score_vs_p_uncound.png
├── saved_models
│   ├── Classifier_free_CIFAR10_puncound_0.0.pth
│   ├── Classifier_free_CIFAR10_puncound_0.1.pth
│   ├── ...
│   └── Classifier_free_MNIST_puncound_0.1.pth
├── metrics_log.txt
├── README.md
└── requirements.txt
```

### File Descriptions

For each dataset directory (**CIFAR** and **MNIST**), the following files are present:

- **blocks.py**: Contains various building blocks used in the UNet architecture, such as convolutional layers, residual blocks, and attention mechanisms.
- **gaussian_diffusion.py**: Implements the Gaussian Diffusion process, including the forward and reverse diffusion steps.
- **generate.py**: Script to sample and generate images using the trained diffusion model.
- **noise.py**: Handles the addition of noise to images as part of the diffusion process.
- **train.py**: Main training script that trains the UNet model with varying `p_uncound` values and logs the performance metrics.
- **unetmodel.py**: Defines the UNet architecture tailored for the diffusion process.
- **utils.py**: Contains utility functions, including `timestep_embedding` and other helper methods.

## Usage

### Training the Model

To train the model on a specific dataset with different `p_uncound` values, follow these steps:

1. **Navigate to the Dataset Directory**

   ```bash
   cd CIFAR
   # or
   cd MNIST
   ```

2. **Configure Training Parameters**

   Adjust hyperparameters and configurations within the `train.py` script as needed, such as `batch_size`, `timesteps`, `epochs`, and `learning_rate`.

3. **Run the Training Script**

   ```bash
   python train.py
   ```

   The script will train the model for each specified `p_uncound` value (FOR NOW ONLY CIFAR10) , compute the IS and FID scores, save the trained models, and log the metrics.

### Generating Images

After training, generate images using the trained model:

1. **Navigate to the Dataset Directory**

   ```bash
   cd CIFAR
   # or
   cd MNIST
   ```

2. **Run the Generation Script**

   ```bash
   python generate.py
   ```

   This will generate a grid of images and save them as `generated.png` in the specified output directory.

## Prerequisites

Ensure you have the following installed:

- **Python 3.7+**
- **PyTorch**: Compatible with your system's CUDA version if using GPU acceleration.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/classifier-free-diffusion-guidance.git
   cd classifier-free-diffusion-guidance
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *If you encounter issues with `torch` installation, refer to [PyTorch's official installation guide](https://pytorch.org/get-started/locally/).*

## Acknowledgements

- **[PyTorch](https://pytorch.org/)**: An open-source machine learning framework.
- **[CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)**: Used for training and evaluating the model.
- **[Matplotlib](https://matplotlib.org/)**: For plotting and visualization.
- **[torchvision](https://pytorch.org/vision/stable/index.html)**: Provides utilities for image processing.
- **[Original GitHub Repository](https://github.com/tatakai1/classifier_free_ddim)**: Inspiration for this implementation.
- **[Gaussian Diffusion Models](https://arxiv.org/abs/2006.11239)**: Inspired by diffusion-based generative models research.

## License

This project is licensed under the [MIT License](./LICENSE).

---

*Feel free to customize this README further to align with your project's specific details and requirements.*
