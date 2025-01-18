'''
The code is modified from
https://github.com/xiaohu2015/nngen/blob/main/models/diffusion_models,
https://github.com/TeaPearce/Conditional_Diffusion_MNIST,
https://www.bilibili.com/video/BV1b541197HX/

Diffusion model is based on "CLASSIFIER-FREE DIFFUSION GUIDANCE" and "Denoising Diffusion Implicit Models"
https://arxiv.org/abs/2207.12598,
https://arxiv.org/abs/2010.02502
'''

import os
import math
from abc import abstractmethod

from PIL import Image
import requests
import numpy as np
import time
from gaussian_diffusion import GaussianDiffusion
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from unetmodel import UnetModel





    
    


    
    
    
batch_size = 128
timesteps = 500

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# use MNIST dataset
dataset = datasets.CIFAR10(root='./dataset/cifar10/', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# define model and diffusion
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UnetModel(
    in_channels=3,
    model_channels=96,
    out_channels=3,
    channel_mult=(1, 2, 2),
    attention_resolutions=[],
    class_num=10
)
model.to(device)


image = next(iter(train_loader))[0][1].squeeze()
label = next(iter(train_loader))[1][1].squeeze()

x_start = image

gaussian_diffusion = GaussianDiffusion(timesteps=500, beta_schedule='linear')

plt.figure(figsize=(16, 5))
for idx, t in enumerate([0, 100, 300, 400, 499]):
    
    # Generate noisy image at timestep t
    x_noisy = gaussian_diffusion.q_sample(x_start.to(device), t=torch.tensor([t]).to(device))
    noisy_image = (x_noisy.squeeze() + 1) * 127.5  # Scale to [0, 255]
    
    # For t=0, use the original image without noise
    if idx == 0:
        noisy_image = (x_start.squeeze() + 1) * 127.5
    
    # Convert to CPU and NumPy array
    noisy_image = noisy_image.cpu().numpy().astype(np.uint8)
    
    # Transpose from (C, H, W) to (H, W, C)
    noisy_image = np.transpose(noisy_image, (1, 2, 0))
    
    # Plot the image
    plt.subplot(1, 5, 1 + idx)
    plt.imshow(noisy_image)
    plt.axis("off")
    plt.title(f"t={t}")
    
    # Save the image
    plt.imsave(f"noisy_image{t}.png", noisy_image)