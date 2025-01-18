import os
import math
from abc import abstractmethod

from PIL import Image
import requests
import numpy as np
import time
from gaussian_diffusion import GaussianDiffusion
from metrics import compute_fid, compute_is
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from unetmodel import UnetModel



device = "cuda" if torch.cuda.is_available() else "cpu"

### Loading the data

batch_size = 128
timesteps = 500

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# use MNIST dataset
dataset = datasets.CIFAR10(root='./dataset/cifar10/', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
real_loader_for_fid = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False, num_workers=4)



model = UnetModel(
    in_channels=1,
    model_channels=96,
    out_channels=1,
    channel_mult=(1, 2, 2),
    attention_resolutions=[],
    class_num=10
)
model.to(device)
gaussian_diffusion = GaussianDiffusion(timesteps=500, beta_schedule='linear')

# train
epochs = 10
p_uncound = 0.2
len_data = len(train_loader)
time_end = time.time()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Inside your training loop, after the epoch loop
for epoch in range(epochs):
    for step, (images, labels) in enumerate(train_loader):     
        time_start = time_end

        optimizer.zero_grad()

        batch_size = images.shape[0]
        images = images.to(device)
        labels = labels.to(device)

        # random generate mask
        z_uncound = torch.rand(batch_size)
        batch_mask = (z_uncound > p_uncound).int().to(device)

        # sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        loss = gaussian_diffusion.train_losses(model, images, t, labels, batch_mask)

        if step % 100 == 0:
            time_end = time.time()
            print("Epoch{}/{}\t  Step{}/{}\t Loss {:.4f}\t Time {:.2f}".format(
                epoch+1, epochs, step+1, len_data, loss.item(), time_end-time_start))
            
        loss.backward()
        optimizer.step()
    
    # Compute metrics at the end of each epoch
    print(f"Epoch {epoch+1} completed. Computing metrics...")
    is_score = compute_is(model, gaussian_diffusion, device, n_samples=500, batch_size=50)
    fid_score_value = compute_fid(model, gaussian_diffusion, device, real_loader_for_fid, n_samples=500, batch_size=50)
    print(f"Epoch {epoch+1} - Inception Score: {is_score:.4f}, FID: {fid_score_value:.4f}")
    
    # Optionally, save metrics to a log file
    with open('metrics_log.txt', 'a') as log_file:
        log_file.write(f"Epoch {epoch+1} - IS: {is_score:.4f}, FID: {fid_score_value:.4f}\n")
        

if not os.path.exists('./saved_models'):
    os.mkdir('./saved_models')
torch.save(model, './saved_models/Classifier_free_MNIST.h5')