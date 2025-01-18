import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import requests
import numpy as np

from gaussian_diffusion import GaussianDiffusion
from metrics import compute_fid, compute_is
from unetmodel import UnetModel

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters and configurations
batch_size = 128
timesteps = 500
epochs = 20
learning_rate = 1e-4

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load CIFAR10 dataset
dataset = datasets.CIFAR10(root='./dataset/cifar10/', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
real_loader_for_fid = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False, num_workers=4)

# Initialize Gaussian Diffusion
gaussian_diffusion = GaussianDiffusion(timesteps=timesteps, beta_schedule='linear')

# Function to train the model for a given p_uncound
def train_model(p_uncound, save_model=True):
    print(f"Starting training with p_uncound = {p_uncound}")

    # Initialize the model
    model = UnetModel(
        in_channels=3,  # Change to 1 if using MNIST
        model_channels=96,
        out_channels=3,
        channel_mult=(1, 2, 2),
        attention_resolutions=[],
        class_num=10
    )
    model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    len_data = len(train_loader)
    time_end = time.time()

    for epoch in range(epochs):
        model.train()
        for step, (images, labels) in enumerate(train_loader):
            time_start = time_end

            optimizer.zero_grad()

            batch_size_current = images.shape[0]
            images = images.to(device)
            labels = labels.to(device)

            # Randomly generate mask
            z_uncound = torch.rand(batch_size_current, device=device)
            batch_mask = (z_uncound > p_uncound).int()

            # Sample t uniformly for every example in the batch
            t = torch.randint(0, timesteps, (batch_size_current,), device=device).long()

            # Compute loss
            loss = gaussian_diffusion.train_losses(model, images, t, labels, batch_mask)

            # Logging
            if step % 100 == 0:
                time_end = time.time()
                print(f"Epoch {epoch+1}/{epochs}\t Step {step+1}/{len_data}\t Loss {loss.item():.4f}\t Time {time_end - time_start:.2f}s")

            # Backpropagation
            loss.backward()
            optimizer.step()

    # Compute metrics after training
    print(f"Training completed for p_uncound = {p_uncound}. Computing metrics...")
    model.eval()
    with torch.no_grad():
        is_score = compute_is(model, gaussian_diffusion, device, n_samples=500, batch_size=50)
        fid_score_value = compute_fid(model, gaussian_diffusion, device, real_loader_for_fid, n_samples=500, batch_size=50)
    print(f"p_uncound = {p_uncound} - Inception Score: {is_score:.4f}, FID: {fid_score_value:.4f}")

    # Optionally, save the model
    if save_model:
        model_save_path = f'./saved_models/Classifier_free_CIFAR10_puncound_{p_uncound}.pth'
        os.makedirs('./saved_models', exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    return is_score, fid_score_value

# Define the range of p_uncound values you want to experiment with
p_uncound_values = [0.0, 0.1, 0.2, 0.4, 0.5,0.9]

# Lists to store metrics
inception_scores = []
fid_scores = []

# Optional: Clear previous metrics log
metrics_log_path = 'metrics_log.txt'
if os.path.exists(metrics_log_path):
    os.remove(metrics_log_path)

# Directory to save plots
plots_dir = './plots'
os.makedirs(plots_dir, exist_ok=True)

# Iterate over different p_uncound values
for p_uncound in p_uncound_values:
    is_score, fid_score = train_model(p_uncound)
    inception_scores.append(is_score)
    fid_scores.append(fid_score)

    # Save metrics to a log file
    with open(metrics_log_path, 'a') as log_file:
        log_file.write(f"p_uncound {p_uncound} - IS: {is_score:.4f}, FID: {fid_score:.4f}\n")

# Plotting the results
plt.figure(figsize=(12, 5))

# Plot Inception Score
plt.subplot(1, 2, 1)
plt.plot(p_uncound_values, inception_scores, marker='o', linestyle='-', color='b')
plt.title('Inception Score vs p_uncound')
plt.xlabel('p_uncound')
plt.ylabel('Inception Score')
plt.grid(True)

# Plot FID Score
plt.subplot(1, 2, 2)
plt.plot(p_uncound_values, fid_scores, marker='o', linestyle='-', color='r')
plt.title('FID Score vs p_uncound')
plt.xlabel('p_uncound')
plt.ylabel('FID Score')
plt.grid(True)

plt.tight_layout()

# Save the plots instead of showing them
inception_plot_path = os.path.join(plots_dir, 'inception_score_vs_p_uncound.png')
fid_plot_path = os.path.join(plots_dir, 'fid_score_vs_p_uncound.png')

# Save each plot individually
# Saving Inception Score plot
plt.figure(figsize=(6, 5))
plt.plot(p_uncound_values, inception_scores, marker='o', linestyle='-', color='b')
plt.title('Inception Score vs p_uncound')
plt.xlabel('p_uncound')
plt.ylabel('Inception Score')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'inception_score_vs_p_uncound.png'))
plt.close()

# Saving FID Score plot
plt.figure(figsize=(6, 5))
plt.plot(p_uncound_values, fid_scores, marker='o', linestyle='-', color='r')
plt.title('FID Score vs p_uncound')
plt.xlabel('p_uncound')
plt.ylabel('FID Score')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'fid_score_vs_p_uncound.png'))
plt.close()

print(f"Plots saved in the directory: {plots_dir}")