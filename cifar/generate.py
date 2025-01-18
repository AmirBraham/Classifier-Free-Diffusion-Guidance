import os
from matplotlib import pyplot as plt
import torch
from gaussian_diffusion import GaussianDiffusion

model = torch.load('./saved_models/Classifier_free_CIFAR10.h5')
gaussian_diffusion = GaussianDiffusion(timesteps=500, beta_schedule='linear')
# Generate images using the diffusion model
generated_images = gaussian_diffusion.sample(
    model=model, 
    image_size=32, 
    batch_size=64, 
    channels=3, 
    n_class=10, 
    w=2, 
    mode='random', 
    clip_denoised=False
)

# Ensure the output directory exists
output_dir = "generated"
os.makedirs(output_dir, exist_ok=True)

# Take the last timestep of generated images
images = torch.tensor(generated_images[-1])  # Shape: [batch_size, channels, 32, 32]
images = (images + 1.0) / 2.0  # Rescale from [-1, 1] to [0, 1]

# Save generated images
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
gs = fig.add_gridspec(8, 8)  # Create an 8x8 grid for visualization

# Ensure the batch size is sufficient for 8x8 images (i.e., at least 64 images)
assert images.size(0) >= 64, f"Not enough images to create an 8x8 grid, got {images.size(0)}."
# Create a figure with a single 8x8 grid
fig, axes = plt.subplots(8, 8, figsize=(12, 12))
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# Plot images in the grid
for i in range(8):
    for j in range(8):
        idx = i * 8 + j
        img = images[idx].permute(1, 2, 0).cpu().numpy()  # Rearrange dimensions to [H, W, C]
        axes[i, j].imshow(img)
        axes[i, j].axis('off')  # Turn off axis labels

# Save the entire grid as a single image
plt.savefig(os.path.join(output_dir, 'generated.png'), 
            bbox_inches='tight', 
            pad_inches=0.1, 
            dpi=300)
plt.close()

print(f"Grid image successfully saved as 'generated.png' in '{output_dir}'")