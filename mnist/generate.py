from matplotlib import pyplot as plt
import torch
from gaussian_diffusion import GaussianDiffusion

model = torch.load('./saved_models/Classifier_free_MNIST.h5')
gaussian_diffusion = GaussianDiffusion(timesteps=500, beta_schedule='linear')

generated_images = gaussian_diffusion.sample(model, 28, batch_size=64, channels=1, n_class=10, w=2, mode='random', clip_denoised=False)


# generate new images
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
gs = fig.add_gridspec(8, 8)

imgs = generated_images[-1].reshape(8, 8, 28, 28)
for n_row in range(8):
    for n_col in range(8):
        plt.imsave(f"generated/generated_image_{n_row}_{n_col}.png",(imgs[n_row, n_col]+1.0) * 255 / 2)