from torchmetrics.image.inception  import InceptionScore
from pytorch_fid import fid_score
import torch
from torchvision import transforms
import tqdm
import os
import shutil


from torchmetrics.image.inception import InceptionScore
from pytorch_fid import fid_score
import torch
from torchvision import transforms
from tqdm import tqdm
import os
import shutil

# Preprocessing for Inception Score
preprocess_is = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1-channel to 3-channel
    transforms.Lambda(lambda x: (x * 255).to(torch.uint8))  # Convert to uint8
])

# Preprocessing for FID
preprocess_fid = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1-channel to 3-channel
    # No normalization needed for FID
])

def compute_is(model, diffusion, device, n_samples=200, batch_size=50):
    model.eval()
    is_metric = InceptionScore().to(device)
    
    with torch.no_grad():
        for _ in tqdm(range(n_samples // batch_size), desc='Computing Inception Score'):
            # Generate samples
            samples = diffusion.sample(model, image_size=28, batch_size=batch_size, channels=1, n_class=10, mode='random')
            # Convert list of numpy arrays to a single tensor (take the last step)
            samples = torch.tensor(samples[-1])  # Shape: [batch_size, 1, 28, 28]
            samples = (samples + 1) / 2  # Rescale from [-1, 1] to [0, 1]
            
            # Apply preprocessing for IS
            preprocessed = torch.stack([preprocess_is(sample) for sample in samples])
            preprocessed = preprocessed.to(device)
            
            # Update IS metric
            is_metric.update(preprocessed)
    
    inception_mean,inception_std = is_metric.compute()
    model.train()
    return inception_mean

def compute_fid(model, diffusion, device, real_loader, n_samples=500, batch_size=50, fid_dir='fid_tmp'):
    model.eval()
    
    # Create directories for generated and real images
    gen_dir = os.path.join(fid_dir, 'generated')
    real_dir = os.path.join(fid_dir, 'real')
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)
    
    # Generate and save samples
    with torch.no_grad():
        for i in tqdm(range(n_samples // batch_size), desc='Generating samples for FID'):
            samples = diffusion.sample(model, image_size=28, batch_size=batch_size, channels=1, n_class=10, mode='random')
            samples = torch.tensor(samples[-1])  # Shape: [batch_size, 1, 28, 28]
            samples = (samples + 1) / 2  # Rescale to [0,1]
            samples = samples.cpu()
            
            for j in range(samples.size(0)):
                img = samples[j]  # Shape: [1, 28, 28]
                img = preprocess_fid(img)  # Apply FID-specific preprocessing
                img = img.clamp(0, 1)  # Ensure values are in [0,1]
                
                # Convert to PIL Image and save
                img_pil = transforms.ToPILImage()(img)
                img_pil.save(os.path.join(gen_dir, f'gen_{i * batch_size + j}.png'))
    
    # Save real images
    count = 0
    with torch.no_grad():
        for images, _ in real_loader:
            images = (images + 1) / 2  # Rescale to [0,1]
            for img in images:
                img = preprocess_fid(img)  # Apply FID-specific preprocessing
                img = img.clamp(0, 1)
                img_pil = transforms.ToPILImage()(img)
                img_pil.save(os.path.join(real_dir, f'real_{count}.png'))
                count += 1
                if count >= n_samples:
                    break
            if count >= n_samples:
                break
    
    # Compute FID using pytorch-fid
    fid_value = fid_score.calculate_fid_given_paths([real_dir, gen_dir],
                                                   batch_size=batch_size,
                                                   device=device,
                                                   dims=2048)
    
    # Clean up temporary directories
    shutil.rmtree(fid_dir)
    
    model.train()
    return fid_value