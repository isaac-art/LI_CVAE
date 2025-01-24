import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from pathlib import Path
from PIL import Image
import torchvision.utils as vutils

def create_latent_space_grid(model, num_points: int = 15, save_path: str = None):
    device = next(model.parameters()).device
    model = model.to(device)
    
    # Create a grid of points in latent space
    x = np.linspace(-3, 3, num_points).astype(np.float32)
    y = np.linspace(-3, 3, num_points).astype(np.float32)
    grid_x, grid_y = np.meshgrid(x, y)
    
    # Generate images for each point in the grid
    images = []
    for i in range(num_points):
        row_images = []
        for j in range(num_points):
            # Create latent vector using first two dimensions
            z = torch.zeros(1, model.latent_dim, device=device, dtype=torch.float32)
            z[0, 0] = torch.tensor(grid_x[i, j], device=device, dtype=torch.float32)
            z[0, 1] = torch.tensor(grid_y[i, j], device=device, dtype=torch.float32)
            
            with torch.no_grad():
                generated = model.decode(z)
            row_images.append(generated)
        
        # Concatenate images in this row
        row_tensor = torch.cat(row_images, dim=0)
        images.append(row_tensor)
    
    # Concatenate all rows
    grid_tensor = torch.cat(images, dim=0)
    
    # Create a grid visualization
    grid_image = vutils.make_grid(grid_tensor, nrow=num_points, padding=2, normalize=True)
    
    if save_path:
        vutils.save_image(grid_image, save_path)
    
    # Display the grid
    plt.figure(figsize=(15, 15))
    plt.imshow(grid_image.cpu().permute(1, 2, 0))
    plt.axis('off')
    plt.show()

def create_latent_walk_animation(model, num_frames: int = 100, radius: float = 2.0, 
                               output_dir: str = "latent_walk_frames"):
    device = next(model.parameters()).device
    model = model.to(device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate points in a circle
    theta = np.linspace(0, 2*np.pi, num_frames)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    # Generate frames
    for i in tqdm(range(num_frames), desc="Generating frames"):
        # Create latent vector
        z = torch.zeros(1, model.latent_dim, device=device)
        z[0, 0] = torch.tensor(x[i], device=device)
        z[0, 1] = torch.tensor(y[i], device=device)
        
        with torch.no_grad():
            generated = model.decode(z)
        
        # Save frame
        vutils.save_image(
            generated,
            f"{output_dir}/frame_{i:04d}.png",
            normalize=True
        )

def generate_tileable_samples(model, num_samples: int = 16, output_dir: str = "tiles"):
    device = next(model.parameters()).device
    model = model.to(device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate samples
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim, device=device)
        generated = model.decode(z)
    
    # Save individual tiles
    for i in range(num_samples):
        vutils.save_image(
            generated[i],
            f"{output_dir}/tile_{i:03d}.png",
            normalize=True
        )
    
    # Create a grid visualization
    grid = vutils.make_grid(generated, nrow=int(np.sqrt(num_samples)), padding=2, normalize=True)
    vutils.save_image(grid, f"{output_dir}/grid.png")
    
    # Create 2x2 tiled versions
    for i in range(num_samples):
        tile = generated[i]
        # Create 2x2 tiling
        tiled = torch.cat([torch.cat([tile, tile], dim=2), 
                          torch.cat([tile, tile], dim=2)], dim=1)
        vutils.save_image(
            tiled,
            f"{output_dir}/tiled_{i:03d}.png",
            normalize=True
        )

def interpolate_samples(model, num_steps: int = 10, save_path: str = None):
    device = next(model.parameters()).device
    model = model.to(device)
    
    # Generate two random latent vectors
    z1 = torch.randn(1, model.latent_dim, device=device)
    z2 = torch.randn(1, model.latent_dim, device=device)
    
    # Generate interpolated images
    interpolated = model.interpolate(z1, z2, num_steps)
    
    # Create a grid visualization
    grid = vutils.make_grid(interpolated, nrow=num_steps, padding=2, normalize=True)
    
    if save_path:
        vutils.save_image(grid, save_path)
    
    # Display the grid
    plt.figure(figsize=(20, 4))
    plt.imshow(grid.cpu().permute(1, 2, 0))
    plt.axis('off')
    plt.show() 