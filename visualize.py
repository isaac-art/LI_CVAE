import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from pathlib import Path
from PIL import Image

def create_conditional_latent_space_anim(model, num_points: int=150, num_frames: int=30):
    device = next(model.parameters()).device
    model = model.to(device)
    x = np.linspace(-3, 3, num_points)
    y = np.linspace(-3, 3, num_points)
    grid_x, grid_y = np.meshgrid(x, y)
    num_classes = model.num_classes
    
    # Create output directory
    os.makedirs("interpolation_frames", exist_ok=True)
    
    # Generate frames interpolating between classes
    frame_count = 0
    for start_class in range(num_classes):
        end_class = (start_class + 1) % num_classes  # Loop back to first class
        
        # Generate frames interpolating between start and end class
        for frame in range(num_frames):
            alpha = frame / (num_frames - 1)  # Interpolation factor
            
            # Create one-hot encoded vectors for start and end classes
            start_one_hot = torch.zeros(1, num_classes, dtype=torch.float32).to(device)
            end_one_hot = torch.zeros(1, num_classes, dtype=torch.float32).to(device)
            start_one_hot[0, start_class] = 1
            end_one_hot[0, end_class] = 1
            
            # Smoothly interpolate between one-hot vectors
            interpolated_label = (1 - alpha) * start_one_hot + alpha * end_one_hot
            
            for i, (xi, yi) in enumerate(zip(grid_x[0], grid_y[0])):
                print(f"Generating frame {frame_count:04d}_{i:03d}")
                z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    # Pass interpolated label vector to decoder
                    generated = model.decode(z, interpolated_label)
                
                # Convert tensor to numpy array and scale to 0-255 range
                img = generated.cpu().squeeze().numpy()
                img = (img * 255).astype(np.uint8)
                
                # Save frame
                img_pil = Image.fromarray(img)
                img_pil.save(f"interpolation_frames/frame_{frame_count:04d}_{i:03d}.png")
            
            frame_count += 1

def create_conditional_latent_space_grid_to_files(model, num_points: int=150):
    device = next(model.parameters()).device
    model = model.to(device)
    x = np.linspace(-3, 3, num_points)
    y = np.linspace(-3, 3, num_points)
    grid_x, grid_y = np.meshgrid(x, y)
    num_classes = model.num_classes
    
    for label in range(num_classes):
        for i, (xi, yi) in enumerate(zip(grid_x[0], grid_y[0])):
            z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
            label_tensor = torch.tensor([label], dtype=torch.long).to(device)
            with torch.no_grad():
                generated = model.decode(z, label_tensor)
            
            # Convert tensor to numpy array and scale to 0-255 range
            img = generated.cpu().squeeze().numpy()
            img = (img * 255).astype(np.uint8)
            
            # Create directory if it doesn't exist
            os.makedirs(f"grids/{label}", exist_ok=True)
            
            # Save image using PIL
            img_pil = Image.fromarray(img)
            img_pil.save(f"grids/{label}/{i:03d}.png")

def create_conditional_latent_space_grid(model, num_points: int = 15):
    # Move model to the same device as its parameters
    device = next(model.parameters()).device
    model = model.to(device)
    
    # Create a grid of points in latent space
    x = np.linspace(-3, 3, num_points)
    y = np.linspace(-3, 3, num_points)
    grid_x, grid_y = np.meshgrid(x, y)
    
    # Create a grid of images for each class
    num_classes = model.num_classes
    fig, axes = plt.subplots(num_classes, num_points, 
                            figsize=(2*num_points, 2*num_classes))
    
    for label in range(num_classes):
        # Create latent vectors for this row
        for i, (xi, yi) in enumerate(zip(grid_x[0], grid_y[0])):
            # Create latent vector and move to correct device
            z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
            # Create label tensor and move to correct device
            label_tensor = torch.tensor([label], dtype=torch.long).to(device)
            
            # Generate image
            with torch.no_grad():
                generated = model.decode(z, label_tensor)
            
            # Plot
            if num_classes > 1:
                ax = axes[label, i]
            else:
                ax = axes[i]
            
            # Move tensor to CPU for plotting
            img = generated.cpu().squeeze().numpy()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
            # Add label on the left side
            if i == 0:
                ax.set_title(f'Class {label}')
    
    plt.tight_layout()
    plt.show()

def visualize_class_interpolation(model, class1: int, class2: int, num_steps: int = 10):
    # Move model to the same device as its parameters
    device = next(model.parameters()).device
    model = model.to(device)
    
    # Create fixed latent vector
    z = torch.randn(1, model.latent_dim).to(device)
    
    # Create interpolated labels
    alphas = np.linspace(0, 1, num_steps)
    fig, axes = plt.subplots(1, num_steps, figsize=(2*num_steps, 2))
    
    for i, alpha in enumerate(alphas):
        # Create interpolated one-hot label
        label = torch.tensor([class1 if alpha < 0.5 else class2], dtype=torch.long).to(device)
        
        # Generate image
        with torch.no_grad():
            generated = model.decode(z, label)
        
        # Plot
        img = generated.cpu().squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'{alpha:.1f}')
    
    plt.suptitle(f'Interpolation: Class {class1} → Class {class2}')
    plt.tight_layout()
    plt.show() 

def visualize_class_samples(model, samples_per_row: int = 8, num_rows: int = 8):
    """
    Creates a grid visualization of random samples for each class.
    Shows samples_per_row × num_rows samples for each class.
    """
    # Move model to the same device as its parameters
    device = next(model.parameters()).device
    model = model.to(device)
    
    num_classes = model.num_classes
    samples_per_class = samples_per_row * num_rows
    
    # Create figure with a subplot for each class
    fig, axes = plt.subplots(num_classes, 1, 
                            figsize=(2*samples_per_row, 2*num_rows*num_classes))
    if num_classes == 1:
        axes = [axes]
    
    for class_idx in range(num_classes):
        # Create subplot grid for this class
        class_fig = plt.subplot2grid((num_classes, 1), (class_idx, 0))
        plt.title(f'Class {class_idx} Samples')
        
        # Generate random samples for this class
        z = torch.randn(samples_per_class, model.latent_dim).to(device)
        labels = torch.full((samples_per_class,), class_idx, dtype=torch.long).to(device)
        
        # Generate images
        with torch.no_grad():
            generated = model.decode(z, labels)
        
        # Create grid of images
        for idx in range(samples_per_class):
            plt.subplot2grid(
                (num_rows, samples_per_row),
                (idx // samples_per_row, idx % samples_per_row),
                fig=class_fig.get_figure()
            )
            img = generated[idx].cpu().squeeze().numpy()
            plt.imshow(img, cmap='gray')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show() 

def generate_and_save_class_samples(model, output_dir: str, samples_per_class: int = 100):
    """
    Generates and saves random samples for each class.
    
    Args:
        model: The VAE model
        output_dir: Directory to save the generated images
        samples_per_class: Number of samples to generate per class
    """
    # Move model to the same device as its parameters
    device = next(model.parameters()).device
    model = model.to(device)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for class_idx in range(model.num_classes):
        # Create class directory
        class_dir = output_path / str(class_idx)
        class_dir.mkdir(exist_ok=True)
        
        # Generate random samples for this class
        z = torch.randn(samples_per_class, model.latent_dim).to(device)
        labels = torch.full((samples_per_class,), class_idx, dtype=torch.long).to(device)
        
        # Generate images with thresholding
        with torch.no_grad():
            generated = model.decode(z, labels, apply_threshold=True)
        
        # Save each image
        for idx in range(samples_per_class):
            img = generated[idx].cpu().squeeze().numpy()
            # Scale to 0-255 range and convert to uint8
            img = (img * 255).astype(np.uint8)
            plt.imsave(
                class_dir / f"{idx}.png",
                img,
                cmap='gray'
            )

def generate_and_save_class_walk(model, output_dir: str, steps: int = 100):
    """
    Generates a circular walk through the latent space for each class.
    
    Args:
        model: The VAE model
        output_dir: Directory to save the generated images
        steps: Number of steps in the circular walk
    """
    # Move model to the same device as its parameters
    device = next(model.parameters()).device
    model = model.to(device)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate points in a circle
    theta = torch.linspace(0, 2*torch.pi, steps)
    radius = 2.0  # Radius of the circle in latent space
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    
    # Combine into latent vectors
    z = torch.stack([x, y], dim=1).to(device)
    
    for class_idx in range(model.num_classes):
        # Create class directory
        class_dir = output_path / f"walk_{class_idx}"
        class_dir.mkdir(exist_ok=True)
        
        # Generate images for this class with thresholding
        labels = torch.full((steps,), class_idx, dtype=torch.long).to(device)
        
        with torch.no_grad():
            generated = model.decode(z, labels, apply_threshold=True)
        
        # Save each image
        for idx in range(steps):
            img = generated[idx].cpu().squeeze().numpy()
            # Scale to 0-255 range and convert to uint8
            img = (img * 255).astype(np.uint8)
            plt.imsave(
                class_dir / f"{idx:03d}.png",
                img,
                cmap='gray'
            )