import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import matplotlib.animation as animation
from vae_model import VAE
from collections import deque

def load_model(model_path: str) -> VAE:
    """Load the trained VAE model."""
    model = VAE(latent_dim=2,
                num_classes=20,
                image_size=128,
                temperature=0.1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def generate_random_walk(
    start_points: torch.Tensor,
    n_steps: int,
    step_size: float = 0.1
) -> torch.Tensor:
    """Generate random walk trajectories for multiple points."""
    batch_size, latent_dim = start_points.size()
    trajectory = torch.zeros((n_steps, batch_size, latent_dim))
    current_points = start_points
    
    for i in range(n_steps):
        # Generate random directions for all points
        directions = torch.randn_like(current_points)
        directions = directions / directions.norm(dim=1, keepdim=True)
        
        # Take steps in those directions
        current_points = current_points + step_size * directions
        trajectory[i] = current_points
    
    return trajectory

def interpolate_latent_space(model: VAE, trajectory: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
    """Generate images from latent space points."""
    with torch.no_grad():
        return model.decode(trajectory, class_labels)

def main():
    # Setup
    model_path = "trained_vae_128.pth"
    model = load_model(model_path)
    device = next(model.parameters()).device
    
    # Initialize the plot with 5x4 grid for 20 classes
    plt.ion()
    fig, axes = plt.subplots(5, 4, figsize=(5, 5))
    plt.show(block=False)
    
    # Load class names if available
    try:
        with open("class_names.txt", "r") as f:
            class_names = [line.strip() for line in f]
    except FileNotFoundError:
        class_names = [f"Class {i}" for i in range(20)]
    
    # Generate initial random points in latent space
    num_classes = 20
    latent_dim = 2
    current_points = torch.randn(num_classes, latent_dim, device=device)
    
    # Create class labels tensor
    class_labels = torch.arange(num_classes, device=device)
    
    # FPS tracking
    frame_times = deque(maxlen=100)
    last_time = time.time()
    frames = 0
    
    while True:
        try:
            # Generate random walk trajectories for all points
            trajectories = generate_random_walk(current_points, n_steps=2)
            trajectories = trajectories.to(device)
            
            # Generate images from the new points
            generated_images = interpolate_latent_space(model, trajectories[-1], class_labels)
            
            # Update the current points
            current_points = trajectories[-1]
            
            # Display all images in the grid
            for idx, (ax, img, class_name) in enumerate(zip(axes.flat, generated_images, class_names)):
                ax.clear()
                ax.imshow(img.squeeze().cpu().numpy(), cmap='gray')
                ax.axis('off')
                ax.set_title(class_name, fontsize=8)
            
            plt.tight_layout()
            
            # Update FPS counter
            current_time = time.time()
            frame_time = current_time - last_time
            frame_times.append(frame_time)
            frames += 1
            
            # if frames % 10 == 0:
            #     avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
            #     print(f'FPS: {avg_fps:.1f}')
            
            plt.pause(0.001)
            last_time = current_time
            
        except KeyboardInterrupt:
            print("\nStopping the latent walks...")
            if frame_times:
                avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
                print(f"Average FPS: {avg_fps:.1f}")
            break
    
    plt.ioff()
    plt.close()

if __name__ == "__main__":
    main() 