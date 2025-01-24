import argparse
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from color_vae_model import ColorVAE
from color_train import ColorImageDataset, train_color_vae
from color_vis import (create_latent_space_grid, create_latent_walk_animation, 
                      generate_tileable_samples, interpolate_samples)

def add_circular_padding_hooks(model):
    def make_circular_conv_hook(module, input):
        x = input[0]
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            # Add circular padding before the convolution
            pad_h = module.padding[0]
            pad_w = module.padding[1]
            if pad_h > 0 or pad_w > 0:
                x = torch.cat([x[:, :, -pad_h:, :], x, x[:, :, :pad_h, :]], dim=2)
                x = torch.cat([x[:, :, :, -pad_w:], x, x[:, :, :, :pad_w]], dim=3)
            return (x,)
        return input

    # Register hooks for all convolution layers
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(module.register_forward_pre_hook(make_circular_conv_hook))
    
    return hooks

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train or load a Color VAE model')
    parser.add_argument('--load', action='store_true', help='Load pre-trained model instead of training')
    parser.add_argument('--image-size', type=int, default=128, help='Image size (square)')
    parser.add_argument('--latent-dim', type=int, default=32, help='Dimension of latent space')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--circular-padding', action='store_true', help='Use circular padding for convolutions')
    args = parser.parse_args()

    # Setup paths
    image_dir = Path("color/both")
    model_path = Path(f"trained_color_vae_both_{args.image_size}.pth")
    
    # Check if image directory exists
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory '{image_dir}' not found")
    
    # Create dataset and dataloader
    dataset = ColorImageDataset(image_dir, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize VAE
    vae = ColorVAE(latent_dim=args.latent_dim, image_size=args.image_size)
    
    # Load or train the model
    if args.load and model_path.exists():
        vae.load_state_dict(torch.load(model_path))
        print("Loaded pre-trained model")
    else:
        print("Training new model...")
        vae = train_color_vae(vae, dataloader)
        torch.save(vae.state_dict(), model_path)
        print(f"Saved trained model to {model_path}")
    

    if args.circular_padding:
        hooks = add_circular_padding_hooks(vae)
    
    # Create output directories
    output_dir = Path("color_outputs_both")
    output_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    print("Creating latent space grid...")
    create_latent_space_grid(vae, save_path=output_dir/"latent_grid.png")
    
    print("Generating tileable samples...")
    generate_tileable_samples(vae, num_samples=16, output_dir=output_dir/"tiles")
    
    print("Creating latent space walk animation...")
    create_latent_walk_animation(vae, output_dir=output_dir/"walk_frames")
    
    print("Generating interpolations...")
    interpolate_samples(vae, save_path=output_dir/"interpolation.png")
    
    print(f"\nAll outputs saved to {output_dir}")

if __name__ == "__main__":
    main() 