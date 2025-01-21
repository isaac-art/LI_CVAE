import argparse
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from vae_model import VAE
from train import ImageDataset, train_vae
from visualize import create_conditional_latent_space_grid, visualize_class_interpolation, visualize_class_samples, generate_and_save_class_samples, generate_and_save_class_walk, create_conditional_latent_space_anim

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train or load a VAE model')
    parser.add_argument('--load', action='store_true', help='Load pre-trained model instead of training')
    parser.add_argument('--image-size', type=int, default=128, help='Image size (square)')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for final sigmoid (lower = sharper)')
    args = parser.parse_args()

    image_dir = Path(f"images_{args.image_size}")
    model_path = Path(f"trained_vae_{args.image_size}.pth")  # Include size in filename
    
    # Check if image directory exists
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory '{image_dir}' not found")
    
    # First, get all unique class names (directory names)
    class_names = sorted(list({p.parent.name for p in image_dir.rglob("*.png")}))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    print("\nFound classes:")
    for class_name, idx in class_to_idx.items():
        print(f"{class_name}: {idx}")
    
    # Define labels for your images based on their directory
    labels = {}
    for img_path in image_dir.rglob("*.png"):
        class_name = img_path.parent.name
        relative_path = str(img_path.relative_to(image_dir))
        labels[relative_path] = class_to_idx[class_name]
        print(f"Added image: {relative_path} with label: {class_to_idx[class_name]} ({class_name})")
    
    if not labels:
        raise ValueError("No images found in the specified directory")
    
    print(f"\nFound {len(labels)} images across {len(class_to_idx)} classes")
    
    # Create dataset and dataloader with specified image size
    dataset = ImageDataset(image_dir, labels, image_size=args.image_size, augment_ratio=0.05)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize VAE with specified image size
    num_classes = len(class_to_idx)
    print(f"Number of classes: {num_classes}")
    vae = VAE(latent_dim=2, 
              num_classes=num_classes, 
              image_size=args.image_size,
              temperature=args.temperature)
    
    # Load or train the model
    if args.load and model_path.exists():
        vae.load_state_dict(torch.load(model_path))
        print("Loaded pre-trained model")
    else:
        vae = train_vae(vae, dataloader)  # Update the same vae variable
        torch.save(vae.state_dict(), model_path)
        print(f"Saved trained model to {model_path}")
    
    # Store class names for later use
    class_names_path = Path("class_names.txt")
    with open(class_names_path, "w") as f:
        for i, name in enumerate(class_names):
            f.write(f"{i}: {name}\n")
    
    create_conditional_latent_space_anim(vae)

    # # After the existing visualizations, add:
    # print("\nGenerating and saving class samples...")
    # generate_and_save_class_samples(vae, "outputs")
    # print("Saved generated samples to 'outputs' directory")

    # print("\nGenerating and saving class walk...")
    # generate_and_save_class_walk(vae, "walks")
    # print("Saved generated walk to 'outputs' directory")

    # visualize_class_samples(vae, samples_per_rqow=8, num_rows=8)

    # Create and display the latent space visualization
    # create_conditional_latent_space_grid_to_files(vae)  # Use vae instead of trained_vae
    
    # Visualize interpolations between classes
    # print("\nGenerating class interpolations...")
    
    # # Interpolate between each consecutive pair of classes
    # for i in range(num_classes - 1):
    #     visualize_class_interpolation(vae, i, i + 1)  




if __name__ == "__main__":
    main() 