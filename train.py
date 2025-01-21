import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import functional as TF
import random

class RandomSkew:
    def __init__(self, angle_range=20):
        self.angle_range = angle_range
    
    def __call__(self, img):
        angle = random.uniform(-self.angle_range, self.angle_range)
        return TF.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=angle)

class ImageDataset(Dataset):
    def __init__(self, image_dir: str, labels: dict[str, int], 
                 augment_ratio: float = 0.1, image_size: int = 128):
        self.image_paths = []
        self.labels = labels
        self.image_dir = Path(image_dir)
        self.is_augmented = []  # Track which samples are augmented
        self.image_size = image_size
        
        # Verify we have valid labels
        unique_labels = set(labels.values())
        if not all(isinstance(label, int) and label >= 0 for label in unique_labels):
            raise ValueError("All labels must be non-negative integers")
        
        print(f"\nInitializing dataset:")
        print(f"Base directory: {self.image_dir}")
        print(f"Number of labels: {len(labels)}")
        
        # Build dataset with original images
        original_paths = []
        for rel_path_str, label in labels.items():
            full_path = self.image_dir / rel_path_str
            if full_path.exists():
                original_paths.append(full_path)
                print(f"Added: {rel_path_str} (label: {label})")
            else:
                print(f"Warning: File not found: {rel_path_str}")
        
        # Add original images
        self.image_paths.extend(original_paths)
        self.is_augmented.extend([False] * len(original_paths))
        
        # Add augmented versions
        num_augmented = int(len(original_paths) * augment_ratio)
        augment_indices = random.sample(range(len(original_paths)), num_augmented)
        for idx in augment_indices:
            self.image_paths.append(original_paths[idx])
            self.is_augmented.append(True)
        
        print(f"\nDataset initialization complete:")
        print(f"Original images: {len(original_paths)}")
        print(f"Augmented images: {num_augmented}")
        print(f"Total images: {len(self.image_paths)}")
        
        # Define augmentation transforms
        self.augment_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.Lambda(lambda x: x.point(lambda p: 0 if p < 190 else 255, '1')),
            transforms.RandomRotation(10, fill=255),  # Set fill to white (255)
            RandomSkew(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                fill=255  # Set fill to white (255)
            ),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
        ])
        
        # Transform for non-augmented images
        self.basic_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
        ])
        
        print(f"Dataset initialized with image size: {image_size}x{image_size}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        relative_path = str(img_path.relative_to(self.image_dir))
        label = self.labels[relative_path]
        
        # Load and convert image to grayscale
        image = Image.open(img_path).convert('L')
        
        # Apply transforms based on whether this is an augmented sample
        if self.is_augmented[idx]:
            image = self.augment_transform(image)
        else:
            image = self.basic_transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

def train_vae(model: nn.Module, 
              dataloader: DataLoader, 
              num_epochs: int = 5000,
              learning_rate: float = 1e-4):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # More lenient early stopping parameters
    best_loss = float('inf')
    patience = 500 
    patience_counter = 0
    min_epochs = 500  # Ensure we train for at least this many epochs
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        recon_loss_sum = 0
        kl_loss_sum = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch, labels in progress_bar:
            x = batch.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            x_hat, mu, log_var = model(x, labels)
            
            # Modified loss calculation with dynamic beta
            reconstruction_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
            edge_loss = model.edge_loss(x, x_hat)
            kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Dynamic beta scheduling
            beta = min(1.0, (epoch * 1.0) / 50)  # Gradually increase beta over first 100 epochs
            loss = reconstruction_loss + beta * kld + 0.1 * edge_loss
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            recon_loss_sum += reconstruction_loss.item()
            kl_loss_sum += kld.item()
            
            progress_bar.set_postfix({
                'loss': f"{loss.item()/len(batch):.4f}",
                'recon': f"{reconstruction_loss.item()/len(batch):.4f}",
                'kl': f"{kld.item()/len(batch):.4f}",
                'beta': f"{beta:.3f}"
            })
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader.dataset)
        scheduler.step(avg_loss)
        
        # Modified early stopping check
        if epoch >= min_epochs:  # Only start checking after min_epochs
            if avg_loss < best_loss * 0.995:  # Allow 0.5% improvement to count as better
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Reconstruction Loss: {recon_loss_sum/len(dataloader.dataset):.4f}")
        print(f"KL Loss: {kl_loss_sum/len(dataloader.dataset):.4f}")
    
    return model 

