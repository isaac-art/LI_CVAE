import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import random

class ColorImageDataset(Dataset):
    def __init__(self, image_dir: str, image_size: int = 128):
        self.image_paths = []
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        
        # Build dataset with original images
        self.image_paths.extend(list(self.image_dir.rglob("*.png")))
        self.image_paths.extend(list(self.image_dir.rglob("*.jpg")))
        self.image_paths.extend(list(self.image_dir.rglob("*.jpeg")))
        
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"\nDataset initialization complete:")
        print(f"Total images: {len(self.image_paths)}")
        
        # Transform for images
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)

def train_color_vae(model: nn.Module, 
                    dataloader: DataLoader, 
                    num_epochs: int = 1000,
                    learning_rate: float = 1e-4):
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training parameters
    best_loss = float('inf')
    patience = 50
    patience_counter = 0
    min_epochs = 100
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        recon_loss_sum = 0
        kl_loss_sum = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            x = batch.to(device)
            optimizer.zero_grad()
            
            x_hat, mu, log_var = model(x)
            
            # Reconstruction loss (MSE for color images)
            reconstruction_loss = F.mse_loss(x_hat, x, reduction='sum')
            
            # KL divergence
            kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Dynamic beta scheduling
            beta = min(1.0, (epoch * 1.0) / 50)  # Gradually increase beta over first 50 epochs
            loss = reconstruction_loss + beta * kld
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            recon_loss_sum += reconstruction_loss.item()
            kl_loss_sum += kld.item()
            
            progress_bar.set_postfix({
                'loss': f"{loss.item()/len(x):.4f}",
                'recon': f"{reconstruction_loss.item()/len(x):.4f}",
                'kl': f"{kld.item()/len(x):.4f}",
                'beta': f"{beta:.3f}"
            })
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader.dataset)
        scheduler.step(avg_loss)
        
        # Early stopping check
        if epoch >= min_epochs:
            if avg_loss < best_loss * 0.995:  # Allow 0.5% improvement to count as better
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), "best_color_vae.pth")
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