import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class SharpenedSigmoid(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, x):
        return torch.sigmoid(x / self.temperature)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 2, num_classes: int = 2, image_size: int = 128, temperature: float = 0.1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.to(self.device)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Calculate number of convolution layers needed
        self.num_conv_layers = int(np.log2(image_size)) - 2  # For 128->5, 256->6, 512->7
        initial_size = image_size // (2 ** self.num_conv_layers)  # Size after all convolutions
        
        # Encoder
        encoder_layers = []
        current_channels = 1 + num_classes
        for i in range(self.num_conv_layers):
            out_channels = min(512, 32 * (2 ** i))
            encoder_layers.extend([
                nn.ReflectionPad2d(1),
                nn.Conv2d(current_channels, out_channels, 4, stride=2, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            current_channels = out_channels
        
        encoder_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate flattened dimension
        self.flatten_dim = current_channels * initial_size * initial_size
        
        # Latent space
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_var = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim + num_classes, self.flatten_dim)
        
        # Decoder layers
        decoder_layers = []
        for i in range(self.num_conv_layers):
            in_channels = current_channels
            out_channels = min(512, 32 * (2 ** (self.num_conv_layers - i - 2)))
            if i == self.num_conv_layers - 1:  # Last layer
                out_channels = 1
                decoder_layers.extend([
                    nn.ReflectionPad2d(1),
                    nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=0),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(out_channels, out_channels, 3, padding=0),
                    SharpenedSigmoid(temperature)
                ])
            else:
                decoder_layers.extend([
                    nn.ReflectionPad2d(1),
                    nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=0),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True)
                ])
            current_channels = out_channels
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Add threshold parameter for inference
        self.threshold = 0.5
        
        print(f"Initialized VAE with image size {image_size}x{image_size}")
        print(f"Number of conv layers: {self.num_conv_layers}")
        print(f"Flattened dimension: {self.flatten_dim}")
        print(f"Temperature: {temperature}")
    
    def _one_hot_encode(self, label):
        # Ensure labels are in range [0, num_classes-1]
        label = label.clamp(0, self.num_classes - 1)
        one_hot = torch.zeros(label.size(0), self.num_classes).to(label.device)
        one_hot.scatter_(1, label.unsqueeze(1), 1)
        return one_hot
    
    def encode(self, x: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Convert label to one-hot and expand to match image dimensions
        one_hot = self._one_hot_encode(label)
        one_hot = one_hot.view(-1, self.num_classes, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        
        # Concatenate image with label channels
        x = torch.cat([x, one_hot], dim=1)
        
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, label: torch.Tensor, apply_threshold: bool = False) -> torch.Tensor:
        # Concatenate latent vector with label
        one_hot = self._one_hot_encode(label)
        z = torch.cat([z, one_hot], dim=1)
        
        x = self.decoder_input(z)
        
        # Calculate the initial size based on the number of conv layers
        initial_size = self.image_size // (2 ** self.num_conv_layers)
        
        # Reshape with the correct dimensions
        x = x.view(-1, 512, initial_size, initial_size)
        output = self.decoder(x)
        
        # Ensure output size matches input size
        if output.shape[-1] != self.image_size:
            output = F.interpolate(output, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        if apply_threshold:
            output = (output > self.threshold).float()
            
        return output
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Ensure input is the correct size
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        mu, log_var = self.encode(x, label)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z, label)
        
        return x_hat, mu, log_var
    
    def generate(self, label: torch.Tensor, num_samples: int = 1, apply_threshold: bool = True) -> torch.Tensor:
        """Generate samples for a given label"""
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.fc_mu.out_features, device=device)
        output = self.decode(z, label)
        
        if apply_threshold:
            output = (output > self.threshold).float()
        
        return output
    
    def slerp_classes(self, 
                      start_label: int, 
                      end_label: int, 
                      num_steps: int = 10,
                      z: torch.Tensor = None) -> torch.Tensor:
        """
        Generate images interpolating between two classes using spherical linear interpolation (SLERP).
        """
        device = next(self.parameters()).device
        start_label = torch.tensor([start_label], device=device)
        end_label = torch.tensor([end_label], device=device)
        start_one_hot = self._one_hot_encode(start_label)
        end_one_hot = self._one_hot_encode(end_label)
        
        # Normalize the one-hot vectors
        start_norm = start_one_hot / start_one_hot.norm()
        end_norm = end_one_hot / end_one_hot.norm()
        
        # Calculate dot product and angle between vectors
        dot_product = (start_norm * end_norm).sum()
        omega = torch.acos(torch.clamp(dot_product, -1, 1))
        
        alphas = torch.linspace(0, 1, num_steps, device=device)

        if z is None:
            z = torch.randn(1, self.fc_mu.out_features, device=device)
            
        # Generate images for each interpolation step
        images = []
        with torch.no_grad():
            for alpha in alphas:
                # SLERP formula
                if omega.abs() > 1e-6:  # Avoid division by zero
                    interpolated_label = (
                        torch.sin((1 - alpha) * omega) / torch.sin(omega) * start_one_hot +
                        torch.sin(alpha * omega) / torch.sin(omega) * end_one_hot
                    )
                else:
                    # If vectors are very close, fall back to linear interpolation
                    interpolated_label = (1 - alpha) * start_one_hot + alpha * end_one_hot
                
                # Generate image
                x = torch.cat([z, interpolated_label], dim=1)
                x = self.decoder_input(x)
                x = x.view(-1, 256, 8, 8)
                image = self.decoder(x)
                images.append(image)
        
        return torch.cat(images, dim=0)
    
    def interpolate_classes(self, 
                           start_label: int, 
                           end_label: int, 
                           num_steps: int = 10,
                           z: torch.Tensor = None) -> torch.Tensor:
        """
        Generate images interpolating between two classes.
        If z is not provided, uses random latent vector.
        """
        device = next(self.parameters()).device
        
        # Use provided latent vector or generate random one
        if z is None:
            z = torch.randn(1, self.fc_mu.out_features, device=device)
        
        # Create interpolation steps for labels
        alphas = torch.linspace(0, 1, num_steps, device=device)
        
        # Convert start and end labels to one-hot
        start_label = torch.tensor([start_label], device=device)
        end_label = torch.tensor([end_label], device=device)
        start_one_hot = self._one_hot_encode(start_label)
        end_one_hot = self._one_hot_encode(end_label)
        
        # Generate images for each interpolation step
        images = []
        with torch.no_grad():
            for alpha in alphas:
                # Interpolate between one-hot vectors
                interpolated_label = (1 - alpha) * start_one_hot + alpha * end_one_hot
                
                # Generate image
                x = torch.cat([z, interpolated_label], dim=1)
                x = self.decoder_input(x)
                x = x.view(-1, 256, 8, 8)
                image = self.decoder(x)
                images.append(image)
        
        return torch.cat(images, dim=0) 
    
    def edge_loss(self, x: torch.Tensor, recon_x: torch.Tensor) -> torch.Tensor:
        # Compute gradients in both x and y directions
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device).float().view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device).float().view(1, 1, 3, 3)
        
        # Pad inputs
        pad = nn.ReflectionPad2d(1)
        x_padded = pad(x)
        recon_x_padded = pad(recon_x)
        
        # Calculate gradients
        x_gradx = F.conv2d(x_padded, sobel_x)
        x_grady = F.conv2d(x_padded, sobel_y)
        recon_gradx = F.conv2d(recon_x_padded, sobel_x)
        recon_grady = F.conv2d(recon_x_padded, sobel_y)
        
        # Compute L1 loss between gradients
        edge_loss = F.l1_loss(x_gradx, recon_gradx) + F.l1_loss(x_grady, recon_grady)
        return edge_loss 