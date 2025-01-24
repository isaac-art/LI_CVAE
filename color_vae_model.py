import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from pathlib import Path

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

class ColorVAE(nn.Module):
    def __init__(self, latent_dim: int = 32, image_size: int = 128, temperature: float = 0.1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.to(self.device)
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Calculate number of convolution layers needed
        self.num_conv_layers = int(np.log2(image_size)) - 2
        initial_size = image_size // (2 ** self.num_conv_layers)
        
        # Encoder
        encoder_layers = []
        current_channels = 3  # RGB input
        for i in range(self.num_conv_layers):
            out_channels = min(512, 64 * (2 ** i))
            encoder_layers.extend([
                nn.Conv2d(current_channels, out_channels, 4, stride=2, padding=1),
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
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        
        # Decoder layers
        decoder_layers = []
        for i in range(self.num_conv_layers):
            in_channels = current_channels
            out_channels = min(512, 64 * (2 ** (self.num_conv_layers - i - 2)))
            if i == self.num_conv_layers - 1:  # Last layer
                out_channels = 3  # RGB output
                decoder_layers.extend([
                    nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.Sigmoid()  # Regular sigmoid for color values
                ])
            else:
                decoder_layers.extend([
                    nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True)
                ])
            current_channels = out_channels
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        print(f"Initialized ColorVAE with image size {image_size}x{image_size}")
        print(f"Number of conv layers: {self.num_conv_layers}")
        print(f"Flattened dimension: {self.flatten_dim}")
        print(f"Latent dimension: {latent_dim}")
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_input(z)
        
        # Calculate the initial size based on the number of conv layers
        initial_size = self.image_size // (2 ** self.num_conv_layers)
        
        # Reshape with the correct dimensions
        x = x.view(-1, 512, initial_size, initial_size)
        output = self.decoder(x)
        
        # Ensure output size matches input size
        if output.shape[-1] != self.image_size:
            output = F.interpolate(output, size=(self.image_size, self.image_size), 
                                 mode='bilinear', align_corners=False)
        
        return output
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Ensure input is the correct size
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), 
                            mode='bilinear', align_corners=False)
        
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        
        return x_hat, mu, log_var
    
    def generate(self, num_samples: int = 1) -> torch.Tensor:
        """Generate random samples"""
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor, 
                   num_steps: int = 10) -> torch.Tensor:
        """Generate images interpolating between two latent vectors"""
        device = next(self.parameters()).device
        alphas = torch.linspace(0, 1, num_steps, device=device)
        
        images = []
        with torch.no_grad():
            for alpha in alphas:
                z = (1 - alpha) * z1 + alpha * z2
                image = self.decode(z)
                images.append(image)
        
        return torch.cat(images, dim=0) 