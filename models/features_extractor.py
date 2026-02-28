import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Type, Union
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class VAEEncoder(nn.Module):
    """
    VAE Encoder để trích xuất features từ ảnh camera
    """
    def __init__(self, input_channels: int = 3, latent_dim: int = 64):
        super(VAEEncoder, self).__init__()
        
        # Encoder layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Latent space layers
        self.fc_mu = nn.Linear(256 * 5 * 5, latent_dim)
        self.fc_var = nn.Linear(256 * 5 * 5, latent_dim)
        
        self.latent_dim = latent_dim
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Normalize input to [0, 1]
        x = x.float() / 255.0
        
        # Convert from (batch, height, width, channels) to (batch, channels, height, width)
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        
        # Encoder forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = self.flatten(x)
        
        # Latent space
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        return mu, log_var
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation"""
        mu, log_var = self.forward(x)
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

class CNNExtractor(nn.Module):
    """
    CNN Encoder đơn giản để trích xuất features từ ảnh
    """
    def __init__(self, input_channels: int = 3, output_dim: int = 64):
        super(CNNExtractor, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Final feature layer
        self.fc = nn.Linear(256 * 5 * 5, output_dim)
        
        self.output_dim = output_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input to [0, 1]
        x = x.float() / 255.0
        
        # Convert from (batch, height, width, channels) to (batch, channels, height, width)
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        
        # CNN forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        
        return x

class CombinedExtractor(BaseFeaturesExtractor):
    """
    Feature extractor kết hợp VAE cho ảnh và MLP cho state vector
    """
    def __init__(
        self, 
        observation_space: spaces.Dict,
        vae_latent_dim: int = 64,
        state_features_dim: int = 32,
        features_dim: int = 128,
        vae_path: str = None
    ):
        super().__init__(observation_space, features_dim)
        
        # VAE Encoder cho ảnh
        image_space = observation_space.spaces['image']
        self.vae_encoder = VAEEncoder(
            input_channels=image_space.shape[2],
            latent_dim=vae_latent_dim
        )
        
        # Load pretrained VAE nếu có
        if vae_path and os.path.exists(vae_path):
            try:
                self.vae_encoder.load_state_dict(torch.load(vae_path, map_location='cpu'))
                self.vae_encoder.eval()
                print(f"✓ Loaded pretrained VAE from {vae_path}")
            except Exception as e:
                print(f"⚠️  Failed to load VAE: {e}")
        
        # MLP cho state vector
        state_space = observation_space.spaces['state']
        state_dim = state_space.shape[0]
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_features_dim),
            nn.ReLU()
        )
        
        # Final combination layer
        combined_dim = vae_latent_dim + state_features_dim
        self.combined_mlp = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process image through VAE encoder
        image_features = self.vae_encoder.encode(observations['image'])
        
        # Process state vector through MLP
        state_features = self.state_mlp(observations['state'])
        
        # Combine features
        combined_features = torch.cat([image_features, state_features], dim=1)
        
        # Final processing
        features = self.combined_mlp(combined_features)
        
        return features

class CNNCombinedExtractor(BaseFeaturesExtractor):
    """
    Feature extractor kết hợp CNN cho ảnh và MLP cho state vector
    """
    def __init__(
        self, 
        observation_space: spaces.Dict,
        cnn_output_dim: int = 64,
        state_features_dim: int = 32,
        features_dim: int = 128
    ):
        super().__init__(observation_space, features_dim)
        
        # CNN Encoder cho ảnh
        image_space = observation_space.spaces['image']
        self.cnn_encoder = CNNExtractor(
            input_channels=image_space.shape[2],
            output_dim=cnn_output_dim
        )
        
        # MLP cho state vector
        state_space = observation_space.spaces['state']
        state_dim = state_space.shape[0]
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_features_dim),
            nn.ReLU()
        )
        
        # Final combination layer
        combined_dim = cnn_output_dim + state_features_dim
        self.combined_mlp = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process image through CNN encoder
        image_features = self.cnn_encoder(observations['image'])
        
        # Process state vector through MLP
        state_features = self.state_mlp(observations['state'])
        
        # Combine features
        combined_features = torch.cat([image_features, state_features], dim=1)
        
        # Final processing
        features = self.combined_mlp(combined_features)
        
        return features

def load_pretrained_vae(vae_path: str, device: str = "cpu") -> VAEEncoder:
    """
    Load pretrained VAE encoder
    """
    vae_encoder = VAEEncoder()
    vae_encoder.load_state_dict(torch.load(vae_path, map_location=device))
    vae_encoder.eval()
    return vae_encoder 

class VAEStateExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        latent_dim = observation_space.spaces["latent"].shape[0]
        state_dim = observation_space.spaces["state"].shape[0]
        self.net = nn.Sequential(
            nn.Linear(latent_dim + state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
        # Đảm bảo model chạy trên GPU
        if torch.cuda.is_available():
            self.net = self.net.cuda()

    def forward(self, observations):
        latent = observations["latent"]
        state = observations["state"]
        # Đảm bảo input cùng dtype với self.net
        dtype = next(self.net.parameters()).dtype
        device = next(self.net.parameters()).device
        latent = latent.to(device=device, dtype=dtype)
        state = state.to(device=device, dtype=dtype)
        x = torch.cat([latent, state], dim=1)
        return self.net(x)