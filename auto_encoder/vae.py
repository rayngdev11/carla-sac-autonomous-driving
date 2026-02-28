import os
import torch
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):  
        super(VariationalEncoder, self).__init__()

        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),  # 79, 39
            nn.LeakyReLU())

        self.encoder_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 40, 20
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        self.encoder_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2),  # 19, 9
            nn.LeakyReLU())

        self.encoder_layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2),  # 9, 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU())

        self.linear = nn.Sequential(
            nn.Linear(9*4*256, 1024),
            nn.LeakyReLU())

        self.mu = nn.Linear(1024, latent_dims)
        self.sigma = nn.Linear(1024, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x):
        x = x.to(device)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        mu =  self.mu(x)
        sigma = torch.exp(self.sigma(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class VariationalDecoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalDecoder, self).__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(latent_dims, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 9*4*256),
            nn.LeakyReLU()
        )
        
        self.decoder_layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2),  # 4, 9
            nn.BatchNorm2d(128),
            nn.LeakyReLU())
            
        self.decoder_layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2),  # 9, 19
            nn.LeakyReLU())
            
        self.decoder_layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),  # 19, 40
            nn.BatchNorm2d(32),
            nn.LeakyReLU())
            
        self.decoder_layer4 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, stride=2),  # 40, 79
            nn.Sigmoid())
    
    def forward(self, z):
        z = z.to(device)
        x = self.linear(z)
        x = x.view(x.size(0), 256, 4, 9)
        x = self.decoder_layer1(x)
        x = self.decoder_layer2(x)
        x = self.decoder_layer3(x)
        x = self.decoder_layer4(x)
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = VariationalDecoder(latent_dims)
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=device)) 