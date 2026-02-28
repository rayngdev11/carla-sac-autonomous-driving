import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from vae import VariationalAutoencoder
import numpy as np

# Clear GPU memory before training
if torch.cuda.is_available():
    torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CarlaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []
        
        # Collect all image files
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.png'):
                    self.image_files.append(os.path.join(root, file))
        
        print(f"Found {len(self.image_files)} images in {data_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

def train_vae():
    # Hyperparameters
    latent_dims = 64
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-4
    beta = 1.0  # KL divergence weight
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((160, 79)),  # Match CARLA camera resolution
        transforms.ToTensor(),
    ])
    
    # Load datasets
    train_dataset = CarlaDataset('dataset/train', transform=transform)
    test_dataset = CarlaDataset('dataset/test', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = VariationalAutoencoder(latent_dims).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_losses = []
    test_losses = []
    
    print("Starting VAE training...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, z = model(data)
            
            # Reconstruction loss
            recon_loss = criterion(recon_batch, data)
            
            # KL divergence loss
            kl_loss = model.encoder.kl
            
            # Total loss
            loss = recon_loss + beta * kl_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.6f}, Recon: {recon_loss.item():.6f}, KL: {kl_loss.item():.6f}')
        
        # Validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                recon_batch, z = model(data)
                loss = criterion(recon_batch, data) + beta * model.encoder.kl
                test_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}')
        
        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.save(f'model/var_autoencoder_epoch_{epoch+1}.pth')
            model.encoder.save()
    
    # Save final model
    model.save('model/var_autoencoder.pth')
    model.encoder.save()
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training History')
    plt.legend()
    plt.savefig('training_history.png')
    plt.show()
    
    print("Training completed!")

if __name__ == "__main__":
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    train_vae() 