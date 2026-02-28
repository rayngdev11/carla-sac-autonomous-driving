import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from vae import VariationalAutoencoder
import numpy as np

# Clear GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def test_vae():
    # Load model
    model = VariationalAutoencoder(latent_dims=64).to(device)
    
    # Load trained weights
    if os.path.exists('model/var_autoencoder.pth'):
        model.load('model/var_autoencoder.pth')
        print("Loaded VAE model")
    else:
        print("No trained model found!")
        return
    
    # Data transform
    transform = transforms.Compose([
        transforms.Resize((160, 79)),
        transforms.ToTensor(),
    ])
    
    # Test on some images
    test_dir = 'dataset/test/class1'
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found!")
        return
    
    # Get first 10 test images
    test_images = []
    for i, filename in enumerate(os.listdir(test_dir)):
        if filename.endswith('.png') and i < 10:
            img_path = os.path.join(test_dir, filename)
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            test_images.append((image_tensor, filename))
    
    print(f"Testing on {len(test_images)} images")
    
    # Create reconstructed directory
    os.makedirs('reconstructed', exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for i, (image_tensor, filename) in enumerate(test_images):
            # Encode and decode
            recon_image, latent = model(image_tensor)
            
            # Convert to PIL images
            original = transforms.ToPILImage()(image_tensor.squeeze(0).cpu())
            reconstructed = transforms.ToPILImage()(recon_image.squeeze(0).cpu())
            
            # Save reconstructed image
            recon_filename = f"{i+1}.png"
            reconstructed.save(os.path.join('reconstructed', recon_filename))
            
            # Display comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            ax1.imshow(original)
            ax1.set_title(f'Original: {filename}')
            ax1.axis('off')
            
            ax2.imshow(reconstructed)
            ax2.set_title(f'Reconstructed: {recon_filename}')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'reconstructed/comparison_{i+1}.png')
            plt.close()
            
            print(f"Processed {filename} -> {recon_filename}")
    
    print("Testing completed! Check the 'reconstructed' folder for results.")

def analyze_latent_space():
    """Analyze the latent space of the VAE"""
    # Load model
    model = VariationalAutoencoder(latent_dims=64).to(device)
    
    if os.path.exists('model/var_autoencoder.pth'):
        model.load('model/var_autoencoder.pth')
        print("Loaded VAE model for latent space analysis")
    else:
        print("No trained model found!")
        return
    
    # Load some test images
    test_dir = 'dataset/test/class1'
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found!")
        return
    
    transform = transforms.Compose([
        transforms.Resize((160, 79)),
        transforms.ToTensor(),
    ])
    
    # Get latent vectors for 100 images
    latent_vectors = []
    model.eval()
    
    with torch.no_grad():
        for i, filename in enumerate(os.listdir(test_dir)):
            if i >= 100:
                break
            if filename.endswith('.png'):
                img_path = os.path.join(test_dir, filename)
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Get latent vector
                latent = model.encode(image_tensor)
                latent_vectors.append(latent.cpu().numpy())
    
    latent_vectors = np.array(latent_vectors).squeeze()
    
    # Plot latent space statistics
    plt.figure(figsize=(15, 5))
    
    # Mean of each latent dimension
    plt.subplot(1, 3, 1)
    plt.bar(range(latent_vectors.shape[1]), np.mean(latent_vectors, axis=0))
    plt.title('Mean of Latent Dimensions')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Mean Value')
    
    # Standard deviation of each latent dimension
    plt.subplot(1, 3, 2)
    plt.bar(range(latent_vectors.shape[1]), np.std(latent_vectors, axis=0))
    plt.title('Std of Latent Dimensions')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Standard Deviation')
    
    # Histogram of all latent values
    plt.subplot(1, 3, 3)
    plt.hist(latent_vectors.flatten(), bins=50, alpha=0.7)
    plt.title('Distribution of All Latent Values')
    plt.xlabel('Latent Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('latent_space_analysis.png')
    plt.show()
    
    print("Latent space analysis completed!")

if __name__ == "__main__":
    print("Testing VAE model...")
    test_vae()
    
    print("\nAnalyzing latent space...")
    analyze_latent_space() 