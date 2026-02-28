# VAE Training cho CARLA

Thư mục này chứa các file để train và sử dụng Variational Autoencoder (VAE) cho việc nén ảnh camera trong môi trường CARLA.

## Cấu trúc thư mục

```
auto_encoder/
├── vae.py                 # Định nghĩa model VAE
├── encoder.py             # Encoder riêng biệt cho inference
├── train_vae.py           # Script train VAE
├── test_vae.py            # Script test và visualize VAE
├── dataset/               # Dataset ảnh CARLA
│   ├── train/             # Training images
│   └── test/              # Test images
├── model/                 # Saved models
│   ├── var_autoencoder.pth
│   └── var_encoder_model.pth
├── reconstructed/         # Ảnh reconstructed
└── runs/                  # TensorBoard logs
```

## Cài đặt

```bash
# Cài đặt dependencies
pip install torch torchvision matplotlib pillow numpy
```

## Training VAE

### 1. Chuẩn bị dataset

Đảm bảo có dataset ảnh CARLA trong thư mục `dataset/`:
- `dataset/train/class1/` - Ảnh training
- `dataset/test/class1/` - Ảnh test

### 2. Train VAE

```bash
cd auto_encoder
python train_vae.py
```

**Hyperparameters mặc định:**
- Latent dimensions: 64
- Batch size: 32
- Learning rate: 1e-4
- Epochs: 100
- Image size: 160x79 (match CARLA camera)

### 3. Test VAE

```bash
python test_vae.py
```

Script này sẽ:
- Load model đã train
- Test trên ảnh từ dataset
- Tạo ảnh reconstructed
- Phân tích latent space

## Model Architecture

### VariationalAutoencoder
- **Encoder**: 4 conv layers + linear layers
- **Decoder**: 4 deconv layers + linear layers
- **Latent space**: 64 dimensions
- **Loss**: MSE + KL divergence

### VariationalEncoder (cho inference)
- Chỉ encoder part
- Tối ưu cho inference nhanh
- Hỗ trợ GPU với FP16

## Sử dụng trong RL

VAE được sử dụng trong `carla_env.py` để:
1. Encode ảnh camera thành latent vector (95 dimensions)
2. Kết hợp với state vector (5 dimensions)
3. Trả về observation cho SAC agent

## Tối ưu hóa Performance

### GPU Optimization
```python
# Sử dụng FP16 cho VAE
vae = vae.half()

# Cache latent vectors
self.latent_cache = {}

# Batch processing
with torch.no_grad():
    latent = vae(image_batch)
```

### Memory Management
```python
# Clear GPU memory
torch.cuda.empty_cache()

# Disable gradients cho inference
vae.eval()
for param in vae.parameters():
    param.requires_grad = False
```

## Troubleshooting

### Lỗi GPU Memory
```bash
# Giảm batch size
python train_vae.py --batch_size 16

# Sử dụng gradient checkpointing
# (đã implement trong code)
```

### Lỗi Dataset
```bash
# Kiểm tra dataset
ls dataset/train/class1/ | wc -l
ls dataset/test/class1/ | wc -l
```

### Lỗi Model Loading
```bash
# Kiểm tra model files
ls -la model/

# Retrain nếu cần
python train_vae.py
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir runs/
```

### Manual Check
```bash
# Kiểm tra loss
tail -f training_history.png

# Kiểm tra reconstructed images
ls reconstructed/
```

## Tips

1. **Dataset Quality**: Sử dụng ảnh đa dạng từ nhiều scenario
2. **Latent Dimensions**: 64-128 cho balance giữa compression và quality
3. **Training Time**: ~30-60 phút trên GPU
4. **Validation**: Kiểm tra reconstructed images thường xuyên
5. **GPU Memory**: Monitor GPU usage với `nvidia-smi`

## Integration với RL

VAE được tích hợp seamless với SAC training:

```python
# Trong carla_env.py
observation = {
    'latent': latent_vector,  # 95 dims từ VAE
    'state': state_vector     # 5 dims từ sensors
}
```

Điều này cho phép agent học từ compressed representation thay vì raw pixels, tăng tốc training và giảm memory usage. 