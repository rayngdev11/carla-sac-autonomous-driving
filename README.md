<div align="center">

# CARLA Autonomous Driving Agent with Soft Actor-Critic (SAC)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CARLA Simulator](https://img.shields.io/badge/CARLA-Simulator-red)](https://carla.org/)
[![Stable Baselines3](https://img.shields.io/badge/Stable%20Baselines3-RL-brightgreen)](https://stable-baselines3.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Active-success)]()

**An advanced Reinforcement Learning project that trains an autonomous vehicle to navigate the CARLA simulator using computer vision and Soft Actor-Critic (SAC).**

</div>

---

## To Hiring Managers & HR Professionals

Welcome! This repository demonstrates my capability to design, train, and deploy **Deep Reinforcement Learning (RL)** models in complex, state-of-the-art simulation environments. 

**What you should know about this project:**
- **The Problem:** Teaching a car to drive itself safely in a realistic 3D environment without explicit programming rules for every situation.
- **The Solution:** A custom AI agent that learns through trial and error (Reinforcement Learning). It "sees" through a camera, compresses the image using an Autoencoder (VAE), and makes driving decisions using the Soft Actor-Critic (SAC) algorithm.
- **Skills Demonstrated:** 
  - Advanced Machine Learning / RL (`PyTorch`, `stable-baselines3`).
  - Computer Vision (`Variational Autoencoders`, `CNNs`).
  - Simulation & System Integration (`CARLA Simulator`).
  - MLOps & Optimization (`Optuna` for hyperparameter tuning, `TensorBoard` for tracking).

---

## Key Features & Highlights

- **End-to-End Deep RL Pipeline:** Built on top of the robust Soft Actor-Critic (SAC) algorithm for stable and sample-efficient learning.
- **Vision-Based Navigation:** Uses a pre-trained **Variational Autoencoder (VAE)** to compress raw camera images (RGB) into a dense, low-dimensional latent space representations, significantly accelerating the RL training process.
- **Curriculum Learning:** The training pipeline implements a two-phase curriculum learning strategy:
  - *Phase 1:* Fixed spawn points and routes to master basic driving mechanics.
  - *Phase 2:* Finetuning with random spawn points to generalize driving skills across the map (Town07).
- **Performance Optimized:** Incorporates Automatic Mixed Precision (AMP), gradient clipping, and custom Residual Blocks to accelerate GPU training times.
- **Hyperparameter Tuning:** Integrated with **Optuna** to dynamically search for the optimal learning rate, batch size, and network architecture.

---

## Technical Architecture

### Tech Stack
*   **Language:** Python
*   **Deep Learning Framework:** PyTorch
*   **RL Library:** Stable-Baselines3
*   **Environment:** CARLA Simulator (0.9.x)
*   **Tracking & Tuning:** TensorBoard, Optuna

### The Brain of the Agent: `CombinedExtractor`
The model uses a custom neural network architecture to process the environment:
1. **Vision:** A VAE Encoder compresses high-dimensional CARLA camera feeds into a 95-dimensional latent vector.
2. **Telemetry:** Vehicle state (speed, steering, etc.) is processed by an MLP.
3. **Fusion:** Semantic vision and hard telemetry are combined and fed into the SAC policy network to output optimal steering, throttle, and braking actions.

---

## Project Structure

```text
Carla-SAC-Autonomous-Driving
 ┣ auto_encoder/    # Architecture and weights for the Variational Autoencoder (VAE)
 ┣ env/             # Custom Gym wrappers for the CARLA simulator 
 ┃ ┣ curriculum.py  # Logic for Curriculum Learning progression
 ┃ ┣ sensors.py     # Camera, Collision, and Lane Invasion sensor setups
 ┣ models/          # Custom neural network layers (Feature Extractors)
 ┣ checkpoints/     # Saved SAC models and replay buffers during training
 ┣ logs/            # TensorBoard metrics and CSV logs
 ┣ carla_env.py     # Main CARLA Gym Environment integration
 ┗ train_sac_carla.py # Main training orchestration script
```

---

## Getting Started

### Prerequisites
*   Windows/Linux with a dedicated NVIDIA GPU (RTX recommended).
*   **CARLA Simulator** installed locally.
*   Python 3.8+ and Anaconda/Miniconda environment.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rayngdev11/carla-sac-autonomous-driving.git
   cd carla-sac-autonomous-driving
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you install the correct PyTorch version compatible with your CUDA toolkit).*

### Running the Training

You can initiate training directly from the command line. The script will automatically launch the CARLA server if it isn't already running.

```bash
python train_sac_carla.py \
  --carla_path "C:/Path/To/CarlaUnreal/CarlaUE4.exe" \
  --total_timesteps 150000 \
  --town Town07 \
  --curriculum \
  --visualize
```

**Useful Arguments:**
- `--optimize`: Run Optuna hyperparameter tuning before training.
- `--resume [path]`: Resume training from a specific `.zip` checkpoint.
- `--finetune_mode`: Spawn the vehicle at random locations to test generalization.

---

<div align="center">
  <i>Developed for the future of Autonomous Driving.</i>
</div>
