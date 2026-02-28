import os
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
import gym

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carla_env import CarlaEnv
from env.curriculum import CurriculumEnv
from .features_extractor import CombinedExtractor, CNNCombinedExtractor, load_pretrained_vae

class SACAgent:
    """
    SAC Agent wrapper với curriculum learning và custom feature extractor
    """
    def __init__(
        self,
        feature_extractor: str = "vae",
        vae_path: str = None,
        device: str = "auto",
        tensorboard_log: str = "./checkpoints/sac_tensorboard/",
        **kwargs
    ):
        self.feature_extractor = feature_extractor
        self.vae_path = vae_path if vae_path is not None else ""
        self.device = device
        self.tensorboard_log = tensorboard_log
        self.kwargs = kwargs
        
        # Create environment
        self.env = self._create_env()
        
        # Initialize SAC with custom feature extractor
        self.model = self._create_model()
        
    def _create_env(self):
        """Tạo environment với SB3 compatibility wrapper"""
        def make_env():
            env = CarlaEnv(
                num_npcs=5,
                frame_skip=8,
                visualize=False,  # Set to False for training
                fixed_delta_seconds=0.05,
                camera_width=160,
                camera_height=80,
                safe_dist=5.0,
                obstacle_weight=2.0,
                lane_weight=2.0,
                yaw_weight=1.0
            )
            env = Monitor(env)
            return env
        
        # Create environment with proper gym compatibility
        env = make_env()
        return env
    
    def _create_model(self):
        """Tạo SAC model với custom feature extractor"""
        # Chọn feature extractor dựa trên tham số
        if self.feature_extractor == "vae":
            extractor_class = CombinedExtractor
            extractor_kwargs = {
                "vae_latent_dim": 64,
                "state_features_dim": 32,
                "features_dim": 128,
                "vae_path": self.vae_path if self.vae_path else None
            }
            
            # Load pretrained VAE nếu có path
            if self.vae_path and os.path.exists(self.vae_path):
                print(f"✓ Loading pretrained VAE from: {self.vae_path}")
            else:
                print("⚠️  No pretrained VAE found, using default VAE encoder")
        else:  # cnn
            extractor_class = CNNCombinedExtractor
            extractor_kwargs = {
                "cnn_output_dim": 64,
                "state_features_dim": 32,
                "features_dim": 128
            }
        
        # Custom feature extractor
        policy_kwargs = {
            "features_extractor_class": extractor_class,
            "features_extractor_kwargs": extractor_kwargs
        }
        
        # SAC parameters
        sac_params = {
            "policy": "MultiInputPolicy",
            "env": self.env,
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "learning_starts": 1000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "target_entropy": -3,  # -dim(A)
            "policy_kwargs": policy_kwargs,
            "tensorboard_log": self.tensorboard_log,
            "verbose": 1,
            "device": self.device,
            **self.kwargs
        }
        
        return SAC(**sac_params)
    
    def train(
        self,
        total_timesteps: int = 100000,
        checkpoint_freq: int = 10000,
        eval_freq: int = 5000,
        save_path: str = "./checkpoints/"
    ):
        """Train SAC agent"""
        os.makedirs(save_path, exist_ok=True)
        
        # Callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=save_path,
            name_prefix="sac_carla"
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback (optional)
        if eval_freq > 0:
            eval_env = self._create_env()  # Create separate eval env
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=eval_freq,
                n_eval_episodes=5,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        final_path = os.path.join(save_path, "sac_carla_final")
        self.model.save(final_path)
        print(f"Training completed. Model saved to {final_path}")
    
    def load(self, path: str):
        """Load trained model"""
        self.model = SAC.load(path, env=self.env)
        print(f"Model loaded from {path}")
    
    def evaluate(self, n_episodes: int = 10, render: bool = True):
        """Evaluate the trained agent"""
        # Create evaluation environment with visualization
        def make_eval_env():
            env = CarlaEnv(
                num_npcs=5,
                frame_skip=8,
                visualize=render,
                fixed_delta_seconds=0.05,
                camera_width=160,
                camera_height=80,
                safe_dist=5.0,
                obstacle_weight=2.0,
                lane_weight=2.0,
                yaw_weight=1.0
            )
            return env
        
        eval_env = make_eval_env()
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs = eval_env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                total_reward += reward
                steps += 1
                
                if render:
                    eval_env.render()
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
        
        eval_env.close()
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        print(f"\nEvaluation Results:")
        print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Mean Episode Length: {mean_length:.1f}")
        
        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_length": mean_length,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths
        }

def main():
    """Main training script"""
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Initialize SAC agent
    agent = SACAgent(
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log="./checkpoints/sac_tensorboard/"
    )
    
    # Train the agent
    agent.train(
        total_timesteps=100000,
        checkpoint_freq=10000,
        eval_freq=5000,
        save_path="./checkpoints/"
    )
    
    # Evaluate the trained agent
    agent.evaluate(n_episodes=5, render=True)

if __name__ == "__main__":
    main() 