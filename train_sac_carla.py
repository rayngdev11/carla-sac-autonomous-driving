import os
import sys
import argparse
import gym
from gym.core import ObservationWrapper
import torch
import torch.nn as nn
import numpy as np
import subprocess
from typing import Union
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from rich.console import Console
import optuna
from stable_baselines3.common.logger import configure  
import serial
import time
import psutil

from carla_env import CustomSAC
from carla_env import CarlaEnv
from env.curriculum import CurriculumEnv
from auto_encoder.encoder import VariationalEncoder
from env.sensors import CameraSensor, CollisionSensor, LaneInvasionSensor
from models.features_extractor import VAEStateExtractor

# Add current directory to path for VAE import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


console = Console()

# Giải phóng GPU memory trước khi train
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# THÊM: Kiểm tra và khởi động CARLA server
def check_carla_connection():
    """Kiểm tra xem CARLA server đã chạy chưa"""
    try:
        import carla
        client = carla.Client('localhost', 2000)
        client.set_timeout(5)
        client.get_world()
        return True
    except:
        return False

def start_carla_server():
    # Kiểm tra xem CARLA đã chạy chưa
    if check_carla_connection():
        console.log(f"[green]CARLA server is already running![/green]")
        return True
        
    try:
        # Chạy trực tiếp CARLA với đường dẫn cụ thể
        import subprocess
        import time
        
        carla_path = "C:/Corsair profile/CarlaUnreal/CarlaUE4.exe"
        console.log(f"[yellow]Starting CARLA directly from: {carla_path}[/yellow]")
        
        # Chạy CARLA với các tham số tối ưu cho training
        process = subprocess.Popen([
            carla_path,
            "-carla-map=Town07",
            "-windowed",
            "-ResX=1280",
            "-ResY=720", 
            "-quality-level=low",
            "-benchmark",
            "-fps=60",
            "-carla-server",
            "-carla-port=2000",
            "-map=Town07",  # Thêm tham số map thứ 2 để đảm bảo
            "-force-map-reload"  # Force reload map
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        console.log(f"[green]CARLA process started with PID: {process.pid}[/green]")
        console.log(f"[yellow]Waiting for CARLA to initialize...[/yellow]")
        
        # Đợi một chút để CARLA khởi động
        time.sleep(20)  # Tăng thời gian chờ vì có thêm các tham số
        
        # Kiểm tra xem process còn chạy không
        if process.poll() is None:
            console.log(f"[green]CARLA is running successfully![/green]")
            
            # Thêm retry logic để đợi CARLA sẵn sàng
            console.log(f"[yellow]Waiting for CARLA to be ready...[/yellow]")
            for i in range(45):  # Tăng thời gian chờ lên 45 giây
                if check_carla_connection():
                    console.log(f"[green]CARLA is ready for connection![/green]")
                    return True
                time.sleep(1)
                if i % 5 == 0:
                    console.log(f"[yellow]Still waiting... ({i+1}/45)[/yellow]")
            
            console.log(f"[red]CARLA started but not responding to connections[/red]")
            return False
        else:
            stdout, stderr = process.communicate()
            console.log(f"[red]CARLA failed to start:[/red]")
            console.log(f"[red]STDOUT: {stdout.decode()}[/red]")
            console.log(f"[red]STDERR: {stderr.decode()}[/red]")
            return False
            
    except FileNotFoundError:
        console.log(f"[red]CARLA executable not found at: {carla_path}[/red]")
        console.log(f"[yellow]Please check if the path is correct.[/yellow]")
        return False
    except Exception as e:
        console.log(f"[red]Error starting CARLA: {e}[/red]")
        return False

# Khởi động CARLA trước khi tạo environment
if not start_carla_server():
    console.log(f"[red]Failed to start CARLA server. Please start manually.[/red]")
    console.log(f"[yellow]Manual command: CarlaUE4.exe -carla-map=Town07 -map=Town07 -windowed -ResX=1280 -ResY=720 -quality-level=low -benchmark -fps=60 -force-map-reload -carla-server -carla-port=2000[/yellow]")
    exit(1)

# NEW: Configure a logger that outputs to stdout, CSV, and tensorboard.

new_logger = configure("./sac_tensorboard/", ["stdout", "csv", "tensorboard"])



from stable_baselines3.common.callbacks import BaseCallback



class EntropyLoggingCallback(BaseCallback):

    def __init__(self, log_interval: int = 1000, verbose: int = 1):

        super(EntropyLoggingCallback, self).__init__(verbose)

        self.log_interval = log_interval



    def _on_step(self) -> bool:

        if self.n_calls % self.log_interval == 0:

            # Directly access the entropy coefficient from the model.

            if hasattr(self.model, 'log_ent_coef') and getattr(self.model, 'log_ent_coef', None) is not None:

                current_ent_coef = torch.exp(getattr(self.model, 'log_ent_coef')).item()

                print(f"[INFO] Step {self.n_calls}: Entropy Coefficient = {current_ent_coef}")

            else:

                print(f"[INFO] Step {self.n_calls}: Entropy Coefficient not available")

        return True





# --- Define a Residual Block for the CNN ---

class ResidualBlock(nn.Module):

    def __init__(self, channels: int):

        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        

    def forward(self, x):

        residual = x

        out = self.relu(self.conv1(x))

        out = self.conv2(out)

        return self.relu(out + residual)



# --- Enhanced Feature Extractor with Attention Fusion ---

class CombinedExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim: int = 100, vae_path=None):

        super().__init__(observation_space, features_dim)
        
        latent_dim = 95
        self.vae_encoder = VariationalEncoder(latent_dim)
        if vae_path is not None:
            self.vae_encoder.load()
        
        # Luôn sử dụng GPU cho tất cả model
        if torch.cuda.is_available():
            self.vae_encoder = self.vae_encoder.cuda()
            self.vae_encoder = self.vae_encoder.half()  # Sử dụng FP16
        else:
            raise RuntimeError("GPU is required but not available!")

        
        self.vae_encoder.eval()
        for param in self.vae_encoder.parameters():
            param.requires_grad = False

        # MLP cho state vector (5 chiều)
        state_dim = getattr(observation_space, 'spaces', {}).get("state", None)
        if state_dim is not None:
            state_dim = state_dim.shape[0]
        else:
            state_dim = 5
        self.state_mlp = nn.Identity()

        # Final MLP
        self.combined_mlp = nn.Sequential(
            nn.Linear(latent_dim + state_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Chuẩn hóa ảnh
        image = observations["image"]
        if image.ndim == 4 and image.shape[1] == 3:
            image = image.float() / 255.0
        else:
            image = image.permute(0, 3, 1, 2).float() / 255.0
        
        # Đảm bảo tất cả tensor đều trên GPU
        image = image.cuda()
        
        with torch.no_grad():
            z = self.vae_encoder(image)
        
        # Đảm bảo state tensor cùng dtype với model
        state = observations["state"].float()
        state = state.cuda()
        state = self.state_mlp(state)
        
        # Đảm bảo combined tensor cùng dtype với final MLP
        combined = torch.cat([z, state], dim=1)
        dtype = next(self.combined_mlp.parameters()).dtype
        device = next(self.combined_mlp.parameters()).device
        combined = combined.to(device=device, dtype=dtype)
        
        return self.combined_mlp(combined)



def make_env(visualize=True, finetune_mode=False, town="Town07", vehicle="vehicle.tesla.model3"):
    def _init():
        base_env = CarlaEnv(
            num_npcs=3,  # Tăng từ 1 lên 3 NPC
            frame_skip=4,  # Giảm từ 12 xuống 4 để mượt hơn
            visualize=True,
            fixed_delta_seconds=0.02,  # Tăng FPS từ 20 lên 50
            camera_width=160,
            camera_height=80,
            finetune_mode=finetune_mode,  # Thêm finetune mode
            town=town,  # Thêm town parameter
            vehicle=vehicle  # Thêm vehicle parameter
        )

        env = CurriculumEnv(base_env, max_level=3)
        
        # Test observation format
        try:
            obs = env.reset()
            # XÓA CÁC LỆNH PRINT DEBUG
            pass
        except Exception as e:
            pass
        
        return env

    return _init



# Callback chuẩn, lưu checkpoint theo số steps
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path='./checkpoints/',
    name_prefix='sac_carla',
    save_replay_buffer=True  # Nếu model hỗ trợ
)



class OverwriteCheckpointCallback(BaseCallback):
    def __init__(self, save_path, save_freq=20000, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)
            # Save replay buffer nếu có
            if hasattr(self.model, 'save_replay_buffer'):
                buffer_path = self.save_path.replace('.zip', '_replay_buffer.pkl')
                self.model.save_replay_buffer(buffer_path)
            # XÓA print debug khi save replay buffer và checkpoint
        return True



def objective(trial):

    lr = 2e-4

    batch_size = trial.suggest_categorical('batch_size', [512])

    tau = 0.002191

    

    policy_kwargs = dict(

        features_extractor_class=CombinedExtractor,

        features_extractor_kwargs=dict(features_dim=512),  # Giảm features_dim

        net_arch=dict(pi=[512, 512], qf=[512, 512]),  # Giảm network size

        normalize_images=False  # Tắt VecTransposeImage

    )

    # Create environment
    env = DummyVecEnv([make_env() for _ in range(1)])  # type: ignore
    
    # Đảm bảo tất cả model chạy trên GPU
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required but not available!")
        
    model = CustomSAC(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./sac_tensorboard/",
        device="cuda",
        learning_rate=lr,
        buffer_size=10000,  # Tăng buffer size để ổn định hơn
        learning_starts=100,  # Tăng learning starts
        batch_size=batch_size,
        tau=tau,
        ent_coef="auto",  # Tự động điều chỉnh entropy
        policy_kwargs=policy_kwargs,
        use_amp=True,  # Enable AMP
        max_grad_norm=1.0,  # Enable gradient clipping
        train_freq=(1, "step"),  # Train mỗi step
        gradient_steps=1,  # 1 gradient step per update
        target_entropy="auto"  # Tự động điều chỉnh target entropy
    )



    # Train for a short trial.

    model.learn(total_timesteps=10000)

    # Evaluate performance over 1000 steps.

    rewards = []

    obs = env.reset()   

    for _ in range(1000):

        action, _ = model.predict(obs, deterministic=True)  # type: ignore

        obs, reward, done, _ = env.step(action)

        rewards.append(reward)

        if done:

            obs = env.reset()

    avg_reward = float(np.mean(rewards))

    return avg_reward



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train SAC on CARLA environment with advanced features")

    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint/model file to resume training from")

    parser.add_argument("--total_timesteps", type=int, default=150000, help="Total timesteps for training")

    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization before training")

    parser.add_argument("--use_amp", action="store_true", default=True, help="Enable AMP (Automatic Mixed Precision)")

    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")

    parser.add_argument("--carla_path", type=str, default="C:/Corsair profile/CarlaUnreal/CarlaUE4.exe", help="Đường dẫn tới file CarlaUE4.exe")

    parser.add_argument("--carla_args", type=str, default="-carla-map=Town07 -map=Town07 -windowed -ResX=1280 -ResY=720 -quality-level=low -benchmark -fps=60 -force-map-reload", help="Tham số cho Carla simulator")
    parser.add_argument("--force_map_reload", action="store_true", help="Force reload map khi khởi động CARLA")
    parser.add_argument("--vehicle", type=str, default="vehicle.tesla.model3", help="Vehicle blueprint để sử dụng (ví dụ: vehicle.tesla.model3)")
    parser.add_argument("--vae_path", type=str, default="./auto_encoder/model/var_encoder_model.pth", help="Path to pretrained VAE model")
    parser.add_argument("--visualize", action="store_true", default=False, help="Enable visualization during training")
    parser.add_argument("--finetune_mode", action="store_true", default=False, help="Enable finetune mode with random spawn points")
    parser.add_argument("--town", type=str, default="Town07", choices=["Town02", "Town07"], help="Town to use for training")
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning (fixed -> finetune)")
    parser.add_argument("--fixed_timesteps", type=int, default=50000, help="Timesteps for fixed training (curriculum mode)")
    parser.add_argument("--finetune_timesteps", type=int, default=50000, help="Timesteps for finetune training (curriculum mode)")


    args = parser.parse_args()



    def is_carla_running(exe_name="CarlaUnreal.exe"):
        for proc in psutil.process_iter(['name']):
            try:
                if proc.info['name'] and exe_name.lower() in proc.info['name'].lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    # Sử dụng tham số từ command line hoặc default
    carla_path = args.carla_path
    carla_args = args.carla_args
    
    exe_name = os.path.basename(carla_path)
    if is_carla_running(exe_name):
        console.log(f"[yellow]{exe_name} đã chạy, không mở lại![/yellow]")
    else:
        console.log(f"[yellow]Đang khởi động CARLA simulator từ: {carla_path}[/yellow]")
        console.log(f"[yellow]Với tham số: {carla_args}[/yellow]")
        try:
            import shlex
            cmd = [carla_path] + shlex.split(carla_args) + ["-carla-server", "-carla-port=2000"]
            console.log(f"[cyan]Full command: {' '.join(cmd)}[/cyan]")
            subprocess.Popen(cmd, shell=True)
            time.sleep(15)  # Tăng thời gian chờ
        except Exception as e:
            console.log(f"[red]Không thể khởi động CARLA simulator: {e}[/red]")
            sys.exit(1)



    if args.optimize:

        console.log("[bold yellow]Starting hyperparameter optimization...[/bold yellow]")

        study = optuna.create_study(direction='maximize')

        study.optimize(objective, n_trials=10)

        best_params = study.best_trial.params

        console.log(f"[bold green]Best hyperparameters: {best_params}[/bold green]")

        learning_rate = best_params['learning_rate']

        batch_size = best_params['batch_size']

        tau = best_params['tau']

    else:

        learning_rate = 2e-4

        batch_size = 512

        tau = 0.004



    console.rule("[bold green]Starting Training")
    
    # Đảm bảo tất cả model chạy trên GPU
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required but not available!")
        console.log(f"[green]GPU available: {torch.cuda.get_device_name(0)}[/green]")

    if args.curriculum:
        console.log("[bold yellow]=== CURRICULUM LEARNING MODE ===[/bold yellow]")
        console.log(f"[cyan]Town: {args.town}[/cyan]")
        console.log(f"[cyan]Fixed training: {args.fixed_timesteps:,} timesteps[/cyan]")
        console.log(f"[cyan]Finetune training: {args.finetune_timesteps:,} timesteps[/cyan]")
        
        # Giai đoạn 1: Fixed Training
        console.log("[bold green]=== PHASE 1: FIXED TRAINING ===[/bold green]")
        console.log(f"[green]Training with fixed spawn points and routes on {args.town}[/green]")
        
        try:
            # Tạo environment với fixed mode
            env = DummyVecEnv([make_env(visualize=args.visualize, finetune_mode=False, town=args.town, vehicle=args.vehicle) for _ in range(1)])  # type: ignore
            
            # Chạy fixed training
            start_time = time.time()
            result = objective({
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'total_timesteps': args.fixed_timesteps
            })
            fixed_time = time.time() - start_time
            
            console.log(f"[green]Fixed training completed in {fixed_time:.1f} seconds[/green]")
            console.log(f"[green]Result: {result}[/green]")
            
        except Exception as e:
            console.log(f"[red]Fixed training failed: {e}[/red]")
            sys.exit(1)
        
        # Giai đoạn 2: Finetune Training
        console.log("[bold yellow]=== PHASE 2: FINETUNE TRAINING ===[/bold yellow]")
        console.log(f"[yellow]Finetuning with random spawn points on {args.town}[/yellow]")
        
        try:
            # Tạo environment với finetune mode
            env = DummyVecEnv([make_env(visualize=args.visualize, finetune_mode=True, town=args.town, vehicle=args.vehicle) for _ in range(1)])  # type: ignore
            
            # Chạy finetune training với learning rate thấp hơn
            start_time = time.time()
            result = objective({
                'learning_rate': learning_rate * 0.1,  # Giảm learning rate cho finetune
                'batch_size': batch_size,
                'total_timesteps': args.finetune_timesteps
            })
            finetune_time = time.time() - start_time
            
            console.log(f"[yellow]Finetune training completed in {finetune_time:.1f} seconds[/yellow]")
            console.log(f"[yellow]Result: {result}[/yellow]")
            
        except Exception as e:
            console.log(f"[red]Finetune training failed: {e}[/red]")
            sys.exit(1)
        
        # Summary
        total_time = fixed_time + finetune_time
        console.log("[bold green]=== CURRICULUM LEARNING COMPLETED ===[/bold green]")
        console.log(f"[green]Total time: {total_time:.1f} seconds[/green]")
        console.log(f"[green]Fixed phase: {fixed_time:.1f}s ({fixed_time/total_time*100:.1f}%)[/green]")
        console.log(f"[green]Finetune phase: {finetune_time:.1f}s ({finetune_time/total_time*100:.1f}%)[/green]")
        
        sys.exit(0)
    else:
        # Normal training mode
        console.log(f"[cyan]Normal training mode: {args.total_timesteps:,} timesteps on {args.town}[/cyan]")
        console.log(f"[cyan]Using vehicle: {args.vehicle}[/cyan]")
        env = DummyVecEnv([make_env(visualize=args.visualize, finetune_mode=args.finetune_mode, town=args.town, vehicle=args.vehicle) for _ in range(1)])  # type: ignore



    # Đảm bảo tất cả model chạy trên GPU
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required but not available!")
        
    policy_kwargs = dict(
        features_extractor_class=VAEStateExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[512, 512], qf=[512, 512]),
        normalize_images=False  # Tắt VecTransposeImage vì chúng ta dùng VAE
    )



    # Thay thế checkpoint_callback bằng OverwriteCheckpointCallback
    # checkpoint_callback = OverwriteCheckpointCallback(save_path='./checkpoints/sac_carla_latest.zip', save_freq=5000)



    class LossLoggingCallback(BaseCallback):

        def __init__(self, log_interval: int = 1000, verbose: int = 1):

            super(LossLoggingCallback, self).__init__(verbose)

            self.log_interval = log_interval



        def _on_step(self) -> bool:

            if self.n_calls % self.log_interval == 0:

                self.logger.record("custom/progress", self.n_calls)

                if self.verbose > 0:

                    print(f"Step: {self.n_calls}")

            return True



    class StuckDetectionCallback(BaseCallback):     

        def __init__(self, verbose: int = 1):

            super(StuckDetectionCallback, self).__init__(verbose)



        def _on_step(self) -> bool: 

            infos = self.locals.get("infos", [])

            for info in infos:

                if isinstance(info, dict) and info.get("stuck", False):

                    print("[yellow][STUCK CALLBACK] Vehicle was respawned due to being stuck.[/yellow]")

            return True

    class RenderCallback(BaseCallback):
        """Callback để render environment trong training"""
        def __init__(self, render_freq=1, verbose=0):
            super(RenderCallback, self).__init__(verbose)
            self.render_freq = render_freq
            
        def _on_step(self) -> bool:
            if self.n_calls % self.render_freq == 0:
                try:
                    # Gọi render cho environment
                    if hasattr(self.training_env, 'envs'):
                        self.training_env.envs[0].render()  # type: ignore
                except Exception as e:
                    # Ignore render errors để không làm crash training
                    pass
            return True

    class EpisodeEndCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(EpisodeEndCallback, self).__init__(verbose)
            self.episode_count = 0
            self.last_log_step = 0

        def _on_step(self) -> bool:
            # Chỉ log mỗi 1000 steps để tránh spam
            if self.n_calls - self.last_log_step >= 1000:
                self.last_log_step = self.n_calls
                try:
                    if self.locals.get('dones', [False])[0]:
                        self.episode_count += 1
                        # Chỉ log số episode, không log chi tiết
                        if self.episode_count % 10 == 0:  # Log mỗi 10 episodes
                            console.log(f"[cyan]Completed {self.episode_count} episodes[/cyan]")
                except Exception as e:
                    pass  # Ignore callback errors
            return True


    loss_logging_callback = LossLoggingCallback(log_interval=1000, verbose=1)
    stuck_detection_callback = StuckDetectionCallback(verbose=1)
    episode_end_callback = EpisodeEndCallback(verbose=1)
    entropy_logging_callback = EntropyLoggingCallback(log_interval=1000, verbose=1)

    if args.resume is not None and os.path.exists(args.resume):

        console.log(f"[yellow]Resuming training from checkpoint: {args.resume}[/yellow]")

        

        # Load the model with new parameters

        # Đảm bảo tất cả model chạy trên GPU
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required but not available!")
            
        model = CustomSAC.load(
            args.resume, 
            env=env,
            device="cuda",
            total_timesteps_for_entropy=args.total_timesteps,
            use_amp=args.use_amp,  # Use command line argument
            max_grad_norm=args.max_grad_norm  # Use command line argument
        )
        
        # Load replay buffer với logic thông minh
        def find_matching_replay_buffer(checkpoint_path):
            """Tìm file replay buffer tương ứng với checkpoint"""
            checkpoint_dir = os.path.dirname(checkpoint_path)
            checkpoint_name = os.path.basename(checkpoint_path).replace('.zip', '')
            
            # Tìm số steps từ tên checkpoint
            if '_steps' in checkpoint_name:
                # Lấy số steps (ví dụ: từ "sac_carla_95000_steps" -> "95000")
                parts = checkpoint_name.split('_')
                if len(parts) >= 3 and parts[-1] == 'steps':
                    steps_number = parts[-2]
                    expected_buffer_name = f"sac_carla_replay_buffer_{steps_number}_steps.pkl"
                    expected_path = os.path.join(checkpoint_dir, expected_buffer_name)
                    
                    if os.path.exists(expected_path):
                        return expected_path, f"Exact match: {expected_buffer_name}"
            
            # Fallback: tìm file có tên gần nhất
            buffer_files = []
            for file in os.listdir(checkpoint_dir):
                if file.endswith('_replay_buffer.pkl') and 'sac_carla' in file:
                    buffer_files.append(file)
            
            if buffer_files:
                # Sắp xếp theo thời gian tạo (mới nhất trước)
                buffer_files.sort(key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)), reverse=True)
                latest_buffer = os.path.join(checkpoint_dir, buffer_files[0])
                return latest_buffer, f"Latest available: {buffer_files[0]}"
            
            return None, "No replay buffer found"
        
        # Tìm và load replay buffer
        replay_buffer_path, buffer_info = find_matching_replay_buffer(args.resume)
        console.log(f"[cyan]Replay buffer search: {buffer_info}[/cyan]")
        
        if replay_buffer_path and os.path.exists(replay_buffer_path):
            console.log(f"[green]Replay buffer file found: {os.path.basename(replay_buffer_path)}[/green]")
            if hasattr(model, 'load_replay_buffer'):
                try:
                    model.load_replay_buffer(replay_buffer_path)
                    console.log(f"[green]Replay buffer loaded successfully![/green]")
                except Exception as e:
                    console.log(f"[red]Failed to load replay buffer: {e}[/red]")
            else:
                console.log(f"[red]Model does not have load_replay_buffer method[/red]")
        else:
            console.log(f"[yellow]No matching replay buffer found for checkpoint[/yellow]")
        
        # Đảm bảo tất cả model components đều trên GPU
        if hasattr(model, 'policy') and hasattr(model.policy, 'features_extractor') and model.policy.features_extractor is not None:
            model.policy.features_extractor = model.policy.features_extractor.cuda()
        if hasattr(model, 'policy') and hasattr(model.policy, 'actor') and model.policy.actor is not None:
            model.policy.actor = model.policy.actor.cuda()
        if hasattr(model, 'policy') and hasattr(model.policy, 'critic') and model.policy.critic is not None:
            model.policy.critic = model.policy.critic.cuda()
            if hasattr(model.policy.critic, 'qf0'):
                model.policy.critic.qf0 = model.policy.critic.qf0.cuda()
            if hasattr(model.policy.critic, 'qf1'):
                model.policy.critic.qf1 = model.policy.critic.qf1.cuda()
        if hasattr(model, 'policy') and hasattr(model.policy, 'critic_target') and model.policy.critic_target is not None:
            model.policy.critic_target = model.policy.critic_target.cuda()
            if hasattr(model.policy.critic_target, 'qf0'):
                model.policy.critic_target.qf0 = model.policy.critic_target.qf0.cuda()
            if hasattr(model.policy.critic_target, 'qf1'):
                model.policy.critic_target.qf1 = model.policy.critic_target.qf1.cuda()
        
        # Update the logger
        model.set_logger(new_logger)
        
        # Calculate the remaining timesteps
        remaining_timesteps = args.total_timesteps - model.num_timesteps
        
        # Log the current state
        with torch.no_grad():
            if hasattr(model, 'log_ent_coef') and model.log_ent_coef is not None:
                current_alpha = torch.exp(model.log_ent_coef).item()
            else:
                current_alpha = 1.0
        
        console.log(f"[bold green]=== CHECKPOINT LOADED SUCCESSFULLY ===[/bold green]")
        console.log(f"[cyan]Checkpoint file: {args.resume}[/cyan]")
        console.log(f"[cyan]Current entropy coefficient: {current_alpha:.4f}[/cyan]")
        console.log(f"[cyan]Current timesteps: {model.num_timesteps}[/cyan]")
        console.log(f"[cyan]Remaining timesteps: {remaining_timesteps}[/cyan]")
        console.log(f"[cyan]AMP enabled: {model.use_amp}[/cyan]")
        console.log(f"[cyan]Max gradient norm: {model.max_grad_norm}[/cyan]")
        
        # Kiểm tra model components
        if hasattr(model, 'policy') and hasattr(model.policy, 'features_extractor'):
            console.log(f"[green]Features extractor: {type(model.policy.features_extractor).__name__}[/green]")
        if hasattr(model, 'policy') and hasattr(model.policy, 'actor'):
            console.log(f"[green]Actor: {type(model.policy.actor).__name__}[/green]")
        if hasattr(model, 'policy') and hasattr(model.policy, 'critic'):
            console.log(f"[green]Critic: {type(model.policy.critic).__name__}[/green]")
    else:
        console.log("[cyan]Creating a new model.[/cyan]")
        
        # THÊM DEBUG VAE
        try:
            from auto_encoder.encoder import VariationalEncoder
            console.log(f"[cyan]Testing VAE encoder...[/cyan]")
            
            # Test VAE creation
            test_vae = VariationalEncoder(95)
            console.log(f"[green]VAE created successfully[/green]")
            
            # Test VAE loading
            vae_path = args.vae_path
            if os.path.exists(vae_path):
                test_vae.load()
                console.log(f"[green]VAE loaded from {vae_path}[/green]")
            else:
                console.log(f"[yellow]VAE file not found at {vae_path}, using random weights[/yellow]")
            
            # Luôn sử dụng GPU cho VAE
            if not torch.cuda.is_available():
                raise RuntimeError("GPU is required but not available!")
                
            test_vae = test_vae.cuda()
            test_vae = test_vae.half()  # Sử dụng FP16
            console.log(f"[green]VAE moved to GPU with FP16[/green]")
            
            # Test VAE forward pass
            test_vae.eval()
            for param in test_vae.parameters():
                param.requires_grad = False
                
            test_image = torch.randn(1, 3, 80, 160, device="cuda", dtype=torch.float16)  # Đưa input lên GPU với FP16
            with torch.no_grad():
                test_output = test_vae(test_image)
            console.log(f"[green]VAE test successful: input {test_image.shape} -> output {test_output.shape}[/green]")
            
        except Exception as e:
            console.log(f"[red]VAE test failed: {e}[/red]")
            import traceback
            console.log(f"[red]VAE traceback: {traceback.format_exc()}[/red]")
        
        # Đảm bảo tất cả model chạy trên GPU
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required but not available!")
            
        model = CustomSAC(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log="./sac_tensorboard/",
            device="cuda",
            learning_rate=learning_rate,
            buffer_size=10000,  # Tăng buffer size để ổn định hơn
            learning_starts=100,  # Tăng learning starts
            batch_size=batch_size,
            tau=tau,
            policy_kwargs=policy_kwargs,
            use_amp=True,  # Enable AMP
            max_grad_norm=1.0,  # Enable gradient clipping
            train_freq=(1, "step"),  # Train mỗi step
            gradient_steps=1,  # 1 gradient step per update
            ent_coef="auto",  # Tự động điều chỉnh entropy
            target_entropy="auto"  # Tự động điều chỉnh target entropy
        )
        
        # Đảm bảo tất cả model components đều trên GPU
        if hasattr(model, 'policy') and hasattr(model.policy, 'features_extractor') and model.policy.features_extractor is not None:
            model.policy.features_extractor = model.policy.features_extractor.cuda()
        if hasattr(model, 'policy') and hasattr(model.policy, 'actor') and model.policy.actor is not None:
            model.policy.actor = model.policy.actor.cuda()
        if hasattr(model, 'policy') and hasattr(model.policy, 'critic') and model.policy.critic is not None:
            model.policy.critic = model.policy.critic.cuda()
            if hasattr(model.policy.critic, 'qf0'):
                model.policy.critic.qf0 = model.policy.critic.qf0.cuda()
            if hasattr(model.policy.critic, 'qf1'):
                model.policy.critic.qf1 = model.policy.critic.qf1.cuda()
        if hasattr(model, 'policy') and hasattr(model.policy, 'critic_target') and model.policy.critic_target is not None:
            model.policy.critic_target = model.policy.critic_target.cuda()
            if hasattr(model.policy.critic_target, 'qf0'):
                model.policy.critic_target.qf0 = model.policy.critic_target.qf0.cuda()
            if hasattr(model.policy.critic_target, 'qf1'):
                model.policy.critic_target.qf1 = model.policy.critic_target.qf1.cuda()



        console.log("[bold green]Starting training...")

    # Setup callbacks
    callbacks = [
        checkpoint_callback,
        loss_logging_callback,
        stuck_detection_callback,
        episode_end_callback,
        entropy_logging_callback
    ]

    # THÊM TRY-CATCH VÀ DEBUG CHI TIẾT
    try:
        console.log(f"[cyan]Training for {args.total_timesteps} timesteps...[/cyan]")
        console.log(f"[cyan]Model device: {model.device}[/cyan]")
        console.log(f"[cyan]Environment observation space: {env.observation_space}[/cyan]")
        console.log(f"[cyan]Model policy: {model.policy}[/cyan]")
        
        console.log(f"[cyan]Setup {len(callbacks)} callbacks:[/cyan]")
        for i, callback in enumerate(callbacks):
            console.log(f"[cyan]  {i+1}. {type(callback).__name__}[/cyan]")
        
        # Test single step trước khi train
        console.log(f"[cyan]Testing single step...[/cyan]")
        try:
            obs = env.reset()
            # SỬA: VecEnv trả về tuple, cần lấy phần tử đầu tiên
            if isinstance(obs, tuple):
                obs = obs[0]
            console.log(f"[cyan]Reset successful, obs keys: {obs.keys() if isinstance(obs, dict) else 'not dict'}[/cyan]")
            
            action, _states = model.predict(obs, deterministic=True)
            console.log(f"[cyan]Predict successful, action shape: {action.shape}[/cyan]")
            
            obs, reward, done, info = env.step(action)
            # SỬA: VecEnv trả về tuple, cần lấy phần tử đầu tiên
            if isinstance(obs, tuple):
                obs = obs[0]
            if isinstance(reward, (list, tuple)):
                reward = reward[0]
            if isinstance(done, (list, tuple)):
                done = done[0]
            console.log(f"[green]Test observation shape: {obs['latent'].shape}, {obs['state'].shape}[/green]")
            console.log(f"[green]Test action shape: {action.shape}[/green]")
            console.log(f"[green]Test step successful: reward={reward}, done={done}[/green]")
            console.log(f"[green]All tests passed, starting training...[/green]")
        except Exception as e:
            console.log(f"[red]Test step failed: {e}[/red]")
            import traceback
            console.log(f"[red]Test traceback: {traceback.format_exc()}[/red]")
            raise e
        
        # THÊM DEBUG: Kiểm tra model state
        console.log(f"[cyan]Model learning rate: {model.learning_rate}[/cyan]")
        console.log(f"[cyan]Model buffer size: {model.buffer_size}[/cyan]")
        console.log(f"[cyan]Model batch size: {model.batch_size}[/cyan]")
        
        # Bắt đầu training với try-catch
        console.log(f"[bold cyan]=== STARTING MODEL.LEARN() ===[/bold cyan]")
        console.log(f"[cyan]About to call model.learn() with {args.total_timesteps} timesteps[/cyan]")
        try:
            # Thêm debug trước khi gọi learn
            console.log(f"[cyan]Model state before learn:[/cyan]")
            console.log(f"[cyan]  - num_timesteps: {model.num_timesteps}[/cyan]")
            console.log(f"[cyan]  - learning_starts: {model.learning_starts}[/cyan]")
            console.log(f"[cyan]  - buffer_size: {model.buffer_size}[/cyan]")
            
            console.log(f"[cyan]Calling model.learn() now...[/cyan]")
            try:
                # Thêm debug trước khi gọi learn
                console.log(f"[cyan]Environment state before learn:[/cyan]")
                console.log(f"[cyan]  - env.num_envs: {env.num_envs}[/cyan]")
                console.log(f"[cyan]  - env.observation_space: {env.observation_space}[/cyan]")
                console.log(f"[cyan]  - env.action_space: {env.action_space}[/cyan]")
                
                # Test một step nữa để đảm bảo environment hoạt động
                console.log(f"[cyan]Testing environment one more time...[/cyan]")
                try:
                    console.log(f"[cyan]Calling env.reset()...[/cyan]")
                    test_obs = env.reset()
                    console.log(f"[cyan]env.reset() completed[/cyan]")
                    
                    console.log(f"[cyan]Calling env.step()...[/cyan]")
                    test_action = np.array([[0.0, 0.5, 0.0]])  # Throttle only
                    test_obs, test_reward, test_done, test_info = env.step(test_action)
                    console.log(f"[cyan]env.step() completed[/cyan]")
                    
                    console.log(f"[cyan]Test step successful: reward={test_reward}, done={test_done}[/cyan]")
                except Exception as e:
                    console.log(f"[red]Test environment failed: {e}[/red]")
                    import traceback
                    console.log(f"[red]Test environment traceback: {traceback.format_exc()}[/red]")
                    raise e
                
                model.learn(
                    total_timesteps=args.total_timesteps, 
                    callback=callbacks,
                    log_interval=10,  # Tăng lên 10 để giảm overhead
                    progress_bar=True  # Hiển thị progress bar
                )
            except Exception as e:
                console.log(f"[bold red]ERROR in model.learn() call: {e}[/bold red]")
                import traceback
                console.log(f"[red]Learn traceback: {traceback.format_exc()}[/red]")
                raise e
            console.log(f"[bold green]=== TRAINING COMPLETED SUCCESSFULLY ===[/bold green]")
        except Exception as e:
            console.log(f"[bold red]ERROR in model.learn(): {e}[/bold red]")
            import traceback
            console.log(f"[red]Full traceback: {traceback.format_exc()}[/red]")
            raise e
        
    except Exception as e:
        console.log(f"[bold red]ERROR during training: {e}[/bold red]")
        console.log(f"[red]Error type: {type(e).__name__}[/red]")
        import traceback
        console.log(f"[red]Traceback: {traceback.format_exc()}[/red]")
        raise e
        
        # Thêm debug thêm
        console.log(f"[red]Model state:[/red]")
        console.log(f"[red]  - num_timesteps: {getattr(model, 'num_timesteps', 'N/A')}[/red]")
        console.log(f"[red]  - learning_starts: {getattr(model, 'learning_starts', 'N/A')}[/red]")
        console.log(f"[red]  - buffer_size: {getattr(model, 'buffer_size', 'N/A')}[/red]")
        
        raise e



    # After model creation, update the environment with the model reference
    # Option 1: Add type ignore
    env.envs[0].model = model  # type: ignore

    # Option 2: Use setattr
    # setattr(env.envs[0], 'model', model)

    # Save replay buffer cùng model
    model_path = "sac_carla_model_enhanced"
    model.save(model_path)
    if hasattr(model, 'save_replay_buffer'):
        replay_buffer_path = model_path + "_replay_buffer.pkl"
        model.save_replay_buffer(replay_buffer_path)
        console.log(f"[green]Replay buffer saved to {replay_buffer_path}[/green]")
    console.log(f"[bold green]Model saved to {model_path}[/bold green]")