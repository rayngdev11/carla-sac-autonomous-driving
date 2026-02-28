import os
import gym 
from gym import spaces
import numpy as np
import carla
import random
import pygame
import cv2 
import time
from rich.console import Console
import math
import torch
from stable_baselines3 import SAC
from auto_encoder.encoder import VariationalEncoder

console = Console()

class CustomSAC(SAC):
    def __init__(self, *args, total_timesteps_for_entropy=150000, use_amp=True, max_grad_norm=1.0, **kwargs):
        kwargs.pop('logger', None)
        if 'ent_coef' in kwargs:
            del kwargs['ent_coef']
        super().__init__(*args, ent_coef="auto", **kwargs)
        
        self.total_timesteps_for_entropy = total_timesteps_for_entropy
        self.num_timesteps_at_start = self.num_timesteps
        self.initial_alpha = 1.0
        self.min_alpha = 0.01  
        self.current_alpha = self.initial_alpha  
        
        # AMP and gradient clipping settings
        self.use_amp = use_amp
        self.max_grad_norm = max_grad_norm
        # Initialize AMP scaler if enabled
        if self.use_amp:
            try:
                self.scaler = torch.cuda.amp.GradScaler()
            except Exception:
                self.scaler = None
        else:
            self.scaler = None

    def learn(self, total_timesteps, callback=None, log_interval=4, tb_log_name="SAC", reset_num_timesteps=True, progress_bar=False):
        if self.num_timesteps_at_start == 0 or reset_num_timesteps:
            self.num_timesteps_at_start = self.num_timesteps
        return super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar)

    def _update_learning_rate(self, optimizers):
        """Update the entropy coefficient based on training progress"""
        super()._update_learning_rate(optimizers)
        
        # Calculate progress based on total timesteps since start
        total_elapsed = self.num_timesteps
        progress_fraction = min(1.0, total_elapsed / self.total_timesteps_for_entropy)
        
        # Linear interpolation between initial_alpha and min_alpha
        new_alpha = self.initial_alpha * (1.0 - progress_fraction) + self.min_alpha * progress_fraction
        self.current_alpha = new_alpha

        # Update the entropy coefficient
        with torch.no_grad():
            if self.log_ent_coef is not None:
                self.log_ent_coef.copy_(torch.log(torch.tensor([new_alpha], device=self.device)))

        if self.logger is not None:
            self.logger.record("train/entropy_coefficient", new_alpha)
            
        if self.num_timesteps % 1000 == 0:
            print(f"[DEBUG] timesteps: {self.num_timesteps}, progress: {progress_fraction:.4f}, entropy_coef: {new_alpha:.4f}")

    @classmethod
    def load(cls, path, env=None, device="auto", custom_objects=None, force_reset=True, total_timesteps_for_entropy=150000, use_amp=True, max_grad_norm=1.0, **kwargs):
        # Load the model
        model = super().load(path, env, device, custom_objects, force_reset, **kwargs)
        
        # Set custom attributes
        model.total_timesteps_for_entropy = total_timesteps_for_entropy
        model.use_amp = use_amp
        model.max_grad_norm = max_grad_norm
        
        # Initialize AMP scaler if enabled
        if model.use_amp:
            model.scaler = torch.cuda.amp.GradScaler()
        else:
            model.scaler = None
            
        return model

class CarlaEnv(gym.Env):
    def __init__(self, num_npcs=3, frame_skip=4, visualize=True,
                 fixed_delta_seconds=0.02, camera_width=84, camera_height=84,
                 finetune_mode=False, town="Town02", vehicle="vehicle.tesla.model3"):
        super(CarlaEnv, self).__init__()
        self.visualize = visualize
        self.frame_skip = frame_skip
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.num_npcs = num_npcs
        self.finetune_mode = finetune_mode
        self.town = town
        self.vehicle_name = vehicle  # Lưu vehicle name để sử dụng sau

        # Curriculum learning parameters
        self.episode_count = 0
        self.fixed_training_episodes = 1000  # Số episode fixed training trước khi finetune
        
        # Checkpoint-teleport logic
        self.checkpoint_waypoint_index = 0
        self.current_waypoint_index = 0
        self.checkpoint_frequency = 100  # Đồng bộ với file tham khảo
        
        # Map configurations (từ code PPO gốc)
        self.map_configs = {
            "Town02": {
                "spawn_point": 1, 
                "total_distance": 780,
                "route_logic": "mixed"  # 650m forward, 130m backward
            },
            "Town07": {
                "spawn_point": 38,
                "total_distance": 750,
                "route_logic": "mixed"  # 650m forward, 100m backward
            }
        }

        # Luôn sử dụng GPU cho tất cả model
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.vae = VariationalEncoder(latent_dims=95)
            self.vae.load()
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False
            self.vae = self.vae.to(self.device)
            self.vae = self.vae.half()  # Sử dụng FP16 để tăng tốc
            console.log(f"[green]VAE loaded on GPU with FP16 optimization[/green]")
        else:
            raise RuntimeError("GPU is required but not available!")
        # Observation space với VAE latent vector
        # Sử dụng torch.float16 cho GPU, torch.float32 cho CPU
        if torch.cuda.is_available():
            dtype = np.float16
        else:
            dtype = np.float32
            
        self.observation_space = spaces.Dict({
            "latent": spaces.Box(low=-np.inf, high=np.inf, shape=(95,), dtype=dtype),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=dtype)
        })

        # Handle SDL video driver configuration
        if self.visualize:
            if os.environ.get("SDL_VIDEODRIVER") == "dummy":
                del os.environ["SDL_VIDEODRIVER"]
            import pygame
            self.pygame = pygame
        else:
            if not os.environ.get("SDL_VIDEODRIVER"):
                os.environ["SDL_VIDEODRIVER"] = "dummy"

        # Set display dimensions cho HD với FPS cao
        if self.visualize:
            self.display_width = 1280  # HD display width
            self.display_height = 720  # HD display height
            pygame.init()
            self.display = pygame.display.set_mode((self.display_width, self.display_height))
            pygame.display.set_caption("Third Person View - HD (60 FPS)")
            self.clock = pygame.time.Clock()
            pygame.event.set_allowed([pygame.QUIT])
            # Tối ưu pygame cho FPS cao
            pygame.display.flip()
        else:
            self.display_width = camera_width
            self.display_height = camera_height
            self.display = None
            self.clock = None

        # Connect to CARLA server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(120.0)
        self.world = self.client.get_world()
        
        # Kiểm tra và load đúng map
        current_map = self.world.get_map()
        current_map_name = current_map.name
        
        console.log(f"[cyan]Current map: {current_map_name}[/cyan]")
        console.log(f"[cyan]Requested town: {self.town}[/cyan]")
        
        # Nếu map hiện tại không đúng, load lại map
        if current_map_name != self.town:
            console.log(f"[yellow]Loading map {self.town}...[/yellow]")
            try:
                self.world = self.client.load_world(self.town)
                console.log(f"[green]Successfully loaded {self.town}[/green]")
            except Exception as e:
                console.log(f"[red]Failed to load {self.town}: {e}[/red]")
                console.log(f"[yellow]Using current map: {current_map_name}[/yellow]")
        
        self.map = self.world.get_map()
        console.log(f"[green]Using map: {self.map.name}[/green]")

        # Enable synchronous mode with faster settings for higher FPS
        settings = self.world.get_settings()
        if not settings.synchronous_mode:
            settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.02  # Tăng FPS từ 20 lên 50 FPS
        # SỬA: Luôn enable rendering để semantic sensor hoạt động
        settings.no_rendering_mode = False  # Luôn False để semantic sensor hoạt động
        self.world.apply_settings(settings)
        console.log(f"[green]Synchronous mode enabled (fixed_delta_seconds = {settings.fixed_delta_seconds}, FPS = {1/settings.fixed_delta_seconds:.1f})[/green]")

        # Observation space đã được định nghĩa ở trên với VAE latent

        # Action space
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Initialize variables
        self.vehicle = None
        self.camera = None
        self.camera_image = None
        self.camera_image_obs = None
        self.npc_vehicles = []
        self.previous_location = None
        self.previous_speed = 0.0
        self.collision_history = False
        self.lane_invasion_history = False
        self.stuck_counter = 0
        self.episode_count = 0
        self.render_counter = 0
        
        # PPO-style parameters
        self.collision_counter = 0
        self.episode_start_time = time.time()
        
        # Waypoint navigation
        self.waypoints = []
        self.current_waypoint_index = 0
        self.lidar_min_distance = float('inf')
        
        # Timestep counter
        self.timesteps = 0
        
        # Cache cho VAE để giảm latency
        self.vae_cache = {}
        self.vae_cache_size = 10  # Cache 10 frames gần nhất

        # Lưu frame sensor
        self.last_rgb = None
        self.last_depth = None
        self.last_segmentation = None
        # XÓA: self.third_person_camera = None
        # XÓA: self.third_person_image = None

    def _wait_for_sensors(self):
        """Đợi tất cả sensors load xong trước khi train"""
        console.log("[cyan]Waiting for sensors to load...[/cyan]")
        
        # Đợi camera sensor
        if hasattr(self, 'camera_sensor') and self.camera_sensor:
            console.log("[yellow]Waiting for camera sensor...[/yellow]")
            timeout = 10.0
            start_time = time.time()
            while not hasattr(self, 'camera_image') or self.camera_image is None:
                time.sleep(0.01)
                if time.time() - start_time > timeout:
                    console.log("[red]Camera sensor timeout![/red]")
                    break
            console.log("[green]Camera sensor loaded![/green]")
        
        # Đợi collision sensor
        if hasattr(self, 'collision_sensor') and self.collision_sensor:
            console.log("[yellow]Waiting for collision sensor...[/yellow]")
            time.sleep(0.1)
            console.log("[green]Collision sensor loaded![/green]")
        
        # Đợi lane invasion sensor
        if hasattr(self, 'lane_invasion_sensor') and self.lane_invasion_sensor:
            console.log("[yellow]Waiting for lane invasion sensor...[/yellow]")
            time.sleep(0.1)
            console.log("[green]Lane invasion sensor loaded![/green]")
        
        # Đợi collision sensor
        if hasattr(self, 'collision_sensor') and self.collision_sensor:
            console.log("[yellow]Waiting for collision sensor...[/yellow]")
            time.sleep(0.1)
            console.log("[green]Collision sensor loaded![/green]")
        
        # Đợi LIDAR sensor
        if hasattr(self, 'lidar_sensor') and self.lidar_sensor:
            console.log("[yellow]Waiting for LIDAR sensor...[/yellow]")
            time.sleep(0.1)
            console.log("[green]LIDAR sensor loaded![/green]")
        
        # Tick world để đảm bảo tất cả sensors hoạt động
        console.log("[yellow]Finalizing sensor setup...[/yellow]")
        for i in range(5):
            self.world.tick()
            time.sleep(0.01)
        
        console.log("[bold green]All sensors loaded successfully![/bold green]")

    def _get_vehicle_blueprint(self, vehicle_name="vehicle.tesla.model3"):
        """Get specific vehicle blueprint with random color - giống code của bạn"""
        try:
            blueprint = self.world.get_blueprint_library().find(vehicle_name)
            if blueprint is None:
                # Fallback nếu không tìm thấy
                blueprints = self.world.get_blueprint_library().filter('vehicle.*')
                blueprint = blueprints[0]
            
            # Random color như code của bạn
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            
            return blueprint
        except Exception as e:
            console.log(f"[red]Error getting vehicle blueprint: {e}[/red]")
            # Fallback
            blueprints = self.world.get_blueprint_library().filter('vehicle.*')
            return blueprints[0] if blueprints else None

    def _get_small_vehicle_blueprints(self):
        """Get small vehicle blueprints for spawning NPCs only"""
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        # SỬA: Ưu tiên các vehicle nhỏ và ổn định
        preferred_vehicles = [
            'vehicle.tesla.model3',
            'vehicle.audi.a2', 
            'vehicle.audi.tt',
            'vehicle.bmw.grandtourer',
            'vehicle.citroen.c3',
            'vehicle.dodge.charger_2020',
            'vehicle.ford.mustang',
            'vehicle.mercedes.coupe',
            'vehicle.mini.cooper_s',
            'vehicle.nissan.micra',
            'vehicle.seat.leon',
            'vehicle.toyota.prius',
            'vehicle.volkswagen.t2'
        ]
        
        # Tìm các vehicle được ưu tiên
        preferred_bps = []
        for vehicle_id in preferred_vehicles:
            try:
                bp = self.world.get_blueprint_library().find(vehicle_id)
                if bp:
                    preferred_bps.append(bp)
            except:
                continue
                
        # Nếu có vehicle được ưu tiên, sử dụng chúng
        if preferred_bps:
            return preferred_bps
        else:
            # Fallback: tìm vehicle nhỏ
            small_vehicles = [bp for bp in blueprints if 'mini' in bp.id or 'tesla' in bp.id or 'audi' in bp.id]
            return small_vehicles if small_vehicles else blueprints

    def spawn_npcs(self):
        """Spawn NPC vehicles"""
        try:
            blueprints = self._get_small_vehicle_blueprints()
            spawn_points = self.world.get_map().get_spawn_points()
            
            for i in range(self.num_npcs):
                blueprint = random.choice(blueprints)
                spawn_point = random.choice(spawn_points)
                
                npc = self.world.try_spawn_actor(blueprint, spawn_point)
                if npc:
                    npc.set_autopilot(True)
                    self.npc_vehicles.append(npc)
                    # XÓA LOG NPC SPAWN ĐỂ GIẢM SPAM
                    
        except Exception as e:
            console.log(f"[red]Error spawning NPCs: {e}[/red]")

    def reset(self):
        """Reset environment với curriculum learning và checkpoint-teleport"""
        try:
            console.log(f"[cyan]--- RESET START --- (fresh_start={getattr(self, 'fresh_start', True)}) ---[/cyan]")
            # Luôn reset về đầu route (waypoint 0), bỏ teleport về checkpoint
            self.current_waypoint_index = 0
            console.log(f"[green]Reset: Spawn ở đầu route (waypoint 0)")
            # Nếu cần, tạo lại self.waypoints ở đây
            # ... giữ nguyên logic tạo self.waypoints ...
            # Reset lại các biến counter
            self.timesteps = 0
            self.stuck_counter = 0
            # Kiểm tra vehicle và sensor
            if not self.vehicle:
                console.log(f"[red]Reset ERROR: Vehicle is None![/red]")
            if not hasattr(self, 'camera_sensor') or self.camera_sensor is None:
                console.log(f"[red]Reset WARNING: Camera sensor is None![/red]")
            if not hasattr(self, 'segmentation_sensor') or self.segmentation_sensor is None:
                console.log(f"[yellow]Reset WARNING: Segmentation sensor is None![/yellow]")
            self.fresh_start = True  # Đặt lại cho episode mới
            console.log(f"[cyan]--- RESET END ---[/cyan]")
        except Exception as e:
            console.log(f"[red]Exception in reset: {e}[/red]")
            import traceback
            console.log(f"[red]{traceback.format_exc()}[/red]")
            raise
        # Debug reset
        if hasattr(self, 'debug_reset_count'):
            self.debug_reset_count += 1
        else:
            self.debug_reset_count = 1
        
        if self.debug_reset_count % 5 == 0:  # Log mỗi 5 resets
            console.log(f"[cyan]Reset {self.debug_reset_count} (Episode {self.episode_count}) - Finetune: {self.finetune_mode}[/cyan]")
        
        # Cleanup với delay để tránh sensor conflict
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None
        for npc in self.npc_vehicles:
            try:
                npc.destroy()
            except:
                pass
        self.npc_vehicles = []
        if self.camera:
            self.camera.destroy()
            self.camera = None

        # Destroy old sensors với delay
        if hasattr(self, 'camera_sensor') and self.camera_sensor:
            try:
                self.camera_sensor.destroy()
                self.camera_sensor = None
                time.sleep(0.2)  # Đợi sensor destroy hoàn toàn
            except:
                pass
        if hasattr(self, 'collision_sensor') and self.collision_sensor:
            try:
                self.collision_sensor.destroy()
                self.collision_sensor = None
                time.sleep(0.2)
            except:
                pass
        if hasattr(self, 'lane_invasion_sensor') and self.lane_invasion_sensor:
            try:
                self.lane_invasion_sensor.destroy()
                self.lane_invasion_sensor = None
                time.sleep(0.2)
            except:
                pass
        if hasattr(self, 'lidar_sensor') and self.lidar_sensor:
            try:
                self.lidar_sensor.destroy()
                self.lidar_sensor = None
                time.sleep(0.2)
            except:
                pass
        if hasattr(self, 'segmentation_sensor') and self.segmentation_sensor:
            try:
                self.segmentation_sensor.destroy()
                self.segmentation_sensor = None
                time.sleep(0.2)
            except:
                pass
        # XÓA: self.third_person_camera = None
        # XÓA: self.third_person_image = None

        # Curriculum Learning: Chọn spawn point theo mode
        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available")
            
        if self.finetune_mode and self.episode_count > self.fixed_training_episodes:
            # Finetune mode: Random spawn point
            spawn_point = random.choice(spawn_points)
            console.log(f"[yellow]Finetune mode: Random spawn point[/yellow]")
        else:
            # Fixed mode: Sử dụng spawn point cố định theo map config
            if self.town in self.map_configs:
                config = self.map_configs[self.town]
                spawn_idx = config["spawn_point"]
                if spawn_idx < len(spawn_points):
                    spawn_point = spawn_points[spawn_idx]
                    console.log(f"[green]Fixed mode: Spawn point {spawn_idx} for {self.town}[/green]")
                else:
                    spawn_point = random.choice(spawn_points)
                    console.log(f"[yellow]Spawn point {spawn_idx} not available, using random[/yellow]")
            else:
                spawn_point = random.choice(spawn_points)
                console.log(f"[yellow]Town {self.town} not in config, using random spawn[/yellow]")

        # Spawn vehicle với blueprint cố định - giống code của bạn
        max_spawn_attempts = 10
        for attempt in range(max_spawn_attempts):
            try:
                # Sử dụng vehicle cố định thay vì random
                vehicle_bp = self._get_vehicle_blueprint(self.vehicle_name)
                if vehicle_bp is None:
                    continue
                    
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if self.vehicle is not None:
                    console.log(f"[green]Spawned vehicle: {vehicle_bp.id}[/green]")
                    break
            except Exception as e:
                console.log(f"[yellow]Spawn attempt {attempt+1} failed: {e}[/yellow]")
                continue
        if self.vehicle is None:
            console.log(f"[red]All spawn attempts failed. Continuing with vehicle=None[/red]")

        # Setup sensors nếu vehicle spawn thành công
        if self.vehicle is not None:
            # Spawn NPCs
            self.spawn_npcs()
            # Đợi lâu hơn để đảm bảo cleanup hoàn tất
            for _ in range(10):
                self.world.tick()
                time.sleep(0.02)
            # Camera RGB góc nhìn thứ 3 (cao hơn, lùi ra sau hơn)
            try:
                camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
                if camera_bp is None:
                    self.camera_image_obs = np.zeros((80, 160, 3), dtype=np.uint8)
                else:
                    camera_bp.set_attribute('image_size_x', '1920')
                    camera_bp.set_attribute('image_size_y', '1080')
                    camera_bp.set_attribute('fov', '90')
                    # Góc nhìn thứ 3 (cao hơn, lùi ra sau hơn, thấy biển số xe)
                    camera_transform = carla.Transform(
                        carla.Location(x=-6, z=3.5),
                        carla.Rotation(pitch=-12, yaw=0)
                    )
                    self.camera_sensor = self.world.spawn_actor(
                        camera_bp,
                        camera_transform,
                        attach_to=self.vehicle
                    )
                    self.camera_sensor.listen(self._on_camera_update)
                    for _ in range(10):
                        self.world.tick()
                        time.sleep(0.01)
                    timeout = 5.0
                    start_time = time.time()
                    while not hasattr(self, 'camera_image') or self.camera_image is None:
                        time.sleep(0.02)
                        if time.time() - start_time > timeout:
                            console.log("[yellow]Camera sensor timeout, using fallback[/yellow]")
                            self.camera_image_obs = np.zeros((80, 160, 3), dtype=np.uint8)
                            break
                    else:
                        self.camera_image_obs = cv2.resize(self.camera_image, (self.camera_width, self.camera_height))
                        if not hasattr(self, 'camera_image') or self.camera_image is None:
                            self.camera_image = self.camera_image_obs.copy()
            except Exception as e:
                console.log(f"[red]Error setting up camera: {e}[/red]")
                self.camera_image_obs = np.zeros((80, 160, 3), dtype=np.uint8)
            # Camera segmentation góc nhìn thứ 3 (cao hơn, lùi ra sau hơn)
            try:
                seg_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
                if seg_bp is not None:
                    seg_bp.set_attribute('image_size_x', '1920')
                    seg_bp.set_attribute('image_size_y', '1080')
                    seg_bp.set_attribute('fov', '90')
                    seg_transform = carla.Transform(
                        carla.Location(x=-6, z=3.5),
                        carla.Rotation(pitch=-12, yaw=0)
                    )
                    self.segmentation_sensor = self.world.spawn_actor(
                        seg_bp,
                        seg_transform,
                        attach_to=self.vehicle
                    )
                    self.segmentation_sensor.listen(self._on_segmentation_update)
            except Exception as e:
                console.log(f"[yellow]Error setting up segmentation sensor: {e}[/yellow]")
                self.segmentation_sensor = None
            # Depth sensor (nếu có) cũng dùng góc nhìn này
            try:
                depth_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
                if depth_bp is not None:
                    depth_bp.set_attribute('image_size_x', '1920')
                    depth_bp.set_attribute('image_size_y', '1080')
                    depth_bp.set_attribute('fov', '90')
                    depth_transform = carla.Transform(
                        carla.Location(x=-6, z=3.5),
                        carla.Rotation(pitch=-12, yaw=0)
                    )
                    self.depth_sensor = self.world.spawn_actor(
                        depth_bp,
                        depth_transform,
                        attach_to=self.vehicle
                    )
                    self.depth_sensor.listen(self._on_depth_update)
            except Exception as e:
                self.depth_sensor = None
            # LIDAR sensor (nếu có) cũng dùng góc nhìn này
            try:
                lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
                lidar_bp.set_attribute('range', '50')
                lidar_bp.set_attribute('rotation_frequency', '10')
                lidar_bp.set_attribute('channels', '16')
                lidar_bp.set_attribute('points_per_second', '20000')
                lidar_bp.set_attribute('upper_fov', '10.0')
                lidar_bp.set_attribute('lower_fov', '-30.0')
                lidar_bp.set_attribute('horizontal_fov', '100.0')
                lidar_transform = carla.Transform(
                    carla.Location(x=-6, z=3.5),
                    carla.Rotation(pitch=-12, yaw=0)
                )
                self.lidar_sensor = self.world.spawn_actor(
                    lidar_bp,
                    lidar_transform,
                    attach_to=self.vehicle
                )
                self.lidar_min_distance = float('inf')
                self.lidar_sensor.listen(self._on_lidar_update)
            except Exception as e:
                console.log(f"[yellow]Error setting up LIDAR sensor: {e}[/yellow]")
                self.lidar_sensor = None
            
            # Collision sensor
            try:
                collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
                collision_transform = carla.Transform(carla.Location(x=1.3, z=0.5))
                self.collision_sensor = self.world.spawn_actor(
                    collision_bp,
                    collision_transform,
                    attach_to=self.vehicle
                )
                self.collision_sensor.listen(self._on_collision)
                self.collision_history = False
                console.log(f"[green]Collision sensor setup successfully[/green]")
            except Exception as e:
                console.log(f"[yellow]Error setting up collision sensor: {e}[/yellow]")
                self.collision_sensor = None
            
            # Lane invasion sensor
            try:
                lane_invasion_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
                lane_invasion_transform = carla.Transform()
                self.lane_invasion_sensor = self.world.spawn_actor(
                    lane_invasion_bp,
                    lane_invasion_transform,
                    attach_to=self.vehicle
                )
                self.lane_invasion_sensor.listen(self._on_lane_invasion)
                self.lane_invasion_history = False
                console.log(f"[green]Lane invasion sensor setup successfully[/green]")
            except Exception as e:
                console.log(f"[yellow]Error setting up lane invasion sensor: {e}[/yellow]")
                self.lane_invasion_sensor = None
        else:
            console.log(f"[yellow]Vehicle spawn failed, skipping sensor setup[/yellow]")
            self.camera_image_obs = np.zeros((80, 160, 3), dtype=np.uint8)

        # Warm-up và validation chỉ khi vehicle tồn tại
        if self.vehicle is not None:
            # Tăng warm-up time để đảm bảo ổn định
            for i in range(15):
                try:
                    self.world.tick()
                    time.sleep(0.01)  # Thêm delay nhỏ
                except Exception as e:
                    if i == 14:
                        console.log(f"[yellow]Warm-up failed at step {i}: {e}[/yellow]")
                        return self._get_default_observation()
            try:
                transform = self.vehicle.get_transform()
                velocity = self.vehicle.get_velocity()
                speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
                self.previous_location = transform.location
                self.previous_speed = speed
            except Exception as e:
                console.log(f"[yellow]Vehicle validation failed: {e}[/yellow]")
                return self._get_default_observation()
        else:
            try:
                self.world.tick()
            except Exception as e:
                pass

        # Reset sensor history
        self.collision_history = False
        self.lane_invasion_history = False
        self.stuck_counter = 0
        self.collision_counter = 0
        self.lidar_min_distance = float('inf')
        self.timesteps = 0  # Reset timestep counter
        if self.vehicle is not None:
            try:
                self.setup_waypoints()
                self.current_waypoint_index = 0
                self.episode_start_time = time.time()
            except Exception as e:
                try:
                    current_location = self.vehicle.get_transform().location
                    waypoint = self.map.get_waypoint(current_location, project_to_road=True)
                    if waypoint is not None:
                        self.waypoints = [waypoint]
                        self.current_waypoint_index = 0
                    else:
                        return self.reset()
                except Exception as e2:
                    return self.reset()
        else:
            self.waypoints = []
            self.current_waypoint_index = 0
            self.episode_start_time = time.time()
        observation = self._get_observation()
        if self.vehicle is not None:
            self._wait_for_sensors()
        
        if self.debug_reset_count % 5 == 0:  # Log mỗi 5 resets
            console.log(f"[cyan]Reset {self.debug_reset_count} completed[/cyan]")
        
        return observation

    def _on_camera_update(self, image):
        """Handle camera updates"""
        try:
            # Convert to RGB
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]  # BGR to RGB
            
            # Store full resolution for rendering
            self.camera_image = array  # Full resolution for rendering
            self.last_rgb = array      # Lưu numpy array để monitor hiển thị đúng
            
        except Exception as e:
            console.log(f"[red]Error in camera update: {e}[/red]")
            self.camera_image = np.zeros((80, 160, 3), dtype=np.uint8)

    def _on_collision(self, event):
        """Handle collision events"""
        try:
            self.collision_history = True
            self.collision_counter += 1
            console.log(f"[yellow]Collision detected! Counter: {self.collision_counter}[/yellow]")
        except Exception as e:
            console.log(f"[red]Error in collision callback: {e}[/red]")
            import traceback
            console.log(f"[red]Collision traceback: {traceback.format_exc()}[/red]")

    def _on_lane_invasion(self, event):
        """Handle lane invasion events"""
        self.lane_invasion_history = True

    def _on_lidar_update(self, lidar_measurement):
        """Handle LIDAR updates"""
        try:
            points = np.frombuffer(lidar_measurement.raw_data, dtype=np.dtype('f4'))
            
            # Ensure the array length is divisible by 3
            if points.shape[0] % 3 != 0:
                # Truncate to nearest multiple of 3
                points = points[:-(points.shape[0] % 3)]
            
            if points.shape[0] > 0:
                points = np.reshape(points, (int(points.shape[0] / 3), 3))
                
                # Calculate minimum distance to obstacles
                distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
                self.lidar_min_distance = np.min(distances) if len(distances) > 0 else float('inf')
            else:
                self.lidar_min_distance = float('inf')
            
        except Exception as e:
            console.log(f"[red]Error in LIDAR update: {e}[/red]")
            self.lidar_min_distance = float('inf')

    def _on_segmentation_update(self, image):
        """Callback cho sensor segmentation"""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # BGR
            # Convert CARLA palette to RGB for visualization
            array = array[:, :, ::-1]
            self.last_segmentation = array
        except Exception as e:
            console.log(f"[red]Error in segmentation update: {e}[/red]")
            self.last_segmentation = np.zeros((80, 160, 3), dtype=np.uint8)

    def _on_depth_update(self, image):
        """Callback cho sensor depth"""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            # Depth sensor trả về grayscale, lấy channel đầu tiên
            depth_array = array[:, :, 0]
            # Normalize depth values (0-255) to (0-1)
            depth_normalized = depth_array.astype(np.float32) / 255.0
            # Convert to 3-channel for visualization
            self.last_depth = np.stack([depth_normalized] * 3, axis=-1)
        except Exception as e:
            console.log(f"[red]Error in depth update: {e}[/red]")
            self.last_depth = np.zeros((80, 160, 3), dtype=np.float32)

    # XÓA: def _on_third_person_camera_update(self, image): ...

    def setup_waypoints(self):
        """Setup waypoint navigation với curriculum learning"""
        try:
            # Get current location
            if self.vehicle is None:
                console.log(f"[red]Vehicle is None in setup_waypoints[/red]")
                return
                
            current_location = self.vehicle.get_transform().location
            waypoint = self.map.get_waypoint(current_location, project_to_road=True)
            
            if waypoint is None:
                console.log(f"[red]No waypoint found for current location[/red]")
                return
            
            # Curriculum Learning: Chọn total_distance theo mode và town
            if self.finetune_mode and self.episode_count > self.fixed_training_episodes:
                # Finetune mode: Random distance hoặc sử dụng config
                if self.town in self.map_configs:
                    config = self.map_configs[self.town]
                    total_distance = config["total_distance"]
                    route_logic = config["route_logic"]
                else:
                    total_distance = random.randint(200, 800)  # Random distance
                    route_logic = "forward"
                console.log(f"[yellow]Finetune mode: Total distance {total_distance}m[/yellow]")
            else:
                # Fixed mode: Sử dụng config cố định
                if self.town in self.map_configs:
                    config = self.map_configs[self.town]
                    total_distance = config["total_distance"]
                    route_logic = config["route_logic"]
                else:
                    total_distance = 780  # Default to Town02 distance
                    route_logic = "mixed"
                console.log(f"[green]Fixed mode: Total distance {total_distance}m for {self.town}[/green]")
            
            # Generate waypoints theo route logic (từ code PPO gốc)
            self.waypoints = [waypoint]  # Bắt đầu với waypoint hiện tại
            current_waypoint = waypoint
            
            for x in range(total_distance):
                try:
                    if route_logic == "mixed":
                        if self.town == "Town07":
                            if x < 650:
                                next_waypoint = current_waypoint.next(1.0)[0]  # Forward
                            else:
                                next_waypoint = current_waypoint.next(1.0)[-1]  # Backward
                        elif self.town == "Town02":
                            if x < 650:
                                next_waypoint = current_waypoint.next(1.0)[-1]  # Backward
                            else:
                                next_waypoint = current_waypoint.next(1.0)[0]  # Forward
                        else:
                            next_waypoint = current_waypoint.next(1.0)[0]  # Default forward
                    else:
                        # Forward logic
                        next_waypoint = current_waypoint.next(1.0)[0]
                    
                    self.waypoints.append(next_waypoint)
                    current_waypoint = next_waypoint
                except Exception as e:
                    console.log(f"[yellow]Error generating waypoint {x}: {e}[/yellow]")
                    break
                
            console.log(f"[green]Setup {len(self.waypoints)} waypoints for {self.town} (route_logic: {route_logic})[/green]")
            
        except Exception as e:
            console.log(f"[red]Error setting up waypoints: {e}[/red]")
            self.waypoints = []

    def get_distance_from_center(self):
        """Get distance from center of lane"""
        try:
            if self.vehicle is None or not hasattr(self, 'waypoints') or len(self.waypoints) == 0:
                return 0.0
                
            current_location = self.vehicle.get_transform().location
            current_waypoint = self.waypoints[self.current_waypoint_index]
            
            # Calculate distance from center
            dx = current_location.x - current_waypoint.transform.location.x
            dy = current_location.y - current_waypoint.transform.location.y
            distance = math.sqrt(dx**2 + dy**2)
            
            return distance
            
        except Exception as e:
            console.log(f"[red]Error getting distance from center: {e}[/red]")
            return 0.0

    def get_angle_to_waypoint(self):
        """Get angle to current waypoint"""
        try:
            if self.vehicle is None or not hasattr(self, 'waypoints') or len(self.waypoints) == 0:
                return 0.0
                
            current_transform = self.vehicle.get_transform()
            current_waypoint = self.waypoints[self.current_waypoint_index]
            
            # Calculate angle
            dx = current_waypoint.transform.location.x - current_transform.location.x
            dy = current_waypoint.transform.location.y - current_transform.location.y
            
            target_angle = math.degrees(math.atan2(dy, dx))
            current_angle = current_transform.rotation.yaw
            
            # Normalize angle difference
            angle_diff = target_angle - current_angle
            while angle_diff > 180:
                angle_diff -= 360
            while angle_diff < -180:
                angle_diff += 360
                
            return angle_diff
            
        except Exception as e:
            console.log(f"[red]Error getting angle to waypoint: {e}[/red]")
            return 0.0

    def update_waypoint_index(self):
        """Update current waypoint index"""
        try:
            if self.vehicle is None or not hasattr(self, 'waypoints') or len(self.waypoints) == 0:
                return
                
            current_location = self.vehicle.get_transform().location
            current_waypoint = self.waypoints[self.current_waypoint_index]
            
            # Check if we've reached the current waypoint
            dx = current_location.x - current_waypoint.transform.location.x
            dy = current_location.y - current_waypoint.transform.location.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < 5.0:  # Within 5 meters
                self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints)
                
        except Exception as e:
            console.log(f"[red]Error updating waypoint index: {e}[/red]")

    def vector(self, v):
        """Convert carla vector to numpy array"""
        return np.array([v.x, v.y, v.z])

    def distance_to_line(self, point_a, point_b, point_c):
        """Calculate distance from point_c to line defined by point_a and point_b"""
        a = point_a - point_b
        b = point_c - point_b
        return np.linalg.norm(np.cross(a, b)) / np.linalg.norm(a)

    def angle_diff(self, v1, v2):
        """Calculate angle difference between two vectors"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        # Check if either vector has zero norm
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)

    def _get_default_observation(self):
        """Get default observation when vehicle is None"""
        # Trả về numpy array với dtype phù hợp
        if self.device.type == "cuda":
            latent = np.zeros(95, dtype=np.float16)
            state = np.zeros(5, dtype=np.float16)
        else:
            latent = np.zeros(95, dtype=np.float32)
            state = np.zeros(5, dtype=np.float32)
        
        return {
            "latent": latent,
            "state": state
        }

    def _validate_observation(self, obs):
        if not isinstance(obs, dict):
            return False
        if "latent" not in obs or "state" not in obs:
            return False
        if obs["latent"].shape != (95,):
            return False
        if obs["state"].shape[0] < 5:
            return False
        return True

    def _get_observation(self):
        """Get current observation với VAE latent vector"""
        if self.vehicle is None:
            return self._get_default_observation()
        
        try:
            # Lấy camera image
            image = self.camera_image  # shape (H, W, 3), uint8 hoặc None
            if image is None:
                image = np.zeros((80, 160, 3), dtype=np.uint8)
            else:
                # Đảm bảo image là numpy array và đúng shape
                image = np.asarray(image)
                image = cv2.resize(image, (160, 80))  # Resize về (W, H)
            
            # Tối ưu: Chuẩn bị image trực tiếp trên GPU để tránh data transfer
            image = image.transpose(2, 0, 1)  # (C, H, W)
            image = image / 255.0
            
            # Tạo hash cho image để cache
            image_hash = hash(image.tobytes())
            
            # Kiểm tra cache trước
            if image_hash in self.vae_cache:
                latent = self.vae_cache[image_hash]
            else:
                # Tạo tensor theo device đã chọn
                if self.device.type == "cuda":
                    image_tensor = torch.tensor(image, dtype=torch.float16, device=self.device).unsqueeze(0)  # FP16 cho GPU
                else:
                    image_tensor = torch.tensor(image, dtype=torch.float32, device=self.device).unsqueeze(0)  # FP32 cho CPU

                # Encode qua VAE với tối ưu hóa
                with torch.no_grad():
                    if self.device.type == "cuda":
                        # Giữ latent trên GPU nếu có thể để tránh transfer
                        latent = self.vae(image_tensor).squeeze()  # shape (95,)
                        # Chỉ chuyển về CPU khi cần thiết
                        latent = latent.cpu().numpy()
                        
                        # Debug: Kiểm tra VAE device (chỉ log mỗi 100 steps)
                        if hasattr(self, 'debug_vae_count'):
                            self.debug_vae_count += 1
                        else:
                            self.debug_vae_count = 1
                        
                        if self.debug_vae_count % 100 == 0:
                            console.log(f"[cyan]VAE running on GPU: {next(self.vae.parameters()).device}[/cyan]")
                    else:
                        latent = self.vae(image_tensor).cpu().numpy().squeeze()  # shape (95,)
                
                # Cache kết quả
                self.vae_cache[image_hash] = latent
                
                # Giới hạn cache size
                if len(self.vae_cache) > self.vae_cache_size:
                    # Xóa cache cũ nhất
                    oldest_key = next(iter(self.vae_cache))
                    del self.vae_cache[oldest_key]

            # Lấy state observation
            velocity = self.vehicle.get_velocity()
            speed = np.linalg.norm([velocity.x, velocity.y, velocity.z]) * 3.6  # Convert to km/h
            
            # Get distance from center and angle
            distance_from_center = self.get_distance_from_center()
            angle_to_waypoint = self.get_angle_to_waypoint()
            
            # Normalize values
            target_speed = 22.0  # km/h
            max_distance_from_center = 3.0
            max_angle = np.deg2rad(20)
            
            normalized_velocity = speed / target_speed
            normalized_distance_from_center = distance_from_center / max_distance_from_center
            normalized_angle = angle_to_waypoint / max_angle
            
            # State vector: [throttle, velocity, normalized_velocity, normalized_distance_from_center, normalized_angle]
            state = np.array([
                getattr(self, 'throttle', 0.0),
                speed,
                normalized_velocity,
                normalized_distance_from_center,
                normalized_angle
            ], dtype=np.float32)
            
            # Trả về numpy array với dtype phù hợp
            if self.device.type == "cuda":
                latent = latent.astype(np.float16)
                state = state.astype(np.float16)
            else:
                latent = latent.astype(np.float32)
                state = state.astype(np.float32)
            
            return {
                "latent": latent,
                "state": state
            }
            
        except Exception as e:
            console.log(f"[red]Error getting observation: {e}[/red]")
            # Fallback observation để training tiếp tục
            return self._get_default_observation()

    def step(self, action):
        try:
            # Debug step
            if hasattr(self, 'debug_step_count'):
                self.debug_step_count += 1
            else:
                self.debug_step_count = 1
            
            if self.debug_step_count % 10 == 0:  # Log mỗi 10 steps
                console.log(f"[cyan]Step {self.debug_step_count}: action={action}[/cyan]")
            
            # Xử lý trường hợp vehicle None
            if self.vehicle is None:
                self.world.tick()
                return self._get_default_observation(), -0.1, False, {}

            # Tăng timestep counter
            self.timesteps += 1

            # Chỉ nhận steer, throttle
            steering, throttle = action[0], action[1]
            
            # Apply control với smoothing
            steer = max(min(steering, 1.0), -1.0)
            throttle = max(min(throttle, 1.0), 0.0)
            
            # Smoothing control như logic của họ
            if not hasattr(self, 'previous_steer'):
                self.previous_steer = 0.0
            if not hasattr(self, 'throttle'):
                self.throttle = 0.0
                
            control = carla.VehicleControl()
            control.throttle = float(self.throttle*0.9 + throttle*0.1)
            control.steer = float(self.previous_steer*0.9 + steer*0.1)
            self.vehicle.apply_control(control)
            self.previous_steer = steer
            self.throttle = throttle
            
            self.world.tick()
            
            # Render chỉ khi cần thiết để tránh slowdown
            if self.visualize and self.display is not None and self.render_counter % 10 == 0:  # Render mỗi 10 steps để giảm lắc
                self.render()
            self.render_counter += 1
            
            # Traffic Light state
            if self.vehicle.is_at_traffic_light():
                traffic_light = self.vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)

            # Velocity of the vehicle
            velocity = self.vehicle.get_velocity()
            speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            
            # Rotation of the vehicle in correlation to the map/lane
            rotation = self.vehicle.get_transform().rotation.yaw
            location = self.vehicle.get_location()

            # Keep track of closest waypoint on the route - logic của họ
            waypoint_index = self.current_waypoint_index
            for _ in range(len(self.waypoints)):
                next_waypoint_index = waypoint_index + 1
                wp = self.waypoints[next_waypoint_index % len(self.waypoints)]
                dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2], 
                           self.vector(location - wp.transform.location)[:2])
                if dot > 0.0:
                    waypoint_index += 1
                else:
                    break

            self.current_waypoint_index = waypoint_index
            
            # Cập nhật checkpoint nếu vượt qua checkpoint mới
            if self.current_waypoint_index >= self.checkpoint_waypoint_index + self.checkpoint_frequency:
                self.checkpoint_waypoint_index = self.current_waypoint_index
                console.log(f"[yellow]Checkpoint updated: {self.checkpoint_waypoint_index}[/yellow]")

            # Calculate deviation from center of the lane
            current_waypoint = self.waypoints[self.current_waypoint_index % len(self.waypoints)]
            next_waypoint = self.waypoints[(self.current_waypoint_index+1) % len(self.waypoints)]
            distance_from_center = self.distance_to_line(
                self.vector(current_waypoint.transform.location),
                self.vector(next_waypoint.transform.location),
                self.vector(location)
            )

            # Get angle difference between closest waypoint and vehicle forward vector
            fwd = self.vector(self.vehicle.get_velocity())
            wp_fwd = self.vector(current_waypoint.transform.rotation.get_forward_vector())
            angle = self.angle_diff(fwd, wp_fwd)
            
            # Collision detection - sử dụng trực tiếp
            collision = self.collision_history
            
            # Rewards calculation - GIỮ LẠI REWARD SHAPING TỐT HƠN CỦA CHÚNG TA
            done = False
            reward = 0

            if collision:
                try:
                    console.log(f"[red]Episode ending due to collision at step {self.timesteps}[/red]")
                    done = True
                    reward = -10
                except Exception as e:
                    console.log(f"[red]Error handling collision in step: {e}[/red]")
                    import traceback
                    console.log(f"[red]Collision step traceback: {traceback.format_exc()}[/red]")
                    done = True
                    reward = -10
            elif distance_from_center > 3.0:  # max_distance_from_center
                done = True
                reward = -10
            elif time.time() - self.episode_start_time > 10 and speed < 1.0:  # Giảm từ 120s xuống 10s
                done = True
                reward = -10
            elif speed > 30.0:  # max_speed
                done = True
                reward = -10

            # Interpolated from 1 when centered to 0 when 3m from center
            centering_factor = max(1.0 - distance_from_center / 3.0, 0.0)
            # Interpolated from 1 when aligned with the road to 0 when +/- 20 degrees of road
            angle_factor = max(1.0 - abs(angle) / np.deg2rad(20), 0.0)
            
            # Reward shaping tốt hơn của chúng ta
            if not done:
                # Thưởng chính dựa trên tốc độ
                if speed < 15.0:  # min_speed
                    reward = (speed / 15.0) * centering_factor * angle_factor    
                elif speed > 25.0:  # target_speed
                    reward = (1.0 - (speed-25.0) / (30.0-25.0)) * centering_factor * angle_factor  
                else:                                         
                    reward = 1.0 * centering_factor * angle_factor
                
                # Penalty nhẹ cho lane invasion - sử dụng trực tiếp
                lane_invasion = self.lane_invasion_history
                if lane_invasion:
                    reward -= 0.5

            # Episode termination conditions theo logic của bạn
            #if self.timesteps >= 7500:  # Thêm timestep limit
                #done = True
            if self.current_waypoint_index >= len(self.waypoints) - 2:
                done = True

            # Tăng episode count chỉ khi episode kết thúc
            if done:
                try:
                    self.episode_count += 1
                    console.log(f"[cyan]Episode {self.episode_count} ended. Reason: {'Collision' if collision else 'Other'}[/cyan]")
                    if self.episode_count % 10 == 0:  # Log mỗi 10 episodes
                        console.log(f"[green]Episode {self.episode_count} completed![/green]")
                except Exception as e:
                    console.log(f"[red]Error updating episode count: {e}[/red]")
                    import traceback
                    console.log(f"[red]Episode count traceback: {traceback.format_exc()}[/red]")

            # Cập nhật checkpoint_waypoint_index mỗi checkpoint_frequency waypoint
            if not getattr(self, 'fresh_start', True):
                if hasattr(self, 'checkpoint_frequency') and self.checkpoint_frequency is not None:
                    self.checkpoint_waypoint_index = (self.current_waypoint_index // self.checkpoint_frequency) * self.checkpoint_frequency

            # Khi done=True:
            #   - Nếu hoàn thành route: self.fresh_start = True
            #   - Nếu lỗi: self.fresh_start = False
            if done:
                if self.current_waypoint_index >= len(self.waypoints) - 2: # Hoàn thành route
                    self.fresh_start = True
                    self.current_waypoint_index = 0 # Reset về đầu route
                    self.checkpoint_waypoint_index = 0 # Reset checkpoint
                    console.log(f"[green]Episode {self.episode_count} completed successfully![green]")
                else: # Lỗi
                    self.fresh_start = False
                    self.current_waypoint_index = self.checkpoint_waypoint_index # Reset về checkpoint gần nhất
                    self.checkpoint_waypoint_index = self.current_waypoint_index # Cập nhật lại checkpoint
                    console.log(f"[red]Episode {self.episode_count} failed, resetting to checkpoint![red]")

            # Get observation
            observation = self._get_observation()
            
            # Tạo overlay cho render
            self._last_overlay = {
                'speed': speed,
                'target_speed': 22.0,
                'reward': reward,
                'distance_from_center': distance_from_center,
                'angle_to_waypoint': angle,
                'steer': steering,
                'throttle': throttle,
                'collision': collision,
                'lane_invasion': lane_invasion if 'lane_invasion' in locals() else False,
                'traffic_light': 'Green',
                'progress': self.current_waypoint_index / len(self.waypoints),
                'episode': self.episode_count,
                'done_reason': 'Normal' if not done else 'Episode End'
            }
            
            return observation, reward, done, {}

        except Exception as e:
            console.log(f"[red]Error in step: {e}[/red]")
            # Thay vì return done=True, chỉ return penalty nhỏ để training tiếp tục
            return self._get_default_observation(), -1.0, False, {}
    


    def render(self, mode='human'):
        """Render the environment with smooth visualization"""
        if not self.visualize or self.display is None:
            return

        try:
            # Thêm check để tránh crash khi có collision
            if hasattr(self, 'collision_history') and self.collision_history:
                # Nếu có collision, render đơn giản hơn
                try:
                    self.display.fill((255, 0, 0))  # Red background for collision
                    font = pygame.font.SysFont('Arial', 24)
                    text = font.render('COLLISION DETECTED', True, (255, 255, 255))
                    text_rect = text.get_rect(center=(self.display_width//2, self.display_height//2))
                    self.display.blit(text, text_rect)
                    pygame.display.flip()
                    return
                except Exception as e:
                    console.log(f"[red]Error in collision render: {e}[/red]")
                    return
            # Handle pygame events quickly
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    console.log("[yellow]User closed window, stopping training...[/yellow]")
                    self.close()
                    return

            # Render lại bằng self.camera_image như cũ
            if hasattr(self, 'camera_image') and self.camera_image is not None and self.camera_image.size > 0:
                import cv2
                import numpy as np
                cam_h, cam_w = self.camera_image.shape[:2]
                win_w, win_h = self.display_width, self.display_height
                scale = min(win_w / cam_w, win_h / cam_h)
                new_w, new_h = int(cam_w * scale), int(cam_h * scale)
                resized = cv2.resize(self.camera_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                x_offset = (win_w - new_w) // 2
                y_offset = (win_h - new_h) // 2
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                surface = pygame.surfarray.make_surface(canvas.swapaxes(0, 1))
                self.display.blit(surface, (0, 0))
            else:
                self.display.fill((0, 0, 0))
                try:
                    font = pygame.font.SysFont('Arial', 24)
                    text = font.render('Waiting for camera...', True, (255, 255, 255))
                    text_rect = text.get_rect(center=(self.display_width//2, self.display_height//2))
                    self.display.blit(text, text_rect)
                except Exception as e:
                    print(f"[DEBUG] Render fallback text error: {e}")

            # Overlay monitor cho camera sensors only
            small_w, small_h = 160, 120
            margin = 10
            idx = 0
            monitor_list = [
                ("RGB", getattr(self, 'last_rgb', None)),
                ("Depth", getattr(self, 'last_depth', None)),
                ("Seg", getattr(self, 'last_segmentation', None)),
            ]
            font = pygame.font.SysFont('Arial', 18)
            for name, img in monitor_list:
                # Tính toán vị trí góc trên bên phải
                x_pos = self.display_width - margin - small_w
                y_pos = margin + idx * (small_h + margin)
                # Ảnh sensor
                if isinstance(img, np.ndarray) and img.size > 0:
                    if img.ndim == 2:
                        img = np.stack([img]*3, axis=-1)
                    img_small = cv2.resize(img, (small_w, small_h))
                    surf = pygame.surfarray.make_surface(np.transpose(img_small, (1, 0, 2)))
                    self.display.blit(surf, (x_pos, y_pos))
                    # Vẽ tên sensor
                    txt_surface = font.render(name, True, (255,255,0))
                    self.display.blit(txt_surface, (x_pos+5, y_pos+5))
                    idx += 1
                # Collision/lane_invasion: bool
                elif isinstance(img, bool):
                    color = (255,0,0) if img else (0,255,0)
                    pygame.draw.rect(self.display, color, (x_pos, y_pos, small_w, small_h))
                    txt_surface = font.render(f"{name}: {'YES' if img else 'NO'}", True, (255,255,255))
                    self.display.blit(txt_surface, (x_pos+5, y_pos+5))
                    idx += 1
                # LIDAR: float
                elif isinstance(img, float):
                    pygame.draw.rect(self.display, (0,0,128), (x_pos, y_pos, small_w, small_h))
                    txt_surface = font.render(f"{name}: {img:.2f}m", True, (255,255,255))
                    self.display.blit(txt_surface, (x_pos+5, y_pos+5))
                    idx += 1
            self.render_counter += 1

            # Add overlay information
            overlay = getattr(self, '_last_overlay', None)
            if overlay:
                try:
                    font = pygame.font.SysFont('Arial', 20)
                    texts = [
                        f"Third Person View - HD",
                        f"Speed: {overlay['speed']:.1f} km/h",
                        f"Reward: {overlay['reward']:.2f}",
                        f"Steer: {overlay['steer']:.2f}"
                        f"Throttle: {overlay['throttle']:.2f}",
                        f"Distance: {overlay['distance_from_center']:.2f}m",
                        f"Angle: {overlay['angle_to_waypoint']:.1f}°",
                        f"Collision: {'YES' if overlay['collision'] else 'NO'}",
                        f"Lane Invasion: {'YES' if overlay['lane_invasion'] else 'NO'}",
                        f"Traffic Light: {overlay['traffic_light']}",
                        f"Progress: {overlay['progress']:.3f}",
                        f"Episode: {overlay['episode']}",
                        f"Status: {overlay['done_reason']}"
                    ]
                    for i, text in enumerate(texts):
                        color = (255, 255, 255)
                        txt_surface = font.render(text, True, color)
                        self.display.blit(txt_surface, (10, 10 + i*30))
                except Exception as e:
                    pass  # Ignore overlay errors

            pygame.display.flip()
            if self.clock is not None:
                self.clock.tick(60)  # Tăng lên 60 FPS
        except Exception as e:
            pass

    def close(self):
        """Clean up resources"""
        try:
            # Destroy sensors với delay để tránh conflict
            if hasattr(self, 'camera_sensor') and self.camera_sensor:
                try:
                    self.camera_sensor.destroy()
                    time.sleep(0.2)
                except:
                    pass
                self.camera_sensor = None
            if hasattr(self, 'collision_sensor') and self.collision_sensor:
                try:
                    self.collision_sensor.destroy()
                    time.sleep(0.2)
                except:
                    pass
                self.collision_sensor = None
            if hasattr(self, 'lane_invasion_sensor') and self.lane_invasion_sensor:
                try:
                    self.lane_invasion_sensor.destroy()
                    time.sleep(0.2)
                except:
                    pass
                self.lane_invasion_sensor = None
            if hasattr(self, 'lidar_sensor') and self.lidar_sensor:
                try:
                    self.lidar_sensor.destroy()
                    time.sleep(0.2)
                except:
                    pass
                self.lidar_sensor = None
            if hasattr(self, 'segmentation_sensor') and self.segmentation_sensor:
                try:
                    self.segmentation_sensor.destroy()
                    time.sleep(0.2)
                except:
                    pass
                self.segmentation_sensor = None
            if self.vehicle:
                try:
                    self.vehicle.destroy()
                    time.sleep(0.2)
                except:
                    pass
            for npc in self.npc_vehicles:
                try:
                    npc.destroy()
                    time.sleep(0.2)
                except:
                    pass
            self.npc_vehicles = []
            if self.camera:
                try:
                    self.camera.destroy()
                    time.sleep(0.2)
                except:
                    pass
            if self.visualize:
                pygame.quit()
        except Exception as e:
            console.log(f"[red]Error during cleanup: {e}[/red]")
