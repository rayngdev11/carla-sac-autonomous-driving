import os, random, math
import numpy as np
import carla
import pygame
import cv2
from gym import spaces
import gym
from .sensors import CameraSensor, CollisionSensor, LaneInvasionSensor

class CarlaEnv(gym.Env):
    def __init__(
        self,
        num_npcs: int = 5,
        frame_skip: int = 8,
        visualize: bool = True,
        fixed_delta_seconds: float = 0.05,
        camera_width: int = 84,
        camera_height: int = 84,
        safe_dist: float = 5.0,
        obstacle_weight: float = 2.0,
        lane_weight: float = 2.0,
        yaw_weight: float = 1.0
    ):
        super().__init__()
        self.num_npcs = num_npcs
        self.frame_skip = frame_skip
        self.visualize = visualize
        self.safe_dist = safe_dist
        self.obstacle_weight = obstacle_weight
        self.lane_weight = lane_weight
        self.yaw_weight = yaw_weight
        self.camera_width = camera_width
        self.camera_height = camera_height

        # SDL / Pygame setup
        if visualize:
            os.environ.pop('SDL_VIDEODRIVER', None)
            pygame.init()
            self.display = pygame.display.set_mode((1280, 768))
            pygame.display.set_caption("Driver's View")
            self.clock = pygame.time.Clock()
        else:
            os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
            self.display = None
            self.clock = None

        # Connect to CARLA
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta_seconds
        self.world.apply_settings(settings)

        # Observation and action spaces
        self.observation_space = spaces.Dict({
            'image': spaces.Box(0, 255, (160, 80, 3), dtype=np.uint8),  # Fixed size for VAE
            'state': spaces.Box(-np.inf, np.inf, (5,), dtype=np.float32)
        })
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Internal state
        self.vehicle = None
        self.npc_vehicles = []
        self.stuck_counter = 0
        self.idle_penalty = 0.0

        # Sensors placeholders
        self.camera_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None

        self.episode_count = 0
        self.previous_location = None
        self.previous_speed = 0.0

    def _get_lane_center(self, location: carla.Location) -> carla.Location:
        waypoint = self.world.get_map().get_waypoint(location)
        return waypoint.transform.location

    def _get_lane_yaw(self, location: carla.Location) -> float:
        waypoint = self.world.get_map().get_waypoint(location)
        return waypoint.transform.rotation.yaw

    def _compute_lane_deviation(self, location: carla.Location) -> float:
        center = self._get_lane_center(location)
        return math.hypot(location.x - center.x, location.y - center.y)

    def _compute_yaw_penalty(self, yaw: float, location: carla.Location) -> float:
        lane_yaw = self._get_lane_yaw(location)
        diff = abs(yaw - lane_yaw) % 360
        return min(diff, 360 - diff) / 180.0

    def spawn_npcs(self):
        try:
            blueprints = self.world.get_blueprint_library().filter('vehicle.*')
            spawn_points = self.world.get_map().get_spawn_points()
            random.shuffle(spawn_points)
            for bp, sp in zip(random.choices(blueprints, k=self.num_npcs), spawn_points):
                npc = self.world.try_spawn_actor(bp, sp)
                if npc:
                    npc.set_autopilot(True)
                    self.npc_vehicles.append(npc)
        except Exception as e:
            print(f"Error spawning NPCs: {e}")

    def reset(self, seed=None, options=None):
        # Cleanup
        if self.vehicle:
            self.vehicle.destroy()
        for npc in self.npc_vehicles:
            npc.destroy()
        self.npc_vehicles.clear()
        for sensor in [self.camera_sensor, self.collision_sensor,
                       self.lane_invasion_sensor]:
            if sensor:
                sensor.destroy()

        # Spawn vehicle with error handling
        try:
            map_spawns = self.world.get_map().get_spawn_points()
            if not map_spawns:
                raise RuntimeError("No spawn points available")
            
            blueprints = self.world.get_blueprint_library().filter('vehicle.*')
            if not blueprints:
                raise RuntimeError("No vehicle blueprints available")
            
            self.vehicle = self.world.try_spawn_actor(
                random.choice(blueprints),
                random.choice(map_spawns)
            )
            
            if self.vehicle is None:
                raise RuntimeError("Failed to spawn vehicle")
                
        except Exception as e:
            print(f"Error spawning vehicle: {e}")
            # Return default observation
            return {
                'image': np.zeros((80, 160, 3), dtype=np.uint8),
                'state': np.zeros(5, dtype=np.float32)
            }

        self.spawn_npcs()

        # Attach sensors with error handling
        try:
            self.camera_sensor = CameraSensor(self.world, self.vehicle)
            self.collision_sensor = CollisionSensor(self.world, self.vehicle)
            self.lane_invasion_sensor = LaneInvasionSensor(self.world, self.vehicle)
        except Exception as e:
            print(f"Error attaching sensors: {e}")

        # Warm-up ticks
        for _ in range(5):
            self.world.tick()

        self.previous_location = self.vehicle.get_transform().location
        self.previous_speed = 0.0
        self.episode_count += 1
        
        # Reset collision data
        if self.collision_sensor:
            self.collision_sensor.reset()

        obs, _, _, _ = self.step([0.0, 0.0, 0.0])
        return obs

    def step(self, action):
        # Validate vehicle exists
        if self.vehicle is None:
            print("Warning: Vehicle is None, returning default observation")
            return {
                'image': np.zeros((80, 160, 3), dtype=np.uint8),
                'state': np.zeros(5, dtype=np.float32)
            }, -10, True, {"vehicle_none": True}

        steer, throttle, brake = action
        control = carla.VehicleControl(
            steer=float(steer), throttle=float(throttle), brake=float(brake)
        )
        
        for _ in range(self.frame_skip):
            self.vehicle.apply_control(control)
            self.world.tick()
            
        # Gather observation with error handling
        try:
            transform = self.vehicle.get_transform()
            loc = transform.location
            yaw = transform.rotation.yaw
            vel = self.vehicle.get_velocity()
            speed = math.hypot(vel.x, vel.y, vel.z)
            
            # Safe sensor data retrieval
            image = self.camera_sensor.get() if self.camera_sensor else np.zeros((80, 160, 3), dtype=np.uint8)
            
            # State vector: [x, y, speed, yaw, lane_deviation]
            lane_dev = self._compute_lane_deviation(loc)
            state = np.array([loc.x, loc.y, speed, yaw, lane_dev], dtype=np.float32)
        except Exception as e:
            print(f"Error gathering observation: {e}")
            return {
                'image': np.zeros((80, 160, 3), dtype=np.uint8),
                'state': np.zeros(5, dtype=np.float32)
            }, -10, True, {"observation_error": True}

        # Reward components
        # distance moved
        dx = loc.x - self.previous_location.x
        dy = loc.y - self.previous_location.y
        distance_reward = math.hypot(dx, dy)
        
        # speed target
        target_speed = 11.5
        speed_reward = math.exp(-((speed - target_speed)**2) / (2 * 3.0**2))
        
        # penalties
        lane_dev = self._compute_lane_deviation(loc)
        yaw_pen = self._compute_yaw_penalty(yaw, loc)
        
        # collision penalty
        collision_data = self.collision_sensor.get() if self.collision_sensor else []
        collision_penalty = -10.0 if len(collision_data) > 0 else 0.0
        
        # --- SAFE DRIVING REWARD ---
        # Traffic light state
        traffic_light = self.vehicle.get_traffic_light() if self.vehicle else None
        red_light = False
        if traffic_light is not None:
            red_light = traffic_light.get_state() == carla.TrafficLightState.Red
        # Vượt đèn đỏ
        run_red_light = False
        if red_light and speed > 1.0:
            run_red_light = True
        # Dừng đúng đèn đỏ
        stop_at_red = red_light and speed < 0.5
        # Vượt tốc độ
        speed_limit = self.vehicle.get_speed_limit() if self.vehicle else 30.0
        over_speed = speed > speed_limit + 2.0
        # Reward shaping
        red_light_penalty = -10.0 if run_red_light else 0.0
        stop_red_reward = 5.0 if stop_at_red else 0.0
        over_speed_penalty = -5.0 if over_speed else 0.0
        # final reward
        reward = (distance_reward + speed_reward)
        reward = reward - self.lane_weight * lane_dev - self.yaw_weight * yaw_pen + collision_penalty
        reward = reward + red_light_penalty + stop_red_reward + over_speed_penalty
        # Done conditions
        done = False
        collision_data = self.collision_sensor.get() if self.collision_sensor else []
        if len(collision_data) > 0:
            done = True
        elif speed < 0.1 and distance_reward < 0.01:
            done = True
        self.previous_location = loc
        self.previous_speed = speed
        # Lưu thông tin overlay cho render
        self._last_overlay = {
            'speed': speed,
            'reward': reward,
            'steer': steer,
            'throttle': throttle,
            'brake': brake,
            'red_light': red_light,
            'run_red_light': run_red_light,
            'stop_at_red': stop_at_red,
            'over_speed': over_speed
        }
        return {'image': image, 'state': state}, reward, done, {}

    def render(self, mode='human'):
        if not self.visualize or self.display is None:
            return
        try:
            frame = self.camera_sensor.get() if self.camera_sensor else np.zeros((80, 160, 3), dtype=np.uint8)
            # Resize frame to fit display
            frame_resized = cv2.resize(frame, (1280, 640))
            surface = pygame.surfarray.make_surface(frame_resized.swapaxes(0,1))
            self.display.blit(surface, (0,0))
            # Overlay thông tin
            overlay = getattr(self, '_last_overlay', None)
            if overlay:
                font = pygame.font.SysFont('Arial', 20)
                texts = [
                    f"Speed: {overlay['speed']:.2f} km/h",
                    f"Reward: {overlay['reward']:.2f}",
                    f"Steer: {overlay['steer']:.2f}",
                    f"Throttle: {overlay['throttle']:.2f}",
                    f"Brake: {overlay['brake']:.2f}",
                    f"Red Light: {'YES' if overlay['red_light'] else 'NO'}",
                    f"Run Red Light: {'YES' if overlay['run_red_light'] else 'NO'}",
                    f"Stop at Red: {'YES' if overlay['stop_at_red'] else 'NO'}",
                    f"Over Speed: {'YES' if overlay['over_speed'] else 'NO'}"
                ]
                for i, text in enumerate(texts):
                    txt_surface = font.render(text, True, (255,0,0) if 'YES' in text else (255,255,255))
                    self.display.blit(txt_surface, (10, 10 + i*25))
            pygame.display.flip()
            if self.clock:
                self.clock.tick(60)
        except Exception as e:
            print(f"Error during rendering: {e}")

    def close(self):
        try:
            for npc in self.npc_vehicles:
                npc.destroy()
            if self.vehicle:
                self.vehicle.destroy()
            for sensor in [self.camera_sensor, self.collision_sensor,
                           self.lane_invasion_sensor]:
                if sensor:
                    sensor.destroy()
            if self.visualize:
                pygame.quit()
        except Exception as e:
            print(f"Error during close: {e}")