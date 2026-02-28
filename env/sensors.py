import math
import numpy as np
import weakref
import pygame
import carla

# CameraSensor - BỎ DEBUG LOGS
class CameraSensor():
    def __init__(self, vehicle):
        self.sensor_name = 'sensor.camera.rgb'
        self.parent = vehicle
        self.front_camera = list()
        world = self.parent.get_world()
        
        self.sensor = self._set_camera_sensor(world)
        
        if self.sensor is None:
            from rich.console import Console
            console = Console()
            console.log(f"[red]Failed to spawn {self.sensor_name}[/red]")
            return
            
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraSensor._get_front_camera_data(weak_self, image))

    def _set_camera_sensor(self, world):
        try:
            front_camera_bp = world.get_blueprint_library().find(self.sensor_name)
            if front_camera_bp is None:
                from rich.console import Console
                console = Console()
                console.log(f"[red]Blueprint {self.sensor_name} not found![/red]")
                return None
                
            front_camera_bp.set_attribute('image_size_x', f'160')
            front_camera_bp.set_attribute('image_size_y', f'80')
            front_camera_bp.set_attribute('fov', f'125')
            
            front_camera = world.spawn_actor(front_camera_bp, carla.Transform(
                carla.Location(x=2.4, z=1.5), carla.Rotation(pitch= -10)), attach_to=self.parent)
            
            return front_camera
        except Exception as e:
            from rich.console import Console
            console = Console()
            console.log(f"[red]Error spawning camera sensor: {e}[/red]")
            return None

    @staticmethod
    def _get_front_camera_data(weak_self, image):
        self = weak_self()
        if not self:
            return
        
        # Bỏ image.convert() vì RGB camera không cần convert
        placeholder = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        placeholder1 = placeholder.reshape((image.height, image.width, 4))
        target = placeholder1[:, :, :3]
        self.front_camera.append(target)

    def get(self) -> np.ndarray:
        if len(self.front_camera) > 0:
            return self.front_camera[-1]
        return np.zeros((80, 160, 3), dtype=np.uint8)

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()

# CollisionSensor giống tác giả
class CollisionSensor:
    def __init__(self, vehicle) -> None:
        self.sensor_name = 'sensor.other.collision'
        self.parent = vehicle
        self.collision_data = list()
        world = self.parent.get_world()
        self.sensor = self._set_collision_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    def _set_collision_sensor(self, world) -> object:
        collision_sensor_bp = world.get_blueprint_library().find(self.sensor_name)
        sensor_relative_transform = carla.Transform(
            carla.Location(x=1.3, z=0.5))
        collision_sensor = world.spawn_actor(
            collision_sensor_bp, sensor_relative_transform, attach_to=self.parent)
        return collision_sensor

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collision_data.append(intensity)

    def get(self) -> list:
        return self.collision_data

    def reset(self):
        self.collision_data = []

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()

# LaneInvasionSensor giống tác giả
class LaneInvasionSensor:
    def __init__(self, vehicle):
        self.sensor_name = 'sensor.other.lane_invasion'
        self.parent = vehicle
        self.history = False
        world = self.parent.get_world()
        self.sensor = world.spawn_actor(world.get_blueprint_library().find(self.sensor_name), carla.Transform(), attach_to=self.parent)
        self.sensor.listen(self._on_invasion)

    def _on_invasion(self, event):
        self.history = True

    def get(self) -> bool:
        return self.history

    def reset(self):
        self.history = False

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()