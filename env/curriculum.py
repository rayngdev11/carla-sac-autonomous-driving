import gym
import numpy as np
from typing import Dict, Any

class CurriculumEnv(gym.Wrapper):
    """
    Curriculum Learning wrapper cho CarlaEnv
    Tăng dần độ khó dựa trên performance của agent
    """
    def __init__(
        self, 
        env, 
        max_level: int = 3, 
        npc_increment: int = 2,
        success_threshold: float = 0.7,
        level_up_episodes: int = 10
    ):
        super().__init__(env)
        self.level = 0
        self.max_level = max_level
        self.npc_increment = npc_increment
        self.success_threshold = success_threshold
        self.level_up_episodes = level_up_episodes
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.collision_count = 0
        self.total_episodes = 0
        
    def reset(self, **kwargs):
        # Remove seed argument for gym compatibility
        if 'seed' in kwargs:
            del kwargs['seed']
            
        # Tăng dần số NPC theo level
        base_npcs = 5  # Số NPC cơ bản
        new_npcs = min(
            base_npcs + self.level * self.npc_increment,
            base_npcs + self.max_level * self.npc_increment
        )
        
        # Cập nhật số NPC trong environment
        if hasattr(self.env, 'num_npcs'):
            self.env.num_npcs = new_npcs
        
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Track episode performance
        if done:
            self.total_episodes += 1
            self.episode_rewards.append(info.get('episode', {}).get('r', 0))
            self.episode_lengths.append(info.get('episode', {}).get('l', 0))
            
            # Check for collision
            if info.get('collision', False):
                self.collision_count += 1
            
            # Keep only recent episodes for level evaluation
            if len(self.episode_rewards) > self.level_up_episodes:
                self.episode_rewards = self.episode_rewards[-self.level_up_episodes:]
                self.episode_lengths = self.episode_lengths[-self.level_up_episodes:]
            
            # Evaluate if should level up
            self._evaluate_level_up()
        
        return obs, reward, done, info
    
    def _evaluate_level_up(self):
        """Đánh giá xem có nên tăng level không"""
        if self.level >= self.max_level:
            return
        
        if len(self.episode_rewards) < self.level_up_episodes:
            return
        
        # Tính success rate dựa trên reward và collision
        recent_rewards = np.array(self.episode_rewards[-self.level_up_episodes:])
        recent_lengths = np.array(self.episode_lengths[-self.level_up_episodes:])
        
        # Success criteria: high rewards, reasonable episode length, low collision rate
        avg_reward = np.mean(recent_rewards)
        avg_length = np.mean(recent_lengths)
        collision_rate = self.collision_count / self.total_episodes
        
        # Normalize reward (assuming good performance > 0)
        normalized_reward = max(0, avg_reward) / 100.0  # Adjust threshold as needed
        
        # Success score
        success_score = (
            normalized_reward * 0.6 +  # Reward weight
            (1 - collision_rate) * 0.3 +  # Safety weight
            min(1.0, avg_length / 100) * 0.1  # Completion weight
        )
        
        if success_score > self.success_threshold:
            self.level += 1
            print(f"🎓 Curriculum: Nâng độ khó lên level {self.level}")
            print(f"   Success Score: {success_score:.3f}")
            print(f"   Avg Reward: {avg_reward:.2f}")
            print(f"   Collision Rate: {collision_rate:.3f}")
            
            # Reset collision count for next level
            self.collision_count = 0
    
    def get_curriculum_info(self) -> Dict[str, Any]:
        """Trả về thông tin về curriculum hiện tại"""
        return {
            "level": self.level,
            "max_level": self.max_level,
            "total_episodes": self.total_episodes,
            "recent_avg_reward": np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0,
            "collision_rate": self.collision_count / max(1, self.total_episodes),
            "npc_count": 5 + self.level * self.npc_increment
        } 