from abc import ABC, abstractmethod
from typing import Any
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

from agent import Agent
from config import visualizer_path, video_record_period


class Trainer(ABC):
    def __init__(
        self,
        agent: Agent,
        env: gym.Env[Any, Any],
        n_steps: int,
        discount_factor: float = 0.99,
        video_path: str = visualizer_path,
        record_period: int = video_record_period,
    ):
        self.agent = agent
        record_video_env = RecordVideo(
            env=env,
            video_folder=video_path + "/video_training",
            episode_trigger=lambda x: x % record_period == 0,
        )
        self.env = RecordEpisodeStatistics(record_video_env)
        self.n_steps = n_steps
        self.discount_factor = discount_factor

    @abstractmethod
    def train(self) -> Any:
        pass
