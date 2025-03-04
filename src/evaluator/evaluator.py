from typing import Any

import gymnasium as gym
import numpy as np
import torch
from tqdm.auto import tqdm

from agent import Agent
from config import device


class Evaluator:
    def __init__(self, agent: Agent, env: gym.Env) -> None:
        self.agent = agent
        self.env = env

    def evaluate_agent(self, num_times: int) -> tuple[np.floating[Any], list[float]]:
        rewards = []
        for _ in tqdm(range(num_times)):
            rewards.append(self.play_game())
        median_reward = np.median(rewards)
        return median_reward, rewards

    def play_game(self) -> float:
        state, _ = self.env.reset()
        done = False
        total_reward = 0.0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            action, _ = self.agent.get_action(observation=state_tensor)
            next_state, reward, terminated, truncated, _ = self.env.step(
                action.cpu().item()
            )
            total_reward += reward
            done = terminated or truncated
            state = next_state
        return total_reward
