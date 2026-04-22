
import os
import sys
import json
import numpy as np
import requests
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# Robust synchronous client for RL training
class SyncHFClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def reset(self):
        response = self.session.post(f"{self.base_url}/reset")
        response.raise_for_status()
        return response.json()

    def step(self, action):
        payload = {
            "action": {
                "operation": action.get("operation", "noop"),
                "rewrite_memory": action.get("rewrite_memory", None)
            }
        }
        response = self.session.post(f"{self.base_url}/step", json=payload)
        response.raise_for_status()
        return response.json()

HF_SPACE_URL = "https://aditya-ranjan1234-long-horizon-memory-v2.hf.space"

class HFEnvWrapper(gym.Env):
    def __init__(self, base_url):
        super(HFEnvWrapper, self).__init__()
        self.client = SyncHFClient(base_url)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=200, shape=(3,), dtype=np.float32)
        self.step_count = 0
        self.keywords = ["bug", "error", "api", "system", "logic", "fix", "issue", "database"]
        self.current_memory = ""

    def _get_obs(self, obs_data):
        observation = obs_data.get("observation", {}) if "observation" in obs_data else obs_data
        text = observation.get("new_message", "").lower()
        rel_score = sum(1 for k in self.keywords if k in text)
        self.current_memory = observation.get("memory", "")
        return np.array([rel_score, observation.get("memory_count", 0), self.step_count], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_data = self.client.reset()
        self.step_count = 0
        return self._get_obs(obs_data), {}

    def step(self, action_idx):
        self.step_count += 1
        ops = ["append", "noop", "rewrite"]
        op = ops[action_idx]
        action = {"operation": op, "rewrite_memory": None}
        if op == "rewrite":
            lines = self.current_memory.splitlines()
            action["rewrite_memory"] = "\n".join(lines[-5:]) if lines else ""
        
        result = self.client.step(action)
        obs_data = result.get("observation", {})
        reward = float(result.get("reward", 0.0))
        done = bool(result.get("done", False))
        
        return self._get_obs(obs_data), reward, done, False, {}

def train_and_eval():
    print("--- Starting RL Training on HF Environment (Fast Mode) ---", flush=True)
    env = HFEnvWrapper(HF_SPACE_URL)
    
    # Configure PPO for very small rollout to work with remote API latency
    model = PPO("MlpPolicy", env, verbose=1, n_steps=20, batch_size=20, learning_rate=3e-4)
    
    print("Training for 40 steps...", flush=True)
    model.learn(total_timesteps=40)
    
    print("\n--- Evaluating RL Agent ---", flush=True)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0
    while not done and steps < 10:
        steps += 1
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        print(f"Step {steps}: Action={['append', 'noop', 'rewrite'][action]} Reward={reward:.2f}", flush=True)
    print(f"Total RL Reward: {total_reward:.2f}", flush=True)

if __name__ == "__main__":
    train_and_eval()
