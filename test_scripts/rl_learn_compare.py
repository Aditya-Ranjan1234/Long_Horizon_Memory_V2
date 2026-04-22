
import os
import sys
import json
import numpy as np
import requests
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# Sync Client
class SyncHFClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def reset(self, episode_id=None):
        params = {"episode_id": str(episode_id)} if episode_id is not None else {}
        response = self.session.post(f"{self.base_url}/reset", params=params)
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
    def __init__(self, base_url, fixed_episode=None):
        super(HFEnvWrapper, self).__init__()
        self.client = SyncHFClient(base_url)
        self.fixed_episode = fixed_episode
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
        obs_data = self.client.reset(episode_id=self.fixed_episode)
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

def evaluate_agent(model, env, label):
    print(f"\n--- Evaluating {label} ---", flush=True)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0
    while not done and steps < 15:
        steps += 1
        # Use predict with deterministic=True
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        print(f"[{label}] Step {steps}: Action={['append', 'noop', 'rewrite'][action]} Reward={reward:.2f}", flush=True)
    return total_reward

def main():
    # Use a fixed episode for consistency
    fixed_ep_id = "1" 
    env = HFEnvWrapper(HF_SPACE_URL, fixed_episode=fixed_ep_id)
    
    # Initialize PPO with correct parameters from the start
    model = PPO("MlpPolicy", env, verbose=0, n_steps=20, batch_size=20, learning_rate=1e-3)
    
    # 1. Untrained Agent
    reward_before = evaluate_agent(model, env, "BEFORE LEARNING")
    
    # 2. Learning
    print("\n--- LEARNING PHASE (Training for 60 steps) ---", flush=True)
    model.learn(total_timesteps=60)
    
    # 3. Trained Agent
    reward_after = evaluate_agent(model, env, "AFTER LEARNING")
    
    print("\n" + "="*40)
    print("RL AGENT PERFORMANCE EVOLUTION")
    print(f"Reward Before Learning: {reward_before:.2f}")
    print(f"Reward After Learning:  {reward_after:.2f}")
    print(f"Improvement:           {reward_after - reward_before:+.2f}")
    print("="*40)

if __name__ == "__main__":
    main()
