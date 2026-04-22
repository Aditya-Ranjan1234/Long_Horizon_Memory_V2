
import os
import sys
import json
import requests
import time

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

# Configuration
HF_SPACE_URL = "https://aditya-ranjan1234-long-horizon-memory-v2.hf.space"
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "gemma:7b"

SYSTEM_PROMPT = """You are a memory compression agent.
Return JSON only with one of these shapes:
{"operation":"append"}
{"operation":"noop"}
{"operation":"rewrite","rewrite_memory":"..."}

Guidance:
- If new_message contains technical facts or important info, prefer append.
- If it's noisy or irrelevant, prefer noop.
- Use rewrite to compress memory when it grows too large.
"""

def ollama_chat(user_prompt):
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL_NAME,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {"temperature": 0.1},
            },
            timeout=45
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("message", {}).get("content", "")
    except Exception as e:
        print(f"Ollama Error: {e}")
        return '{"operation":"noop"}'

def run_ollama_on_hf():
    print(f"--- Running Ollama ({MODEL_NAME}) on HF Episode 1 ---")
    client = SyncHFClient(base_url=HF_SPACE_URL)
    obs_data = client.reset(episode_id="1")
    observation = obs_data.get("observation", {}) if "observation" in obs_data else obs_data
    
    total_reward = 0.0
    steps = 0
    done = False
    
    while not done and steps < 15:
        steps += 1
        user_prompt = f"Memory: {observation.get('memory', '')}\nNew message: {observation.get('new_message', '')}\nDecision:"
        
        response_text = ollama_chat(user_prompt)
        try:
            action = json.loads(response_text)
        except Exception as e:
            action = {"operation": "noop"}
            
        result = client.step(action)
        observation = result.get("observation", {})
        reward = float(result.get("reward", 0.0))
        done = bool(result.get("done", False))
        
        total_reward += reward
        print(f"[OLLAMA] Step {steps}: Action={action.get('operation')} Reward={reward:.2f}")
        
    print(f"\nOLLAMA FINAL REWARD: {total_reward:.2f}")

if __name__ == "__main__":
    run_ollama_on_hf()
