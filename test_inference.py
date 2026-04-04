#!/usr/bin/env python3
"""
Test script for inference.py using the HF Space endpoint.
"""

import os
import sys
import json
import requests
from typing import Dict, Any

# Set environment variables for testing
os.environ['API_BASE_URL'] = 'https://aditya-ranjan1234-long-horizon-memory-env.hf.space'
os.environ['MODEL_NAME'] = 'local'
os.environ['MAX_STEPS'] = '5'
os.environ['TASKS'] = 'easy'
os.environ['LLM_TIMEOUT_SECONDS'] = '30'
os.environ['MAX_MODEL_RETRIES'] = '1'
os.environ['BASELINE_SEED'] = '1337'
os.environ['RUN_ALL_EPISODES'] = 'false'

class SimpleAgent:
    """Simple rule-based agent for testing."""
    
    def __init__(self):
        self.step_count = 0
    
    def decide_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Make a simple decision based on observation."""
        self.step_count += 1
        
        # Simple heuristic: add first few messages, then noop
        if observation.get('memory_count', 0) < 3 and self.step_count <= 3:
            return {"action": {"operation": "add"}}
        elif observation.get('memory_count', 0) >= 5:
            return {"action": {"operation": "remove", "remove_index": 0}}
        else:
            return {"action": {"operation": "noop"}}

def test_episode_with_agent(base_url: str, agent: SimpleAgent) -> Dict[str, Any]:
    """Test a single episode with the agent."""
    
    # Reset environment
    reset_response = requests.post(f"{base_url}/reset", json={}, timeout=30)
    reset_response.raise_for_status()
    reset_data = reset_response.json()
    
    print(f"🎬 Starting Episode")
    print(f"   Domain: {reset_data['observation'].get('domain', 'N/A')}")
    print(f"   Task: {reset_data['observation'].get('task_name', 'N/A')}")
    print(f"   Initial message: {reset_data['observation'].get('new_message', 'N/A')[:60]}...")
    
    total_reward = 0.0
    steps = 0
    max_steps = 5
    
    observation = reset_data['observation']
    
    for step in range(max_steps):
        print(f"\n📍 Step {step + 1}")
        print(f"   Message: {observation.get('new_message', 'N/A')[:60]}...")
        print(f"   Current memory: {observation.get('memory_count', 0)} items")
        
        # Agent makes decision
        action = agent.decide_action(observation)
        print(f"   Action: {action['action']['operation']}")
        
        # Execute action
        step_response = requests.post(f"{base_url}/step", json=action, timeout=30)
        step_response.raise_for_status()
        step_data = step_response.json()
        
        observation = step_data['observation']
        reward = step_data.get('reward', 0.0)
        total_reward += reward
        steps += 1
        
        print(f"   Reward: {reward:.3f}")
        print(f"   Total reward: {total_reward:.3f}")
        print(f"   Memory after action: {observation.get('memory_count', 0)} items")
        
        if step_data.get('done', False):
            print(f"   🏁 Episode completed after {step + 1} steps!")
            break
    
    # Get final state
    state_response = requests.get(f"{base_url}/state", timeout=30)
    state_response.raise_for_status()
    state_data = state_response.json()
    
    return {
        'steps': steps,
        'total_reward': total_reward,
        'avg_reward': total_reward / steps if steps > 0 else 0,
        'final_memory_count': observation.get('memory_count', 0),
        'done': step_data.get('done', False),
        'final_state': state_data
    }

def main():
    """Main test function."""
    print("🧪 Testing Inference with HF Space")
    print("=" * 50)
    
    base_url = "https://aditya-ranjan1234-long-horizon-memory-env.hf.space"
    agent = SimpleAgent()
    
    try:
        # Test health first
        print("🔍 Checking HF Space health...")
        health_response = requests.get(f"{base_url}/health", timeout=10)
        health_response.raise_for_status()
        print(f"✅ Health check: {health_response.json()}")
        
        # Test a few episodes
        episodes_to_test = 3
        
        for episode in range(episodes_to_test):
            print(f"\n{'='*50}")
            print(f"🎮 Episode {episode + 1}/{episodes_to_test}")
            print(f"{'='*50}")
            
            try:
                result = test_episode_with_agent(base_url, agent)
                
                print(f"\n📊 Episode Results:")
                print(f"   Steps taken: {result['steps']}")
                print(f"   Total reward: {result['total_reward']:.3f}")
                print(f"   Average reward: {result['avg_reward']:.3f}")
                print(f"   Final memory count: {result['final_memory_count']}")
                print(f"   Episode completed: {result['done']}")
                
                # Simple success criteria
                if result['avg_reward'] > 0.3:
                    print("   ✅ Good performance!")
                else:
                    print("   ⚠️ Could be better")
                
            except Exception as e:
                print(f"❌ Episode {episode + 1} failed: {e}")
                continue
        
        print(f"\n{'='*50}")
        print("🎉 Inference testing completed!")
        print("🌐 Web interface: https://aditya-ranjan1234-long-horizon-memory-env.hf.space")
        print("📚 API docs: https://aditya-ranjan1234-long-horizon-memory-env.hf.space/docs")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
