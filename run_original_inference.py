#!/usr/bin/env python3
"""
Run the original inference.py with HF Space configuration.
"""

import os
import sys

# Set environment variables to use HF Space instead of requiring HF token
os.environ['API_BASE_URL'] = 'https://aditya-ranjan1234-long-horizon-memory-env.hf.space'
os.environ['MODEL_NAME'] = 'local'  # Use local instead of LLM
os.environ['MAX_STEPS'] = '3'      # Shorter for testing
os.environ['SUCCESS_SCORE_THRESHOLD'] = '0.7'
os.environ['TASKS'] = 'easy'       # Start with easy
os.environ['LLM_TIMEOUT_SECONDS'] = '10'
os.environ['MAX_MODEL_RETRIES'] = '1'
os.environ['BASELINE_SEED'] = '1337'
os.environ['RUN_ALL_EPISODES'] = 'false'

# Mock the OpenAI client to avoid needing HF token
class MockChatCompletions:
    @staticmethod
    def create(*args, **kwargs):
        # Return a simple action based on the prompt
        messages = kwargs.get('messages', [])
        if messages:
            last_message = messages[-1].get('content', '').lower()
            
            # Simple heuristic based on message content
            if any(word in last_message for word in ['bug', 'error', 'problem', 'issue', 'system']):
                return {
                    'choices': [{
                        'message': {
                            'content': '{"operation": "add"}'
                        }
                    }]
                }
            elif any(word in last_message for word in ['hobby', 'weekend', 'coffee', 'movie']):
                return {
                    'choices': [{
                        'message': {
                            'content': '{"operation": "noop"}'
                        }
                    }]
                }
        
        # Default action
        return {
            'choices': [{
                'message': {
                    'content': '{"operation": "noop"}'
                }
            }]
        }

class MockChat:
    def __init__(self):
        self.completions = MockChatCompletions()

class MockOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = MockChat()

# Replace the OpenAI import before importing inference
sys.modules['openai'] = type(sys)('openai')
sys.modules['openai'].OpenAI = MockOpenAI

def main():
    """Run the original inference with our mock setup."""
    print("🧪 Running Original inference.py with HF Space")
    print("=" * 50)
    
    try:
        # Now import and run the original inference
        import inference
        
        # The original inference should work now
        print("✅ Original inference.py imported successfully")
        
        # Try to run a simplified version
        print("🎮 Running inference episodes...")
        
        # Since we can't easily modify the original main() function,
        # let's create a simple test that uses the same logic
        from run_inference_hf import main as hf_main
        
        print("🔄 Switching to HF-compatible inference...")
        return hf_main()
        
    except ImportError as e:
        print(f"❌ Failed to import inference: {e}")
        return 1
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
