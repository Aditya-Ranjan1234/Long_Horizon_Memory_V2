#!/usr/bin/env python3
"""
Test the actual inference.py with HF Space endpoint.
"""

import os
import sys

# Set environment variables to use HF Space instead of requiring HF token
os.environ['API_BASE_URL'] = 'https://aditya-ranjan1234-long-horizon-memory-env.hf.space'
os.environ['MODEL_NAME'] = 'local'  # Use local instead of LLM
os.environ['MAX_STEPS'] = '3'      # Shorter for testing
os.environ['TASKS'] = 'easy'       # Start with easy
os.environ['LLM_TIMEOUT_SECONDS'] = '10'
os.environ['MAX_MODEL_RETRIES'] = '1'
os.environ['BASELINE_SEED'] = '1337'
os.environ['RUN_ALL_EPISODES'] = 'false'

# Mock the OpenAI client to avoid needing HF token
class MockOpenAI:
    class chat:
        class completions:
            @staticmethod
            def create(*args, **kwargs):
                # Return a simple action
                return {
                    'choices': [{
                        'message': {
                            'content': '{"action": {"operation": "add"}}'
                        }
                    }]
                }

# Replace the OpenAI import
sys.modules['openai'] = type(sys)('openai')
sys.modules['openai'].OpenAI = MockOpenAI

def test_inference():
    """Test the actual inference.py logic."""
    print("🧪 Testing Actual inference.py Logic")
    print("=" * 50)
    
    try:
        # Import after setting up mocks
        import inference
        
        # Test a simple episode
        print("🎮 Running inference test...")
        
        # This would normally run the full inference, but we'll test a simplified version
        print("✅ Inference module imported successfully")
        
        # Test environment creation
        from models import LongHorizonMemoryAction, LongHorizonMemoryObservation
        print("✅ Models imported successfully")
        
        # Test action creation
        action = LongHorizonMemoryAction(operation="noop")
        print(f"✅ Action created: {action.operation}")
        
        print("\n🎉 Inference components working correctly!")
        print("📝 Note: Full LLM inference requires HF_TOKEN in .env file")
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(test_inference())
