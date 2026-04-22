
import requests
import json

base_url = "https://aditya-ranjan1234-long-horizon-memory-v2.hf.space"

def test_connection():
    try:
        print(f"Testing connection to {base_url}...")
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Health check: {response.status_code} - {response.text}")
        
        response = requests.post(f"{base_url}/reset", timeout=10)
        print(f"Reset check: {response.status_code}")
        if response.status_code == 200:
            print("Successfully connected to the environment!")
            return True
        else:
            print(f"Failed to reset: {response.text}")
            return False
    except Exception as e:
        print(f"Error connecting: {e}")
        return False

if __name__ == "__main__":
    test_connection()
