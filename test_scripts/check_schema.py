
import requests
import json

base_url = "https://aditya-ranjan1234-long-horizon-memory-v2.hf.space"

def check_schema():
    try:
        response = requests.get(f"{base_url}/schema", timeout=10)
        print(f"Schema status: {response.status_code}")
        print(f"Schema: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_schema()
