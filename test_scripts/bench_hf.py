
import requests
import time

base_url = "https://aditya-ranjan1234-long-horizon-memory-v2.hf.space"

def benchmark_step():
    session = requests.Session()
    print("Resetting...")
    start = time.time()
    session.post(f"{base_url}/reset")
    print(f"Reset took {time.time() - start:.2f}s")
    
    print("Stepping...")
    start = time.time()
    payload = {"action": {"operation": "noop", "rewrite_memory": None}}
    res = session.post(f"{base_url}/step", json=payload)
    print(f"Step took {time.time() - start:.2f}s")
    print(f"Result: {res.status_code}")

if __name__ == "__main__":
    benchmark_step()
