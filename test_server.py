#!/usr/bin/env python3
"""
Test script to check if SGLang server is running.
"""

import requests
import time

def check_server(url="http://127.0.0.1:30000"):
    """Check if server is running."""
    print(f"Checking server at {url}...")
    
    try:
        # Try health endpoint
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy!")
            return True
    except:
        pass
    
    try:
        # Try models endpoint
        response = requests.get(f"{url}/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running!")
            print(f"Models: {response.json()}")
            return True
    except:
        pass
    
    print("❌ Server is not responding")
    return False

if __name__ == "__main__":
    # Check multiple times
    for i in range(3):
        if check_server():
            break
        print(f"Waiting... ({i+1}/3)")
        time.sleep(5)