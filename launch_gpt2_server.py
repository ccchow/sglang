#!/usr/bin/env python3
"""
Launch SGLang server with GPT-2 model for MeZO training demonstration.
"""

import subprocess
import time
import requests
import sys
import os

# Add path to SGLang
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

from sglang.utils import wait_for_server

def launch_server():
    """Launch SGLang server with GPT-2."""
    print("Launching SGLang server with GPT-2...")
    
    # Server launch command
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", "gpt2",  # Use the smallest GPT-2 model
        "--host", "127.0.0.1",
        "--port", "30000",
        "--mem-fraction-static", "0.8",  # Reserve memory for training
        "--trust-remote-code",
        "--log-level", "info",
        "--disable-cuda-graph",  # Disable for more flexibility during training
        "--grammar-backend", "none"  # Disable grammar backend to avoid xgrammar dependency
    ]
    
    # Launch server as subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Print output in real-time
    print("Server starting up...")
    server_ready = False
    
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        if "Server is ready" in line or "Uvicorn running on" in line:
            server_ready = True
            break
    
    if server_ready:
        print("\n✅ Server is ready!")
        print("Server URL: http://127.0.0.1:30000")
        print("\nYou can now:")
        print("1. Send requests to the server")
        print("2. Use the OpenAI-compatible API")
        print("3. Run MeZO training scripts")
        print("\nPress Ctrl+C to stop the server")
        
        # Keep the server running
        try:
            while True:
                line = process.stdout.readline()
                if line:
                    print(line, end='')
                else:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nShutting down server...")
            process.terminate()
            process.wait()
            print("Server stopped.")
    else:
        print("\n❌ Failed to start server")
        process.terminate()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(launch_server())