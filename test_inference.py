#!/usr/bin/env python3
"""
Test inference with SGLang server.
"""

import requests
import json

def test_completion():
    """Test text completion with the server."""
    url = "http://127.0.0.1:30000/v1/completions"
    
    # Test request
    data = {
        "model": "gpt2",
        "prompt": "Once upon a time, there was a",
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    print("Sending completion request...")
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Success!")
        print(f"Prompt: {data['prompt']}")
        print(f"Completion: {result['choices'][0]['text']}")
        print(f"\nFull response: {json.dumps(result, indent=2)}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

def test_chat():
    """Test chat completion with the server."""
    url = "http://127.0.0.1:30000/v1/chat/completions"
    
    # Test request
    data = {
        "model": "gpt2",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    print("\nSending chat completion request...")
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Success!")
        print(f"Question: {data['messages'][0]['content']}")
        print(f"Answer: {result['choices'][0]['message']['content']}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

def get_model_info():
    """Get model information."""
    url = "http://127.0.0.1:30000/v1/models"
    
    print("\nGetting model info...")
    response = requests.get(url)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Available models:")
        for model in result['data']:
            print(f"  - {model['id']}")
            print(f"    Created: {model.get('created', 'N/A')}")
            print(f"    Object: {model.get('object', 'N/A')}")
    else:
        print(f"❌ Error: {response.status_code}")

if __name__ == "__main__":
    print("Testing SGLang Server Inference")
    print("="*50)
    
    get_model_info()
    test_completion()
    test_chat()