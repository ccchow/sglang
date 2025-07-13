#!/bin/bash
# Launch SGLang server in background

cd /home/lei/git/ccchow/sglang

echo "Launching SGLang server with GPT-2..."

python -m sglang.launch_server \
    --model-path gpt2 \
    --host 127.0.0.1 \
    --port 30000 \
    --mem-fraction-static 0.8 \
    --trust-remote-code \
    --log-level info \
    --disable-cuda-graph \
    --grammar-backend none \
    > server.log 2>&1 &

SERVER_PID=$!
echo "Server launched with PID: $SERVER_PID"

# Wait for server to start
echo "Waiting for server to start..."
sleep 10

# Check if server is running
if ps -p $SERVER_PID > /dev/null; then
    echo "✅ Server process is running"
    
    # Test the server
    echo "Testing server..."
    python test_server.py
else
    echo "❌ Server process died"
    echo "Last 20 lines of server.log:"
    tail -20 server.log
fi

echo "To stop the server, run: kill $SERVER_PID"