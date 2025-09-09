#!/bin/bash

# =============================================================================
# vLLM Disaggregated Serving Script - P2P NCCL XpYd Architecture
# =============================================================================
# This script demonstrates disaggregated prefill and decode serving using
# P2P NCCL communication. The architecture supports various XpYd configurations:
#
# - 1P3D: 1 Prefill server + 3 Decode servers (current default)
# - 3P1D: 3 Prefill servers + 1 Decode server
# - etc.
#
# Configuration can be customized via environment variables:
#   MODEL: Model to serve
#   PREFILL_GPUS: Comma-separated GPU IDs for prefill servers
#   DECODE_GPUS: Comma-separated GPU IDs for decode servers
#   PREFILL_PORTS: Comma-separated ports for prefill servers
#   DECODE_PORTS: Comma-separated ports for decode servers
#   PROXY_PORT: Proxy server port used to setup XpYd connection.
#   TIMEOUT_SECONDS: Server startup timeout
# =============================================================================

# Configuration - can be overridden via environment variables
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_PORT=${PROXY_PORT:-30001}

# Default 1P3D configuration (1 Prefill + 3 Decode)
AMD_GPU=5
AMD_PORT=20003

PIDS=()

# Switch to the directory of the current script
cd "$(dirname "${BASH_SOURCE[0]}")"

check_required_files() {
    local files=("proxy.py")
    for file in "${files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo "Required file $file not found in $(pwd)"
            exit 1
        fi
    done
}

check_hf_token() {
    if [ -z "$HF_TOKEN" ]; then
        echo "HF_TOKEN is not set. Please set it to your Hugging Face token."
        echo "Example: export HF_TOKEN=your_token_here"
        exit 1
    fi
    if [[ "$HF_TOKEN" != hf_* ]]; then
        echo "HF_TOKEN is not a valid Hugging Face token. Please set it to your Hugging Face token."
        exit 1
    fi
    echo "HF_TOKEN is set and valid."
}

ensure_python_library_installed() {
    echo "Checking if $1 is installed..."
    if ! python3 -c "import $1" > /dev/null 2>&1; then
        echo "$1 is not installed. Please install it via pip install $1."
        exit 1
    else
        echo "$1 is installed."
    fi
}

cleanup() {
    echo "Stopping everything…"
    trap - INT TERM        # prevent re-entrancy
    pkill -9 -f "proxy.py"
    kill -- -$$            # negative PID  ==  "this whole process-group"
    wait                   # reap children so we don't leave zombies
    exit 0
}

wait_for_server() {
  local port=$1
  local timeout_seconds=$TIMEOUT_SECONDS
  local start_time=$(date +%s)

  echo "Waiting for server on port $port..."

  while true; do
    if curl -s "localhost:${port}/v1/completions" > /dev/null; then
      echo "Server on port $port is ready."
      return 0
    fi

    local now=$(date +%s)
    if (( now - start_time >= timeout_seconds )); then
      echo "Timeout waiting for server on port $port"
      return 1
    fi

    sleep 1
  done
}

main() {
    #check_required_files
    check_hf_token
    #check_num_gpus
    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm
    ensure_python_library_installed quart

    trap cleanup INT
    trap cleanup USR1
    trap cleanup TERM

    echo "Launching disaggregated serving components..."
    echo "Please check the log files for detailed output:"
    echo "  - server.log: server logs"

    # =============================================================================
    # Launch Proxy Server
    # =============================================================================
    
    #echo ""
    #echo "Starting proxy server on port $PROXY_PORT..."
    #python3 proxy.py &
    #PIDS+=($!)
    # =============================================================================
    # Launch Prefill Servers (X Producers)
    # =============================================================================
    echo ""
    echo "Starting server(s)..."
    local gpu_id=$AMD_GPU
    local port=$AMD_PORT

    echo "  Server: GPU $gpu_id, Port $port"
    CUDA_VISIBLE_DEVICES=$gpu_id VLLM_USE_V1=1 vllm serve $MODEL \
    --host 0.0.0.0 \
    --port $port \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --dtype float16 \
    --max-model-len 10000 \
    --max-num-batched-tokens 10000 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode": "FULL"}' \
    --gpu-memory-utilization 0.9 > server.log &
    PIDS+=($!)
    

    # =============================================================================

    # =============================================================================
    # Wait for All Servers to Start
    # =============================================================================
    echo ""
    echo "Waiting for all servers to start..."
    if ! wait_for_server $AMD_PORT; then
            echo "Failed to start server on port $port"
            cleanup
            exit 1
    fi
    

    echo ""
    echo "All servers are up. Starting benchmark..."

    # =============================================================================
    # Run Benchmark
    # =============================================================================
    #cd ../../../benchmarks/
    #vllm bench serve --port 20003 --seed $(date +%s) \
    #    --model $MODEL \
    #    --dataset-name random --random-input-len 7500 --random-output-len 200 \
    #    --num-prompts 1 --burstiness 100 --request-rate 2 | tee benchmark.log
    
    #echo "Benchmarking done. Cleaning up..."

    #output1=$(curl -X POST -s http://localhost:30001/v1/completions \
    #-H "Content-Type: application/json" \
    #-d '{
    #"model": "'"$MODEL"'",
    #"prompt": "San Francisco is a",
    #"max_tokens": 10,
    #"temperature": 0
    #}')
    python3 single_serve.py --port 20003 --model $MODEL
    cleanup

    #echo $output1
}

main
