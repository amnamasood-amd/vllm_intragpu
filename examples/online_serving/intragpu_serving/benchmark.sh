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
#   DECODE_PORTS: Comma-separated port
# =============================================================================

export HF_TOKEN="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"
#export MISCOPE_ROOT="/workspace/miscope"
#export HSA_TOOLS_LIB="/opt/rocm/lib/librocprofiler64.so.1"
#export LD_LIBRARY_PATH=/opt/rocm/lib/:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/workspace/clr/build/hipamd/lib:/opt/rocm/lib:$LD_LIBRARY_PATH # 6.4


# Configuration - can be overridden via environment variables
MODEL=${MODEL:-meta-llama/Llama-3.1-70B-Instruct}
#MODEL=${MODEL:-mistralai/Mixtral-8x7B-Instruct-v0.1}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_PORT=${PROXY_PORT:-60002}

# Default 1P3D configuration (1 Prefill + 3 Decode)
PREFILL_GPUS=${PREFILL_GPUS:-2}
DECODE_GPUS=${DECODE_GPUS:-2}
PREFILL_PORTS=${PREFILL_PORTS:-60003}
DECODE_PORTS=${DECODE_PORTS:-60005}

echo "Warning: P2P NCCL disaggregated prefill XpYd support for vLLM v1 is experimental and subject to change."
echo ""
echo "Architecture Configuration:"
echo "  Model: $MODEL"
echo "  Prefill GPUs: $PREFILL_GPUS, Ports: $PREFILL_PORTS"
echo "  Decode GPUs: $DECODE_GPUS, Ports: $DECODE_PORTS"
echo "  Proxy Port: $PROXY_PORT"
echo "  Timeout: ${TIMEOUT_SECONDS}s"
echo ""

PIDS=()

# Switch to the directory of the current script
cd "$(dirname "${BASH_SOURCE[0]}")"

check_required_files() {
    local files=("disagg_proxy.py")
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

check_num_gpus() {
    # Check if the number of GPUs are >=2 via nvidia-smi
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$num_gpus" -lt 2 ]; then
        echo "You need at least 2 GPUs to run disaggregated prefill."
        exit 1
    else
        echo "Found $num_gpus GPUs."
    fi
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
    pkill -9 -f "disagg_proxy.py"
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
    check_required_files
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
    echo "  - prefill*.log: Prefill server logs"
    echo "  - decode*.log: Decode server logs"
    echo "  - proxy.log: Proxy server log"

    # =============================================================================
    # Launch Proxy Server
    # =============================================================================
    echo ""
    echo "Starting proxy server on port $PROXY_PORT..."
    python3 disagg_proxy.py &
    PIDS+=($!)

    # Parse GPU and port arrays
    IFS=',' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
    IFS=',' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"

    echo ""
    # =============================================================================
    # Launch Decode Servers (Y Decoders)
    # =============================================================================

    echo "Starting ${#DECODE_GPU_ARRAY[@]} decode server(s)..."
    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        local gpu_id=${DECODE_GPU_ARRAY[$i]}
        local port=${DECODE_PORT_ARRAY[$i]}
        local kv_port=$((22001 + i))

        echo "  Decode server $((i+1)): GPU $gpu_id, Port $port, KV Port $kv_port"
        VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_USE_TRITON_FLASH_ATTN=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve $MODEL \
        --host 0.0.0.0 \
        --port $port \
        --tensor-parallel-size 8 \
        --seed 1024 \
        --dtype float16 \
	--async_scheduling \
        --max-model-len 131072 \
	    --max-num-seqs 512 \
        --gpu-memory-utilization 0.8 \
        --trust-remote-code \
        --no-enable-prefix-caching \
        --kv-transfer-config \
        "{\"kv_connector\":\"IntraGPUConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"$kv_port\"}" > decode.log 2> decode_external.log &
        PIDS+=($!)
        #--enforce-eager \
        #--max-num-seqs 256 \
        #--compilation-config '{"cudagraph_mode":"FULL"}' \
    done

    # =============================================================================
    # Wait for All Servers to Start
    # =============================================================================
    # echo ""
    # echo "Waiting for decode servers to start..."
    # for port in "${DECODE_PORT_ARRAY[@]}"; do
    #     if ! wait_for_server $port; then
    #         echo "Failed to start server on port $port"
    #         cleanup
    #         exit 1
    #     fi
    # done

    # =============================================================================
    # Launch Prefill Servers (X Producers)
    # =============================================================================
    echo ""
    echo "Starting ${#PREFILL_GPU_ARRAY[@]} prefill server(s)..."
    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        local gpu_id=${PREFILL_GPU_ARRAY[$i]}
        local port=${PREFILL_PORT_ARRAY[$i]}
        local kv_port=$((21001 + i))
 
        echo "  Prefill server $((i+1)): GPU $gpu_id, Port $port, KV Port $kv_port"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve $MODEL \
        --host 0.0.0.0 \
        --port $port \
        --tensor-parallel-size 8 \
        --seed 1024 \
        --dtype float16 \
        --max-model-len 131072 \
        --max-num-batched-tokens 20000 \
        --max-num-seqs 64 \
	    --trust-remote-code \
        --gpu-memory-utilization 0.8 \
        --no-enable-prefix-caching \
        --kv-transfer-config \
        "{\"kv_connector\":\"IntraGPUConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"1e1\",\"kv_port\":\"$kv_port\"}" > prefill.log 2> prefill_external.log &
        PIDS+=($!)
        #--max-num-seqs 256 \
        #--compilation-config '{"cudagraph_mode":"FULL"}' \
    done

    echo ""
    echo "Waiting for servers to start..."
    for port in "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}"; do
        if ! wait_for_server $port; then
            echo "Failed to start server on port $port"
            cleanup
            exit 1
        fi
    done


    echo ""
    echo "All servers are up. Starting benchmark..."




    # =============================================================================
    # Run Benchmark
    # =============================================================================
    # cd ../../../benchmarks/
    results_folder=/workspace/datasets/miscope

    tag=intragpu-both_torch-llama-lmsys-20k_64_512-testlogs
    input_len=1024
    for qps in 25; do
    #python3 /workspace/miscope/miscope.py \
    #--gpus=0,4 \
    #--prefix="/workspace/datasets/$tag-qps$qps" \
    #--cmd="
	vllm bench serve --port 10003 \
          --model $MODEL \
          --goodput tpot:100 \
          --dataset-name custom \
          --dataset-path /workspace/lmsys_custom_prompts_10k.jsonl \
          --endpoint /v1/completions \
          --num-prompts 10000 \
          --request-rate $qps \
          --save-result \
          --save-detailed \
          --result-dir $results_folder \
          --result-filename $tag-qps$qps.json \
          --custom-skip-chat-template

          rm -rf /workspace/vllm_intragpu/examples/online_serving/intragpu_serving/req_block_data/*
          rm -rf /workspace/vllm_intragpu/examples/online_serving/intragpu_serving/*pkl
          sleep 2
          

    done
    # echo "Benchmarking done. Cleaning up..."

    #python3 single_serve.py --port $PROXY_PORT --model $MODEL
    #python3 multi_serve.py --port 10001 --model $MODEL
    # python3 multi_serve_no_proxy.py --model $MODEL
    rm -rf /workspace/vllm_intragpu/examples/online_serving/intragpu_serving/*.pkl
    rm -rf /workspace/vllm_intragpu/examples/online_serving/intragpu_serving/req_block_data/*
    cleanup



    #rm /workspace/vllm_intragpu/examples/online_serving/intragpu_serving/scheduler_output_prefill*.pkl

    #echo $output1
}

main

