#!/usr/bin/env bash
#
# Two-node online HashGram training for Qwen3-8B.
#
#   inference node: 8 independent single-GPU verifier replicas (vLLM DP=8)
#                   + Mooncake master + hidden-state producer
#   training node:  8 local DDP ranks + Mooncake hidden-state consumer
#
# When PET_NNODES=2 and PET_NODE_RANK are provided, run this script exactly once
# on each node: node 0 becomes inference and node 1 becomes training automatically.
# Otherwise, set ROLE explicitly:
#
#   # Node A: verifier inference / hidden-state extraction
#   ROLE=inference \
#     FABRIC_SUBNET=10.0.0. \
#     bash examples/train/hashgram_qwen3_8b_sharegpt_2node_mooncake.sh
#
#   # Node B: HashGram training
#   ROLE=training \
#     INFERENCE_ADDR=<node-A-fast-fabric-ip> \
#     FABRIC_SUBNET=10.0.0. \
#     bash examples/train/hashgram_qwen3_8b_sharegpt_2node_mooncake.sh
#
# Both roles default to Mooncake RDMA. This repository's connector stages hidden
# states through pinned CPU memory; RDMA is used for the cross-node DRAM transfer,
# not GPUDirect GPU-to-GPU transfer.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
workspace_root="$(cd "$repo_root/.." && pwd)"
ROOT="$workspace_root"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$ROOT/.secrets/wandb_key}"

cluster_nnodes="${PET_NNODES:-}"
cluster_node_rank="${PET_NODE_RANK:-}"
cluster_master_addr="${PET_MASTER_ADDR:-${MASTER_ADDR:-}}"
cluster_master_port="${PET_MASTER_PORT:-${MASTER_PORT:-}}"
inference_node_rank="${INFERENCE_NODE_RANK:-0}"

role="${ROLE:-}"
if [[ -z "$role" && "$cluster_nnodes" == "2" && "$cluster_node_rank" =~ ^[0-9]+$ ]]; then
  if (( cluster_node_rank == inference_node_rank )); then
    role="inference"
  else
    role="training"
  fi
fi
model_path="/inspire/sfs/project/inf-multimodal/public/wumengke/model_weights/Qwen--Qwen3-8B"
data_dir="/inspire/sfs/project/inf-multimodal/public/wumengke/datasets/jihwan1205--perfectblend-qwen3-8b-regen/800k-len8192"
run_name="${RUN_NAME:-hashgram-qwen3-8b-perfectblend-mooncake}"
run_dir="${RUN_DIR:-$repo_root/runs/hashgram_qwen3_8b_mooncake/$run_name}"
log_dir="$run_dir/logs"

fabric_subnet="${FABRIC_SUBNET:-}"
fabric_iface="${FABRIC_IFACE:-}"
local_fabric_addr="${LOCAL_FABRIC_ADDR:-}"
inference_addr="${INFERENCE_ADDR:-}"
if [[ -z "$inference_addr" && "$inference_node_rank" == "0" ]]; then
  inference_addr="$cluster_master_addr"
fi

derived_mooncake_port=""
derived_vllm_port=""
derived_metrics_port=""
derived_training_port=""
if [[ "$cluster_master_port" =~ ^[0-9]+$ ]] && (( cluster_master_port <= 65430 )); then
  derived_mooncake_port="$((cluster_master_port + 100))"
  derived_vllm_port="$((cluster_master_port + 101))"
  derived_metrics_port="$((cluster_master_port + 102))"
  derived_training_port="$((cluster_master_port + 103))"
fi
vllm_port="${VLLM_PORT:-${derived_vllm_port:-8000}}"
mooncake_master_port="${MOONCAKE_MASTER_PORT:-${derived_mooncake_port:-50051}}"
mooncake_metrics_port="${MOONCAKE_METRICS_PORT:-${derived_metrics_port:-9003}}"
training_master_addr="${TRAINING_MASTER_ADDR:-127.0.0.1}"
training_master_port="${TRAINING_MASTER_PORT:-${derived_training_port:-29599}}"
mooncake_protocol="${MOONCAKE_PROTOCOL:-rdma}"
mooncake_metadata_server="${MOONCAKE_METADATA_SERVER:-P2PHANDSHAKE}"
mooncake_device_name="${MOONCAKE_DEVICE_NAME:-${NCCL_IB_HCA:-}}"

seq_length="${SEQ_LENGTH:-8192}"
target_layer_ids_string="${TARGET_LAYER_IDS:-1 9 17 25 33}"
read -r -a target_layer_ids <<< "$target_layer_ids_string"

default_cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
node_processes="${PET_NPROC_PER_NODE:-8}"
inference_cuda_visible_devices="${INFERENCE_CUDA_VISIBLE_DEVICES:-$default_cuda_visible_devices}"
inference_dp_size="${INFERENCE_DP_SIZE:-$node_processes}"
inference_api_servers="${INFERENCE_API_SERVERS:-1}"
vllm_gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
vllm_max_num_seqs="${VLLM_MAX_NUM_SEQS:-4}"
vllm_max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
vllm_start_timeout="${VLLM_START_TIMEOUT:-1800}"
vllm_python="${VLLM_PYTHON:-$repo_root/vllm_venv/bin/python}"
mooncake_master_bin="${MOONCAKE_MASTER_BIN:-$repo_root/vllm_venv/bin/mooncake_master}"
training_python="${TRAIN_PYTHON:-$repo_root/speculators_venv/bin/python}"
network_python="${NETWORK_PYTHON:-}"
if [[ -z "$network_python" ]]; then
  network_python="$(command -v python3 || command -v python || true)"
fi
ip_bin="${IP_BIN:-$(command -v ip || true)}"

# DP=8 creates eight producer clients. Keep the per-client segment modest: a
# maximum-length sample with five auxiliary layers plus the final bf16 hidden
# tensor is about 1.5 GiB.
producer_global_segment_gib="${MOONCAKE_PRODUCER_GLOBAL_SEGMENT_GIB:-8}"
producer_local_buffer_gib="${MOONCAKE_PRODUCER_LOCAL_BUFFER_GIB:-4}"
consumer_local_buffer_gib="${MOONCAKE_CONSUMER_LOCAL_BUFFER_GIB:-4}"
mooncake_writer_threads="${MOONCAKE_WRITER_THREADS:-4}"

training_cuda_visible_devices="${TRAINING_CUDA_VISIBLE_DEVICES:-$default_cuda_visible_devices}"
training_nproc="${TRAINING_NPROC:-$node_processes}"
epochs="${EPOCHS:-3}"
train_data_ratio="${TRAIN_DATA_RATIO:-0.98}"
adamw_learning_rate="${LR:-1e-4}"
muon_learning_rate="${MUON_LR:-2e-4}"
noise_std="${NOISE_STD:-0}"
scheduler_type="${SCHEDULER_TYPE:-cosine}"
scheduler_warmup_ratio="${SCHEDULER_WARMUP_RATIO:-0.04}"
checkpoint_freq="${CHECKPOINT_FREQ:-0.1}"
num_layers="${NUM_LAYERS:-5}"
sliding_window="${SLIDING_WINDOW:-2048}"
block_size="${BLOCK_SIZE:-8}"
max_anchors="${MAX_ANCHORS:-512}"
torchrun_bin="${TORCHRUN_BIN:-$repo_root/speculators_venv/bin/torchrun}"
wandb_project="${WANDB_PROJECT:-hashgram-qwen3-8b}"

hashgram_rank="${HASHGRAM_RANK:-128}"
hashgram_top_k="${HASHGRAM_TOP_K:-16}"
hashgram_bigram_buckets="${HASHGRAM_BIGRAM_BUCKETS:-1048576}"
hashgram_trigram_buckets="${HASHGRAM_TRIGRAM_BUCKETS:-1048576}"
hashgram_num_hashes="${HASHGRAM_NUM_HASHES:-1}"
hashgram_loss_alpha="${HASHGRAM_LOSS_ALPHA:-1.0}"
hashgram_markov_rank="${HASHGRAM_MARKOV_RANK:-256}"

usage() {
  cat <<'EOF'
Usage:
  # Cluster auto mode: PET_NNODES=2, PET_NODE_RANK, PET_MASTER_ADDR are injected
  bash <script>   # run exactly once on each node

  # Explicit mode
  ROLE=inference [INFERENCE_ADDR=<fast-ip>] FABRIC_SUBNET=<prefix> bash <script>
  ROLE=training INFERENCE_ADDR=<inference-fast-ip> FABRIC_SUBNET=<prefix> bash <script>

Required on both nodes:
  - The outer scheduler must launch exactly one copy of this script per node;
    this script launches the per-GPU vLLM/DDP processes itself.
  - The same repository and Qwen3-8B weights must be visible.
  - hs_connectors and mooncake-transfer-engine must be installed.
  - RDMA requires working Mellanox/IB/RoCE devices and permissions.

Important overrides:
  RUN_DIR, VLLM_PYTHON, MOONCAKE_MASTER_BIN, TRAIN_PYTHON, TORCHRUN_BIN,
  INFERENCE_CUDA_VISIBLE_DEVICES, TRAINING_CUDA_VISIBLE_DEVICES,
  MOONCAKE_PROTOCOL (rdma|tcp), MOONCAKE_DEVICE_NAME, MOONCAKE_MASTER,
  INFERENCE_NODE_RANK, TRAINING_MASTER_ADDR, TRAINING_MASTER_PORT,
  WANDB_KEY_FILE,
  DRY_RUN=1 (print resolved topology and exit).
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

load_wandb_api_key() {
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    return 0
  fi
  [[ -r "$WANDB_KEY_FILE" ]] || die \
    "W&B key file is not readable: $WANDB_KEY_FILE"
  local wandb_key=""
  IFS= read -r wandb_key < "$WANDB_KEY_FILE" || true
  [[ -n "$wandb_key" ]] || die "W&B key file is empty: $WANDB_KEY_FILE"
  export WANDB_API_KEY="$wandb_key"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

interface_ipv4() {
  "$network_python" - "$1" <<'PY'
import fcntl
import socket
import struct
import sys

interface = sys.argv[1].encode()[:15]
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    packed = struct.pack("256s", interface)
    result = fcntl.ioctl(sock.fileno(), 0x8915, packed)
print(socket.inet_ntoa(result[20:24]))
PY
}

route_source_ipv4() {
  "$network_python" - "$1" <<'PY'
import socket
import sys

remote = socket.gethostbyname(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    sock.connect((remote, 9))
    print(sock.getsockname()[0])
PY
}

validate_visible_devices() {
  local visible_devices="$1"
  local requested_processes="$2"
  local purpose="$3"
  local visible_count
  visible_count="$(awk -F, '{print NF}' <<< "$visible_devices")"
  if (( requested_processes > visible_count )); then
    die "$purpose requests $requested_processes processes but CUDA_VISIBLE_DEVICES exposes $visible_count GPU(s). Launch this script once per node, not once per GPU."
  fi
}

resolve_fabric() {
  if [[ -z "$fabric_iface" ]]; then
    local socket_ifname="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-}}"
    socket_ifname="${socket_ifname%%,*}"
    socket_ifname="${socket_ifname#=}"
    if [[ -n "$socket_ifname" && "$socket_ifname" != ^* && "$socket_ifname" != *\** ]]; then
      fabric_iface="$socket_ifname"
    fi
  fi

  if [[ -z "$local_fabric_addr" ]]; then
    if [[ -z "$fabric_iface" && -n "$fabric_subnet" ]]; then
      [[ -n "$ip_bin" ]] || die \
        "FABRIC_SUBNET discovery requires iproute2; use the injected GLOO_SOCKET_IFNAME or set FABRIC_IFACE."
      fabric_iface="$(
        "$ip_bin" -o -4 addr show | awk -v subnet="$fabric_subnet" \
          'index($4, subnet) == 1 {print $2; exit}'
      )"
    fi
    if [[ -n "$fabric_iface" ]]; then
      local_fabric_addr="$(interface_ipv4 "$fabric_iface" 2>/dev/null || true)"
    fi
  fi

  if [[ -z "$local_fabric_addr" && -n "$inference_addr" ]]; then
    local_fabric_addr="$(route_source_ipv4 "$inference_addr" 2>/dev/null || true)"
  fi

  [[ -n "$local_fabric_addr" ]] || die \
    "Could not resolve local fabric IP from PET/NCCL/Gloo variables; set LOCAL_FABRIC_ADDR."

  if [[ -n "$fabric_iface" ]]; then
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$fabric_iface}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$fabric_iface}"
  fi
  export MOONCAKE_LOCAL_HOSTNAME="$local_fabric_addr"
}

resolve_mooncake_device_name() {
  mooncake_device_name="${mooncake_device_name#=}"
  if [[ "$mooncake_device_name" == ^* || "$mooncake_device_name" == *\** ]]; then
    echo "WARN: NCCL_IB_HCA is an exclusion/pattern expression; Mooncake will auto-discover HCAs." >&2
    mooncake_device_name=""
    return
  fi

  local normalized=()
  local hca
  local hcas=()
  IFS=',' read -r -a hcas <<< "$mooncake_device_name"
  for hca in "${hcas[@]}"; do
    hca="${hca%%:*}"
    [[ -n "$hca" ]] && normalized+=("$hca")
  done
  mooncake_device_name="$(IFS=,; echo "${normalized[*]}")"
}

append_no_proxy() {
  local addresses=("127.0.0.1" "localhost" "$local_fabric_addr")
  if [[ -n "$inference_addr" ]]; then
    addresses+=("$inference_addr")
  fi
  local joined
  local current_no_proxy="${no_proxy:-${NO_PROXY:-}}"
  joined="$(IFS=,; echo "${addresses[*]}")"
  export no_proxy="${current_no_proxy:+$current_no_proxy,}$joined"
  export NO_PROXY="$no_proxy"
}

wait_for_tcp() {
  local host="$1"
  local port="$2"
  local timeout_seconds="$3"
  local deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS < deadline )); do
    if (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; then
      exec 3>&-
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_for_vllm() {
  local endpoint="$1"
  local timeout_seconds="$2"
  local process_id="${3:-}"
  local deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS < deadline )); do
    if curl -fsS "$endpoint/health" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "$process_id" ]] && ! kill -0 "$process_id" 2>/dev/null; then
      return 1
    fi
    sleep 2
  done
  return 1
}

terminate_process_group() {
  local process_id="${1:-}"
  [[ -n "$process_id" ]] || return 0
  kill -0 "$process_id" 2>/dev/null || return 0
  kill -TERM -- "-$process_id" 2>/dev/null || kill -TERM "$process_id" 2>/dev/null || true
  for _ in {1..30}; do
    kill -0 "$process_id" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -- "-$process_id" 2>/dev/null || kill -KILL "$process_id" 2>/dev/null || true
}

run_inference() {
  require_command curl
  require_command setsid
  [[ -x "$vllm_python" ]] || die "vLLM Python is not executable: $vllm_python"
  [[ -x "$mooncake_master_bin" ]] || die \
    "Mooncake master is not executable: $mooncake_master_bin"
  "$vllm_python" -c 'import importlib.util as u; assert u.find_spec("mooncake") and u.find_spec("transformers") and u.find_spec("vllm")' || die \
    "Inference environment must contain mooncake, transformers, and vllm"
  [[ -f "$model_path/config.json" ]] || die "Model config not found: $model_path/config.json"

  if [[ -z "$inference_addr" ]]; then
    inference_addr="$local_fabric_addr"
  fi
  local mooncake_master_address="${MOONCAKE_MASTER:-$inference_addr:$mooncake_master_port}"
  local endpoint="http://$inference_addr:$vllm_port"
  append_no_proxy
  mkdir -p "$log_dir"

  local master_pid=""
  local vllm_pid=""
  cleanup_inference() {
    terminate_process_group "$vllm_pid"
    terminate_process_group "$master_pid"
  }
  trap cleanup_inference EXIT INT TERM

  echo "Starting Mooncake master at $mooncake_master_address"
  setsid "$mooncake_master_bin" \
    --rpc_port "$mooncake_master_port" \
    --metrics_port "$mooncake_metrics_port" \
    --rpc_thread_num 8 \
    --enable_disk_eviction=false \
    --logtostderr=true \
    >"$log_dir/mooncake_master.log" 2>&1 &
  master_pid=$!

  if ! wait_for_tcp "$inference_addr" "$mooncake_master_port" 60; then
    tail -n 100 "$log_dir/mooncake_master.log" >&2 || true
    die "Mooncake master did not open port $mooncake_master_port"
  fi

  echo "Starting Qwen3-8B verifier DP=$inference_dp_size, TP=1"
  echo "Mooncake protocol=$mooncake_protocol devices=${mooncake_device_name:-auto}; log: $log_dir/vllm.log"
  setsid env \
    -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    -u GROUP_RANK -u ROLE_RANK -u MASTER_ADDR -u MASTER_PORT \
    VLLM_PLUGINS="${VLLM_PLUGINS:-}" \
    CUDA_VISIBLE_DEVICES="$inference_cuda_visible_devices" \
    "$vllm_python" "$repo_root/scripts/launch_vllm.py" "$model_path" \
      --hidden-states-backend mooncake \
      --mooncake-master "$mooncake_master_address" \
      --mooncake-metadata-server "$mooncake_metadata_server" \
      --mooncake-protocol "$mooncake_protocol" \
      --mooncake-device-name "$mooncake_device_name" \
      --mooncake-global-segment-gib "$producer_global_segment_gib" \
      --mooncake-local-buffer-gib "$producer_local_buffer_gib" \
      --mooncake-writer-threads "$mooncake_writer_threads" \
      --target-layer-ids "${target_layer_ids[@]}" \
      -- \
      --host 0.0.0.0 \
      --port "$vllm_port" \
      --tensor-parallel-size 1 \
      --data-parallel-size "$inference_dp_size" \
      --data-parallel-backend mp \
      --api-server-count "$inference_api_servers" \
      --max-model-len "$((seq_length + 1))" \
      --max-num-seqs "$vllm_max_num_seqs" \
      --max-num-batched-tokens "$vllm_max_num_batched_tokens" \
      --gpu-memory-utilization "$vllm_gpu_memory_utilization" \
      --no-enable-prefix-caching \
      --disable-uvicorn-access-log \
      >"$log_dir/vllm.log" 2>&1 &
  vllm_pid=$!

  if ! wait_for_vllm "$endpoint" "$vllm_start_timeout" "$vllm_pid"; then
    tail -n 200 "$log_dir/vllm.log" >&2 || true
    die "vLLM failed to become healthy at $endpoint"
  fi

  echo "Inference node ready: $endpoint/v1"
  echo "Start ROLE=training on the other node with INFERENCE_ADDR=$inference_addr"

  while kill -0 "$vllm_pid" 2>/dev/null && kill -0 "$master_pid" 2>/dev/null; do
    sleep 5
  done
  if ! kill -0 "$master_pid" 2>/dev/null; then
    tail -n 100 "$log_dir/mooncake_master.log" >&2 || true
    die "Mooncake master exited while vLLM was running"
  fi
  wait "$vllm_pid"
}

run_training() {
  require_command curl
  require_command setsid
  require_command "$torchrun_bin"
  [[ -x "$training_python" ]] || die "Training Python is not executable: $training_python"
  "$training_python" -c 'import importlib.util as u; assert u.find_spec("mooncake") and u.find_spec("speculators") and u.find_spec("wandb")' || die \
    "Training environment must contain mooncake, speculators, and wandb"
  [[ -n "$inference_addr" ]] || die "ROLE=training requires INFERENCE_ADDR"
  [[ -f "$model_path/config.json" ]] || die "Model config not found: $model_path/config.json"
  [[ -f "$data_dir/state.json" ]] || die "Prepared dataset state.json not found: $data_dir"
  compgen -G "$data_dir/*.arrow" >/dev/null || die \
    "No prepared Arrow shards found in: $data_dir"
  load_wandb_api_key

  local mooncake_master_address="${MOONCAKE_MASTER:-$inference_addr:$mooncake_master_port}"
  local endpoint="http://$inference_addr:$vllm_port"
  append_no_proxy
  mkdir -p "$log_dir" "$run_dir/checkpoints"

  echo "Waiting for verifier endpoint: $endpoint"
  if ! wait_for_vllm "$endpoint" "$vllm_start_timeout"; then
    die "Remote vLLM endpoint is not healthy: $endpoint"
  fi

  local train_pid=""
  cleanup_training() {
    terminate_process_group "$train_pid"
  }
  trap cleanup_training EXIT INT TERM

  echo "Starting single-node HashGram DDP with $training_nproc ranks"
  echo "Training rendezvous=static://$training_master_addr:$training_master_port"
  echo "Mooncake=$mooncake_master_address protocol=$mooncake_protocol devices=${mooncake_device_name:-auto}"
  echo "Training log: $log_dir/train.log"
  setsid env \
    -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    -u GROUP_RANK -u ROLE_RANK -u MASTER_ADDR -u MASTER_PORT \
    -u PET_NNODES -u PET_NODE_RANK -u PET_NPROC_PER_NODE \
    -u PET_MASTER_ADDR -u PET_MASTER_PORT \
    CUDA_VISIBLE_DEVICES="$training_cuda_visible_devices" \
    "$torchrun_bin" \
      --nnodes 1 \
      --node-rank 0 \
      --rdzv-backend static \
      --master-addr "$training_master_addr" \
      --master-port "$training_master_port" \
      --nproc-per-node "$training_nproc" \
      "$repo_root/scripts/train.py" \
      --verifier-name-or-path "$model_path" \
      --data-path "$data_dir" \
      --save-path "$run_dir/checkpoints" \
      --epochs "$epochs" \
      --checkpoint-freq "$checkpoint_freq" \
      --train-data-ratio "$train_data_ratio" \
      --optimizer muon \
      --muon-lr "$muon_learning_rate" \
      --lr "$adamw_learning_rate" \
      --noise-std "$noise_std" \
      --scheduler-type "$scheduler_type" \
      --scheduler-warmup-ratio "$scheduler_warmup_ratio" \
      --total-seq-len "$seq_length" \
      --speculator-type hashgram \
      --block-size "$block_size" \
      --max-anchors "$max_anchors" \
      --sample-from-anchor \
      --num-layers "$num_layers" \
      --sliding-window "$sliding_window" \
      --sliding-window-non-causal \
      --target-layer-ids "${target_layer_ids[@]}" \
      --hashgram-rank "$hashgram_rank" \
      --hashgram-top-k "$hashgram_top_k" \
      --hashgram-bigram-buckets "$hashgram_bigram_buckets" \
      --hashgram-trigram-buckets "$hashgram_trigram_buckets" \
      --hashgram-num-hashes "$hashgram_num_hashes" \
      --hashgram-loss-alpha "$hashgram_loss_alpha" \
      --hashgram-markov-rank "$hashgram_markov_rank" \
      --loss-fn '{"ce":0.1,"tv":0.9}' \
      --hidden-states-backend mooncake \
      --mooncake-master "$mooncake_master_address" \
      --mooncake-metadata-server "$mooncake_metadata_server" \
      --mooncake-protocol "$mooncake_protocol" \
      --mooncake-device-name "$mooncake_device_name" \
      --mooncake-global-segment-gib 0 \
      --mooncake-local-buffer-gib "$consumer_local_buffer_gib" \
      --vllm-endpoint "$endpoint/v1" \
      --on-missing generate \
      --on-generate delete \
      --request-timeout 900 \
      --max-retries 5 \
      --generation-validation-retries 2 \
      --max-consecutive-generation-failures 20 \
      --log-freq 20 \
      --logger wandb \
      --log-dir "$log_dir" \
      --run-name "$run_name" \
      >"$log_dir/train.log" 2>&1 &
  train_pid=$!

  local train_status=0
  wait "$train_pid" || train_status=$?
  train_pid=""
  if (( train_status != 0 )); then
    tail -n 200 "$log_dir/train.log" >&2 || true
    die "HashGram training exited with status $train_status"
  fi
  echo "Training completed. Checkpoints: $run_dir/checkpoints"
}

if [[ "$role" != "inference" && "$role" != "training" ]]; then
  usage
  [[ -n "$role" ]] && echo "ERROR: unsupported ROLE=$role" >&2
  exit 2
fi

export PYTHONPATH="$repo_root/hs_connectors/src:$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$workspace_root/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$workspace_root/datasets/.cache}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export TORCH_DISTRIBUTED_USE_LIBUV=0
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_PROJECT="$wandb_project"

require_command "$network_python"
resolve_fabric
resolve_mooncake_device_name

if [[ "$mooncake_protocol" == "rdma" ]] && ! command -v ibv_devices >/dev/null 2>&1; then
  echo "WARN: ibv_devices is unavailable; verify RDMA devices from Mooncake logs." >&2
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cat <<EOF
Resolved two-node HashGram topology:
  role=$role
  model_path=$model_path
  data_dir=$data_dir
  cluster_nnodes=${cluster_nnodes:-unset}
  cluster_node_rank=${cluster_node_rank:-unset}
  local_fabric_addr=$local_fabric_addr
  fabric_iface=${fabric_iface:-auto}
  inference_addr=${inference_addr:-$local_fabric_addr}
  vllm_port=$vllm_port
  mooncake_master_port=$mooncake_master_port
  mooncake_metrics_port=$mooncake_metrics_port
  training_master_addr=$training_master_addr
  training_master_port=$training_master_port
  mooncake_protocol=$mooncake_protocol
  mooncake_device_name=${mooncake_device_name:-auto}
  mooncake_master_bin=$mooncake_master_bin
  inference_dp_size=$inference_dp_size
  training_nproc=$training_nproc
  epochs=$epochs
  train_data_ratio=$train_data_ratio
  optimizer=muon
  muon_lr=$muon_learning_rate
  adamw_lr=$adamw_learning_rate
  noise_std=$noise_std
  scheduler_type=$scheduler_type
  scheduler_warmup_ratio=$scheduler_warmup_ratio
  num_layers=$num_layers
  sliding_window=$sliding_window
  sliding_window_non_causal=true
  checkpoint_freq=$checkpoint_freq
  logger=wandb
  wandb_project=$wandb_project
  wandb_key_file=$WANDB_KEY_FILE
  inference_cuda_visible_devices=$inference_cuda_visible_devices
  training_cuda_visible_devices=$training_cuda_visible_devices
EOF
  exit 0
fi

cd "$repo_root"
if [[ "$role" == "inference" ]]; then
  validate_visible_devices "$inference_cuda_visible_devices" "$inference_dp_size" "Inference DP"
  run_inference
else
  validate_visible_devices "$training_cuda_visible_devices" "$training_nproc" "Training DDP"
  run_training
fi
