#!/usr/bin/env bash
# One seed, CIFAR100/20, unchanged upstream FP32 training, MagMax lambda=0.5.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ACTION="${1:-status}"
PYTHON_BIN="${MAGMAX_PYTHON:-python}"
GPU_ID="${MAGMAX_GPU_ID:-0}"
DATA_ROOT="${MAGMAX_DATA_ROOT:-$ROOT/datasets}"
OUT="$ROOT/outs/magmax_CIFAR100_20_seed5"
CKPTS="$ROOT/checkpoints/ViT-B-16/sequential_finetuning/class_incremental/CIFAR100-20/ft-epochs-10-seed:5"
die() { echo "[STOP] $*" >&2; exit 1; }
[[ "$GPU_ID" == 0 || "$GPU_ID" == 1 ]] || die "MAGMAX_GPU_ID must be 0 or 1."
# Upstream saves full modules. Preserve its loading behavior under PyTorch >=2.6.
[[ "${TORCH_FORCE_WEIGHTS_ONLY_LOAD:-0}" != 1 ]] || die "Unset TORCH_FORCE_WEIGHTS_ONLY_LOAD for upstream full-module checkpoints."
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

ARGS=(--model ViT-B-16 --dataset CIFAR100 --n_splits 20 --split_strategy class
      --sequential-finetuning --epochs 10 --seed 5 --batch-size 128
      --lr 1e-5 --wd 0.1 --ls 0.0 --warmup_length 500 --data-location "$DATA_ROOT")

check_environment() {
  env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" - <<'PY'
import importlib.metadata as metadata
import platform
if platform.system() != "Linux":
    raise SystemExit("Run on the Linux GPU server.")
if not (3, 9) <= tuple(map(int, platform.python_version_tuple()[:2])) <= (3, 12):
    raise SystemExit("Pinned dependencies require Python 3.9-3.12; upstream used 3.10.14.")
for name, expected in {"numpy": "1.26.4",
                       "open-clip-torch": "2.26.1", "timm": "1.0.7",
                       "pillow": "10.4.0", "tqdm": "4.66.4"}.items():
    installed = metadata.version(name)
    if installed.split("+")[0] != expected:
        raise SystemExit(f"{name}: expected {expected}, installed {installed}.")
import torch
import torchvision
import open_clip
from src.datasets.registry import get_dataset
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Expected a CUDA PyTorch build with exactly one visible GPU.")
probe = torch.randn(8, 8, device='cuda', requires_grad=True)
(probe @ probe).sum().backward()
torch.cuda.synchronize()
del probe
print(f"[ENV] Python={platform.python_version()} torch={torch.__version__} torchvision={torchvision.__version__} CUDA={torch.version.cuda}")
if (platform.python_version(), torch.__version__.split('+')[0],
    torchvision.__version__.split('+')[0], torch.version.cuda) != ('3.10.14', '2.4.0', '0.19.0', '11.8'):
    print("[ENV] Reusing ai_env runtime; versions differ from the upstream environment export.")
free, total = torch.cuda.mem_get_info()
print(f"[GPU] {torch.cuda.get_device_name(0)}; free={free/2**30:.2f}/{total/2**30:.2f} GiB")
print("[OK] CUDA/import checks and pinned MagMax dependencies passed.")
PY
  echo "[CONFIG] CIFAR100 / 20 tasks / ViT-B-16 / 10 epochs / batch128 / seed5 / lambda0.5"
  echo "[GPU] physical index=$GPU_ID"
  df -h "$ROOT"
}

check_training_state() {
  local count=0 idx
  for idx in {0..19}; do
    if [[ -f "$CKPTS/finetuned_${idx}.pt" ]]; then count=$((count + 1)); fi
  done
  if (( count > 0 && count < 20 )); then
    die "Found $count/20 task checkpoints. Upstream cannot restore the exact RNG state. Preserve/rename the task checkpoint directory before a fresh run."
  fi
  echo "[CHECKPOINTS] $count/20 exist; complete sets are reused by upstream."
}

lock_job() {
  if [[ "${MAGMAX_LOCK_HELD:-0}" != 1 ]]; then
    exec 9> "$OUT/launcher.lock"
    flock -n 9 || die 'A MagMax job is already active.'
  fi
}

dataset_ready() {
  local name digest
  for name in train test meta; do
    case "$name" in
      train) digest=16019d7e3df5f24257cddd939b257f8d ;;
      test) digest=f0ef6b0ae62326f3e7ffdfab6717acfc ;;
      meta) digest=7973b15100ade9c7d40fb424638fde48 ;;
    esac
    [[ -s "$DATA_ROOT/cifar-100-python/$name" ]] || return 1
    printf '%s  %s\n' "$digest" "$DATA_ROOT/cifar-100-python/$name" | md5sum --status -c - || return 1
  done
}

wait_then_run() {
  mkdir -p "$OUT"
  lock_job
  trap 'code=$?; printf "%s\n" "$code" > "$OUT/exit_code.txt"' EXIT
  command -v md5sum >/dev/null || die 'md5sum is required.'
  printf 'waiting_dataset\n' > "$OUT/stage.txt"
  local tunable_out="$(dirname "$ROOT")/tunable-MagMax/outs/tunable_CIFAR100_20_A_seed5"
  while ! dataset_ready; do
    if [[ -s "$tunable_out/exit_code.txt" && -s "$tunable_out/stage.txt" &&
          "$(<"$tunable_out/exit_code.txt")" != 0 && "$(<"$tunable_out/stage.txt")" == dataset ]]; then
      die 'Tunable dataset preparation failed. Inspect its download.log before retrying.'
    fi
    echo "[WAIT] CIFAR100 not ready at $DATA_ROOT. Waiting for the shared download; no download launched here."
    sleep 30
  done
  echo '[DATA READY] CIFAR100 train/test/meta checksums passed; starting MagMax.'
  # Publish the training PID only after data is ready. The Tunable launcher must
  # not mistake this waiting job for an existing MagMax dataset downloader.
  printf '%s\n' "$$" > "$OUT/job.pid"
  run_job
}

run_job() {
  mkdir -p "$OUT"
  lock_job
  trap 'code=$?; printf "%s\n" "$code" > "$OUT/exit_code.txt"' EXIT
  check_environment
  check_training_state
  export CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1
  "$PYTHON_BIN" -m pip freeze > "$OUT/environment.txt"
  "$PYTHON_BIN" -c 'import sys, torch, torchvision; print(sys.version); print("torch", torch.__version__, "torchvision", torchvision.__version__, "CUDA", torch.version.cuda)' > "$OUT/runtime.txt"
  git rev-parse HEAD > "$OUT/source_commit.txt"
  printf '%s\n' "$GPU_ID" > "$OUT/gpu.txt"
  printf '%q ' "$PYTHON_BIN" finetune_splitted.py "${ARGS[@]}" > "$OUT/training_command.txt"
  printf '\n' >> "$OUT/training_command.txt"
  local began=$SECONDS
  printf 'train\n' > "$OUT/stage.txt"
  "$PYTHON_BIN" -u finetune_splitted.py "${ARGS[@]}" 2>&1 | tee "$OUT/train.log"
  printf 'merge\n' > "$OUT/stage.txt"
  "$PYTHON_BIN" -u merge_magmax_only.py "${ARGS[@]}" --save "$OUT" 2>&1 | tee "$OUT/merge.log"
  printf '%s\n' "$((SECONDS - began))" > "$OUT/total_seconds.txt"
  printf 'done\n' > "$OUT/stage.txt"
  echo "[DONE] Result: $OUT/result.json"
}

case "$ACTION" in
  check) check_environment; check_training_state ;;
  start|retry)
    command -v flock >/dev/null || die 'Linux flock is required.'
    mkdir -p "$OUT"
    # Resolve Python now so nohup uses the exact currently activated Conda environment.
    PYTHON_BIN="$(command -v "$PYTHON_BIN")" || die "Activate ai_env (or the magmax Conda environment)."
    exec 9> "$OUT/launcher.lock"
    flock -n 9 || die "A MagMax job is already active for this output directory."
    check_environment
    check_training_state
    RUN_ACTION=run
    PID_FILE=job.pid
    if [[ "$ACTION" == retry ]]; then
      [[ -s "$OUT/exit_code.txt" && "$(<"$OUT/exit_code.txt")" != 0 ]] || die 'retry requires a previous failed launch.'
      for idx in {0..19}; do
        [[ ! -e "$CKPTS/finetuned_${idx}.pt" ]] || die 'retry is only for zero task checkpoints; preserve partial runs or use merge for a complete set.'
      done
      ARCHIVE="$OUT/failed_attempts/$(date +%Y%m%d_%H%M%S)_$$"
      mkdir -p "$ARCHIVE"
      for artifact in "$OUT"/*; do
        [[ -f "$artifact" && "${artifact##*/}" != launcher.lock ]] || continue
        mv -- "$artifact" "$ARCHIVE/"
      done
      echo "[PRESERVED] Previous failed launch: $ARCHIVE"
      printf 'waiting_dataset\n' > "$OUT/stage.txt"
      RUN_ACTION=wait-run
      PID_FILE=queue.pid
    else
      [[ ! -e "$OUT/exit_code.txt" ]] || die "A previous launch exists. Inspect status; use 'retry' with zero task checkpoints or 'merge' with a complete set."
    fi
    nohup env MAGMAX_GPU_ID="$GPU_ID" MAGMAX_PYTHON="$PYTHON_BIN" MAGMAX_DATA_ROOT="$DATA_ROOT" MAGMAX_LOCK_HELD=1 \
      bash "$ROOT/scripts/run_magmax_mica.sh" "$RUN_ACTION" > "$OUT/nohup.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$OUT/$PID_FILE"
    echo "[STARTED] PID=$!; physical GPU=$GPU_ID"
    echo "[LOG] $OUT/nohup.log"
    ;;
  run) run_job ;;
  wait-run) wait_then_run ;;
  merge)
    mkdir -p "$OUT"
    exec 9> "$OUT/launcher.lock"
    flock -n 9 || die "A MagMax job is already active for this output directory."
    check_environment
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u merge_magmax_only.py "${ARGS[@]}" --save "$OUT" 2>&1 | tee "$OUT/merge.log"
    ;;
  status)
    if [[ -s "$OUT/exit_code.txt" ]]; then
      echo "[FINISHED] exit_code=$(<"$OUT/exit_code.txt") (0 means success)"
    elif [[ -s "$OUT/job.pid" ]] && kill -0 "$(<"$OUT/job.pid")" 2>/dev/null; then
      echo "[RUNNING] PID=$(<"$OUT/job.pid")"
    elif [[ -s "$OUT/queue.pid" ]] && kill -0 "$(<"$OUT/queue.pid")" 2>/dev/null; then
      echo "[RUNNING] PID=$(<"$OUT/queue.pid")"
    else
      echo "[IDLE/STOPPED] No active job confirmed."
    fi
    [[ ! -s "$OUT/stage.txt" ]] || echo "[STAGE] $(<"$OUT/stage.txt")"
    [[ ! -s "$OUT/nohup.log" ]] || tail -n 25 "$OUT/nohup.log"
    ;;
  follow) [[ -f "$OUT/nohup.log" ]] || die "No log yet; run start first."; tail -f "$OUT/nohup.log" ;;
  summary) "$PYTHON_BIN" scripts/summarize_mica.py --run-dir "$OUT" ;;
  *) echo "Usage: bash scripts/run_magmax_mica.sh {check|start|retry|status|follow|merge|summary}"; exit 2 ;;
esac
