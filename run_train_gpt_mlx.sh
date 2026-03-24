#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Backends:
#   RUN_BACKEND=auto  -> macOS uses MLX, everything else uses PyTorch
#   RUN_BACKEND=mlx   -> force MLX
#   RUN_BACKEND=torch -> force PyTorch
#
# Presets:
#   RUN_PRESET=default  -> stronger general-purpose run
#   RUN_PRESET=seq2048  -> same idea, but with 2048-token context
#
# Override any value at launch time, for example:
# RUN_ID=my_run ITERATIONS=4000 TRAIN_BATCH_TOKENS=131072 ./run_train_gpt_mlx.sh
# RUN_PRESET=seq2048 RUN_ID=ctx2k ./run_train_gpt_mlx.sh
# RUN_BACKEND=torch NPROC_PER_NODE=1 ./run_train_gpt_mlx.sh
: "${RUN_BACKEND:=auto}"
: "${RUN_PRESET:=${MLX_PRESET:-default}}"
: "${RUN_ID:=mlx_run_$(date +%Y%m%d_%H%M%S)}"
: "${DATA_PATH:=./data/datasets/fineweb10B_sp1024}"
: "${TOKENIZER_PATH:=./data/tokenizers/fineweb_1024_bpe.model}"
: "${VOCAB_SIZE:=1024}"
: "${OUT_DIR:=logs}"

OS_NAME="$(uname -s)"
if [[ "$RUN_BACKEND" == "auto" ]]; then
  if [[ "$OS_NAME" == "Darwin" ]]; then
    RUN_BACKEND="mlx"
  else
    RUN_BACKEND="torch"
  fi
fi

case "$RUN_PRESET" in
  default|seq2048)
    ;;
  *)
    echo "Unknown RUN_PRESET: $RUN_PRESET" >&2
    echo "Expected one of: default, seq2048" >&2
    exit 1
    ;;
esac

case "$RUN_BACKEND" in
  mlx)
    case "$RUN_PRESET" in
      default)
        : "${ITERATIONS:=2000}"
        : "${VAL_LOSS_EVERY:=0}"
        : "${VAL_BATCH_SIZE:=131072}"
        : "${TRAIN_LOG_EVERY:=50}"
        : "${TRAIN_BATCH_TOKENS:=131072}"
        : "${GRAD_ACCUM_STEPS:=8}"
        : "${TRAIN_SEQ_LEN:=1024}"
        : "${MLX_MAX_MICROBATCH_TOKENS:=8192}"
        : "${MLX_EAGER_EVAL:=1}"
        : "${WARMUP_STEPS:=20}"
        : "${MAX_WALLCLOCK_SECONDS:=0}"
        ;;
      seq2048)
        : "${ITERATIONS:=2000}"
        : "${VAL_LOSS_EVERY:=0}"
        : "${VAL_BATCH_SIZE:=131072}"
        : "${TRAIN_LOG_EVERY:=50}"
        : "${TRAIN_BATCH_TOKENS:=131072}"
        : "${GRAD_ACCUM_STEPS:=8}"
        : "${TRAIN_SEQ_LEN:=2048}"
        : "${MLX_MAX_MICROBATCH_TOKENS:=8192}"
        : "${MLX_EAGER_EVAL:=1}"
        : "${WARMUP_STEPS:=20}"
        : "${MAX_WALLCLOCK_SECONDS:=0}"
        ;;
    esac
    export GRAD_ACCUM_STEPS
    export MLX_MAX_MICROBATCH_TOKENS
    export MLX_EAGER_EVAL
    TRAIN_SCRIPT="train_gpt_mlx.py"
    LAUNCH_CMD=(python3 "$TRAIN_SCRIPT")
    EXTRA_SUMMARY="grad_accum_steps=$GRAD_ACCUM_STEPS"
    ;;
  torch)
    case "$RUN_PRESET" in
      default)
        : "${ITERATIONS:=2000}"
        : "${VAL_LOSS_EVERY:=0}"
        : "${VAL_BATCH_SIZE:=524288}"
        : "${TRAIN_LOG_EVERY:=50}"
        : "${TRAIN_BATCH_TOKENS:=524288}"
        : "${TRAIN_SEQ_LEN:=1024}"
        : "${WARMUP_STEPS:=20}"
        : "${MAX_WALLCLOCK_SECONDS:=0}"
        ;;
      seq2048)
        : "${ITERATIONS:=2000}"
        : "${VAL_LOSS_EVERY:=0}"
        : "${VAL_BATCH_SIZE:=524288}"
        : "${TRAIN_LOG_EVERY:=50}"
        : "${TRAIN_BATCH_TOKENS:=524288}"
        : "${TRAIN_SEQ_LEN:=2048}"
        : "${WARMUP_STEPS:=20}"
        : "${MAX_WALLCLOCK_SECONDS:=0}"
        ;;
    esac
    : "${NPROC_PER_NODE:=1}"
    TRAIN_SCRIPT="train_gpt.py"
    if command -v torchrun >/dev/null 2>&1; then
      LAUNCH_CMD=(torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "$TRAIN_SCRIPT")
      EXTRA_SUMMARY="nproc_per_node=$NPROC_PER_NODE launcher=torchrun"
    else
      LAUNCH_CMD=(python3 "$TRAIN_SCRIPT")
      EXTRA_SUMMARY="nproc_per_node=1 launcher=python3"
    fi
    ;;
  *)
    echo "Unknown RUN_BACKEND: $RUN_BACKEND" >&2
    echo "Expected one of: auto, mlx, torch" >&2
    exit 1
    ;;
esac

export RUN_BACKEND
export RUN_PRESET
export RUN_ID
export DATA_PATH
export TOKENIZER_PATH
export VOCAB_SIZE
export ITERATIONS
export VAL_LOSS_EVERY
export VAL_BATCH_SIZE
export TRAIN_LOG_EVERY
export TRAIN_BATCH_TOKENS
export TRAIN_SEQ_LEN
export WARMUP_STEPS
export MAX_WALLCLOCK_SECONDS
export OUT_DIR

echo "Launching $TRAIN_SCRIPT with RUN_ID=$RUN_ID"
echo "  backend=$RUN_BACKEND preset=$RUN_PRESET os=$OS_NAME"
echo "  data_path=$DATA_PATH"
echo "  tokenizer_path=$TOKENIZER_PATH"
echo "  iterations=$ITERATIONS train_batch_tokens=$TRAIN_BATCH_TOKENS train_seq_len=$TRAIN_SEQ_LEN"
echo "  val_loss_every=$VAL_LOSS_EVERY val_batch_size=$VAL_BATCH_SIZE out_dir=$OUT_DIR"
echo "  $EXTRA_SUMMARY"

"${LAUNCH_CMD[@]}" "$@"
