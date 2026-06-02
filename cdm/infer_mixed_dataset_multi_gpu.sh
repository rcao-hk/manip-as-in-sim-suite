#!/usr/bin/env bash
set -uo pipefail

# Run CDM inference on 8 datasets with official/DAV2-style preprocessing.
# One GPU runs one worker. Each worker sequentially runs a static subset of tasks.
# This avoids the queue/exit bug where a worker exits after its first task.
#
# Usage:
#   bash run_cdm_official_8datasets_fixed.sh "4"
#   bash run_cdm_official_8datasets_fixed.sh "1,4,5,6"
#
# Common overrides:
#   SCRIPT=infer_mixed_dataset.py \
#   MODEL_PATH=cdm_d435.ckpt \
#   METHOD=cdm_d435_zs_official \
#   bash run_cdm_official_8datasets_fixed.sh "1,4,5,6"

GPU_ARG="${1:-${GPUS:-4}}"
IFS=',' read -r -a GPUS_ARR <<< "$GPU_ARG"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-infer_mixed_dataset.py}"
DATASET_ROOT="${DATASET_ROOT:-/data/robotarm/dataset}"
SPLIT_ROOT="${SPLIT_ROOT:-/home/robotarm/object_depth_percetion/dataset/splits}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/robotarm/result/depth/mixed}"

ENCODER="${ENCODER:-vitl}"
MODEL_PATH="${MODEL_PATH:-cdm_d435.ckpt}"
METHOD="${METHOD:-cdm_d435_zs_official}"
INPUT_SIZE="${INPUT_SIZE:-518}"
MAX_DEPTH="${MAX_DEPTH:-25}"
LOG_DIR="${LOG_DIR:-logs/cdm_official}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Camera choices only affect raw-depth path selection in your inference script.
HAMMER_CAMERA="${HAMMER_CAMERA:-d435}"
TRANSCG_CAMERA="${TRANSCG_CAMERA:-d435}"
DEFAULT_CAMERA="${DEFAULT_CAMERA:-d435}"

# Split filenames. Override if local filenames differ.
HAMMER_SPLIT="${HAMMER_SPLIT:-HAMMER_test.txt}"
HOUSECAT_SPLIT="${HOUSECAT_SPLIT:-HouseCat6D_test.txt}"
TRANSCG_SPLIT="${TRANSCG_SPLIT:-TransCG_d435_test.txt}"
XYZIBD_SPLIT="${XYZIBD_SPLIT:-XYZ-IBD_test.txt}"
YCBV_SPLIT="${YCBV_SPLIT:-YCB-V_test.txt}"
TLESS_SPLIT="${TLESS_SPLIT:-T-LESS_test_primesense.txt}"
GNTRANS_SPLIT="${GNTRANS_SPLIT:-GN-Trans_test.txt}"
ROBI_SPLIT="${ROBI_SPLIT:-ROBI_test.txt}"

mkdir -p "$LOG_DIR"

TASKS=(
  # "HAMMER|${HAMMER_SPLIT}|${HAMMER_CAMERA}"
  # "HouseCat6D|${HOUSECAT_SPLIT}|"
  # "TransCG|${TRANSCG_SPLIT}|${TRANSCG_CAMERA}"
  # "XYZ-IBD|${XYZIBD_SPLIT}|${DEFAULT_CAMERA}"
  # "YCB-V|${YCBV_SPLIT}|${DEFAULT_CAMERA}"
  # "T-LESS|${TLESS_SPLIT}|${DEFAULT_CAMERA}"
  "GN-Trans|${GNTRANS_SPLIT}|${DEFAULT_CAMERA}"
  # "ROBI|${ROBI_SPLIT}|${DEFAULT_CAMERA}"
)

num_tasks="${#TASKS[@]}"
num_gpus="${#GPUS_ARR[@]}"
FAILED_FILE="${LOG_DIR}/failed_${METHOD//\//__}_$(date +%Y%m%d_%H%M%S).txt"
: > "$FAILED_FILE"

if [[ "$num_gpus" -lt 1 ]]; then
  echo "[ERROR] No GPU id provided. Example: bash $0 \"4\""
  exit 1
fi

if [[ ! -f "$SCRIPT" ]]; then
  echo "[ERROR] Inference script not found: $SCRIPT"
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[ERROR] CDM checkpoint not found: $MODEL_PATH"
  echo "        Set MODEL_PATH=/path/to/cdm_d435.ckpt"
  exit 1
fi

echo "[INFO] Script       : $SCRIPT"
echo "[INFO] Dataset root : $DATASET_ROOT"
echo "[INFO] Split root   : $SPLIT_ROOT"
echo "[INFO] Output root  : $OUTPUT_ROOT"
echo "[INFO] Encoder      : $ENCODER"
echo "[INFO] Model path   : $MODEL_PATH"
echo "[INFO] Method       : $METHOD"
echo "[INFO] Input size   : $INPUT_SIZE"
echo "[INFO] Max depth    : $MAX_DEPTH"
echo "[INFO] GPUs         : ${GPUS_ARR[*]}"
echo "[INFO] Num tasks    : $num_tasks"
echo "[INFO] Log dir      : $LOG_DIR"
echo "[INFO] Failed file  : $FAILED_FILE"
echo "[INFO] Extra args   : ${EXTRA_ARGS:-<none>}"
echo "[NOTE] Make sure rgbddepth/dpt.py uses keep_aspect_ratio=True for official CDM preprocessing."

resolve_split_path() {
  local dataset="$1"
  local split_file="$2"
  local split_path="${SPLIT_ROOT}/${split_file}"

  if [[ -f "$split_path" ]]; then
    echo "$split_path"
    return 0
  fi

  if [[ "$dataset" == "HouseCat6D" ]]; then
    local fallback="${SPLIT_ROOT}/housecat6d_test.txt"
    if [[ -f "$fallback" ]]; then
      echo "$fallback"
      return 0
    fi
  fi

  echo "$split_path"
  return 1
}

run_one_task() {
  local gpu="$1"
  local task="$2"
  local dataset split_file camera split_path log_file status

  IFS='|' read -r dataset split_file camera <<< "$task"

  if ! split_path="$(resolve_split_path "$dataset" "$split_file")"; then
    log_file="${LOG_DIR}/${dataset}_${METHOD//\//__}_gpu${gpu}.log"
    echo "[GPU ${gpu}] FAIL  ${dataset}: split file not found: ${split_path}"
    echo "${dataset}|gpu=${gpu}|missing_split=${split_path}" >> "$FAILED_FILE"
    return 0
  fi

  log_file="${LOG_DIR}/${dataset}_${METHOD//\//__}_gpu${gpu}.log"
  echo "[GPU ${gpu}] START ${dataset} -> ${log_file}"

  cmd=(
    "$PYTHON_BIN" "$SCRIPT"
    --encoder "$ENCODER"
    --dataset "$dataset"
    --model-path "$MODEL_PATH"
    --method "$METHOD"
    --dataset_root "$DATASET_ROOT"
    --split "$split_path"
    --output_root "$OUTPUT_ROOT"
    --input-size "$INPUT_SIZE"
    --max-depth "$MAX_DEPTH"
  )

  if [[ -n "$camera" ]]; then
    cmd+=(--camera "$camera")
  fi

  # shellcheck disable=SC2206
  extra_args_arr=( $EXTRA_ARGS )
  if [[ "${#extra_args_arr[@]}" -gt 0 ]]; then
    cmd+=("${extra_args_arr[@]}")
  fi

  {
    echo "[CMD] CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
    echo "[TIME] start: $(date)"
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"
    status=$?
    echo "[TIME] end  : $(date)"
    echo "[EXIT] ${status}"
  } > "$log_file" 2>&1

  status=$?
  if [[ "$status" -eq 0 ]]; then
    echo "[GPU ${gpu}] DONE  ${dataset}"
  else
    echo "[GPU ${gpu}] FAIL  ${dataset} (see ${log_file})"
    echo "${dataset}|gpu=${gpu}|log=${log_file}|status=${status}" >> "$FAILED_FILE"
  fi

  return 0
}

worker() {
  local worker_id="$1"
  local gpu_raw="$2"
  local gpu="${gpu_raw//[[:space:]]/}"
  local idx task

  echo "[GPU ${gpu}] worker ${worker_id}/${num_gpus} started."

  idx="$worker_id"
  while (( idx < num_tasks )); do
    task="${TASKS[$idx]}"
    run_one_task "$gpu" "$task"
    idx=$((idx + num_gpus))
  done

  echo "[GPU ${gpu}] worker finished."
}

pids=()
for i in "${!GPUS_ARR[@]}"; do
  worker "$i" "${GPUS_ARR[$i]}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid" || true
done

if [[ -s "$FAILED_FILE" ]]; then
  echo "[ERROR] Some CDM inference tasks failed:"
  cat "$FAILED_FILE"
  exit 1
fi

echo "[DONE] All CDM official inference tasks finished."
