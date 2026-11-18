#!/usr/bin/env bash
# Fine-tune a PaddleOCR text recognition model on the dataset made by prepare_dataset.py.
#
# This wraps PaddleOCR's own tools/train.py and tools/export_model.py. Config keys and file names
# differ between PaddleOCR releases: the keys here were exercised end to end (train, export, predict, evaluate,
# register) on PaddleOCR main with paddlepaddle 3.3.1, CPU. Re-check them if you change versions. Everything runs from local files; nothing is downloaded.
#
# Required environment:
#   PADDLEOCR_DIR   a local checkout of the PaddleOCR repo (same version as the pretrained weights)
#   BASE_CONFIG     the recognition config for the model you start from (inside PADDLEOCR_DIR/configs/rec/...)
#   PRETRAINED      path to the pretrained training weights, without the .pdparams extension
#   DATA_DIR        output folder of prepare_dataset.py (train.txt, val_real.txt, images/)
#   OUT_DIR         where checkpoints and the exported inference model are written
# Optional: EPOCHS (default 30), BATCH (default 64), EVAL_STEP (default 50 steps), USE_GPU (default true), EVAL_BATCH (default 4)
#
# Learning rate: the default schedules are piecewise lists. Set the values in BASE_CONFIG (or a copy of it)
# to suit your batch size; the PaddleOCR fine-tuning guide suggests about [1e-4, 2e-5] for batch 128 on one GPU.
# Keep the original character dictionary unless you must change it: a new dictionary resets the output layer.
set -euo pipefail

: "${PADDLEOCR_DIR:?set PADDLEOCR_DIR}" "${BASE_CONFIG:?set BASE_CONFIG}"
PRETRAINED="${PRETRAINED:-}"
USE_GPU="${USE_GPU:-true}"
# Nothing here may reach the network: the training server is offline, and a silent download attempt would hang the run.
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Training from scratch never gives a usable model, so it is refused unless asked for by name (a pipeline check).
if [ -z "$PRETRAINED" ] && [ "${ALLOW_SCRATCH:-0}" != "1" ]; then
  echo "ERROR: PRETRAINED is not set. Set it to the pretrained training weights, or ALLOW_SCRATCH=1 for a pipeline check only." >&2
  exit 1
fi
# A training checkpoint is loaded as code can be (it is a pickle file), so its checksum is checked before every run.
if [ -n "$PRETRAINED" ]; then
  [ -f "$PRETRAINED.pdparams" ] || { echo "ERROR: $PRETRAINED.pdparams not found" >&2; exit 1; }
  if [ -n "${PRETRAINED_SHA256:-}" ]; then
    GOT="$(sha256sum "$PRETRAINED.pdparams" | cut -d' ' -f1)"
    [ "$GOT" = "$PRETRAINED_SHA256" ] || { echo "ERROR: $PRETRAINED.pdparams does not match PRETRAINED_SHA256" >&2; exit 1; }
  elif [ "${ALLOW_UNVERIFIED_PRETRAINED:-0}" != "1" ]; then
    echo "ERROR: set PRETRAINED_SHA256 to the SHA-256 of $PRETRAINED.pdparams (sha256sum prints it), or ALLOW_UNVERIFIED_PRETRAINED=1." >&2
    exit 1
  fi
fi
: "${DATA_DIR:?set DATA_DIR}" "${OUT_DIR:?set OUT_DIR}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-64}"

[ -s "$DATA_DIR/val_real.txt" ] || { echo "val_real.txt is empty: add real, labelled lines before training" >&2; exit 1; }
mkdir -p "$OUT_DIR"
cd "$PADDLEOCR_DIR"

# Keys below were checked against PaddleOCR main, configs/rec/PP-OCRv5/multi_language/arabic_PP-OCRv5_mobile_rec.yaml:
# these configs use a MultiScaleSampler, so the batch size is Train.sampler.first_bs (not only loader.batch_size_per_card).
python3 tools/train.py -c "$BASE_CONFIG" -o \
  Global.use_gpu="$USE_GPU" Global.distributed=false \
  Global.pretrained_model="$PRETRAINED" \
  Global.save_model_dir="$OUT_DIR/train" \
  Global.epoch_num="$EPOCHS" \
  Global.seed="${SEED:-1}" \
  Global.eval_batch_step="[0,${EVAL_STEP:-50}]" \
  Train.dataset.data_dir="$DATA_DIR" \
  Train.dataset.label_file_list="[$DATA_DIR/train.txt]" \
  Train.sampler.first_bs="$BATCH" \
  Eval.loader.batch_size_per_card="${EVAL_BATCH:-4}" \
  Eval.dataset.data_dir="$DATA_DIR" \
  Eval.dataset.label_file_list="[$DATA_DIR/val_real.txt]"

# best_accuracy exists only if an evaluation ran. With a small dataset the default (every 50 steps) may never fire:
# lower EVAL_STEP so it runs at least once per epoch. Fall back to the last checkpoint rather than failing.
CKPT="$OUT_DIR/train/best_accuracy"
[ -f "$CKPT.pdparams" ] || { echo "WARNING: no best_accuracy (no evaluation ran); exporting latest. Lower EVAL_STEP." >&2; CKPT="$OUT_DIR/train/latest"; }
python3 tools/export_model.py -c "$BASE_CONFIG" -o \
  Global.use_gpu="$USE_GPU" \
  Global.pretrained_model="$CKPT" \
  Global.save_inference_dir="$OUT_DIR/inference"

echo "Exported to $OUT_DIR/inference"
echo "Next: copy it into MODELS_DIR, run training/predict_lines.py + evaluate.py on val_real.txt, then register_model.py"
