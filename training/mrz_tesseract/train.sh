#!/bin/sh
# Train the project's own Tesseract model for MRZ print (OCR B). Offline once the two inputs are on the machine.
#
# Inputs (see README.md here for where they come from and their terms):
#   BASE   eng.traineddata from tesseract-ocr/tessdata_best (Apache-2.0); the float model, NOT the one apt installs
#   FONT   OCRB.ttf, the upright regular OCR B font
# Needs: tesseract 5 with its training tools (lstmtraining, lstmeval, combine_tessdata), Python with Pillow and NumPy.
#
#   BASE=/path/eng.traineddata FONT=/path/OCRB.ttf training/mrz_tesseract/train.sh /work/mrz [iterations]
#
# Result: /work/mrz/ocrb_int.traineddata and its SHA256SUMS. Copy both into MODELS_DIR/tessdata; the service uses the model
# when the checksum matches. Before replacing the model in use, score both on real documents: tools/accuracy_real.py.
set -eu
: "${BASE:?set BASE}" "${FONT:?set FONT}"
WORK="${1:?work folder}"; ITER="${2:-3000}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$WORK/td/configs" "$WORK/out"; cd "$WORK"
cp "$BASE" td/eng.traineddata
CONFIGS="$(dirname "$(dirname "$(command -v tesseract)")")/share/tessdata/configs/lstm.train"
[ -f "$CONFIGS" ] || CONFIGS="$(ls /usr/share/tesseract-ocr/*/tessdata/configs/lstm.train | head -1)"
cp "$CONFIGS" td/configs/
OCRB_FONT="$FONT" python3 "$HERE/make_lines.py" train 3600 1
OCRB_FONT="$FONT" python3 "$HERE/make_lines.py" eval 300 2          # other seed: never trained on
for f in train/*.png eval/*.png; do tesseract "$f" "${f%.png}" --tessdata-dir "$WORK/td" --psm 13 -l eng lstm.train >/dev/null 2>&1; done
ls "$WORK"/train/*.lstmf > train.list; ls "$WORK"/eval/*.lstmf > eval.list
combine_tessdata -e td/eng.traineddata eng.lstm >/dev/null
echo "base model on the held-out lines:"; lstmeval --model eng.lstm --traineddata td/eng.traineddata --eval_listfile eval.list 2>&1 | tail -1
lstmtraining --continue_from eng.lstm --traineddata td/eng.traineddata --train_listfile train.list --eval_listfile eval.list \
  --model_output out/ocrb --max_iterations "$ITER" --learning_rate 0.0005 > train.log 2>&1
BEST="$(ls out/ocrb_*.checkpoint | sort -t_ -k2 -n | head -1)"
lstmtraining --stop_training --continue_from "$BEST" --traineddata td/eng.traineddata --model_output best.traineddata >/dev/null 2>&1
lstmtraining --stop_training --convert_to_int --continue_from "$BEST" --traineddata td/eng.traineddata --model_output ocrb_int.traineddata >/dev/null 2>&1
echo "trained model on the held-out lines ($BEST):"; lstmeval --model best.traineddata --eval_listfile eval.list 2>&1 | tail -1
sha256sum ocrb_int.traineddata > SHA256SUMS
echo "done: $WORK/ocrb_int.traineddata"
