# Runbook: PaddleOCR models, fine-tuning and real-specimen measurement

For checklist items B1, B2, B3, C1, C2 and the go-live gate. These steps need an internet-connected machine (model download)
and, for training, a GPU server. The production host stays offline. Nothing here has been run end to end with real weights:
the first run will need small fixes (PaddleOCR config names change between releases). Times are rough.

## Part 1. Download the models (B1, B2) - 30 minutes, connected laptop or server
1. `python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
2. `python tools/fetch_models.py` - downloads PP-OCRv5 mobile detection, English and Arabic recognition into `models/` and writes
   their SHA-256 into `models/manifest.json`.
3. Check: `ls models/` shows three folders; `python -c "import json;print([m['sha256'] is not None for m in json.load(open('models/manifest.json'))['models']])"` prints `True` for each.
4. Copy the whole `models/` folder to the offline server (same path as `MODELS_DIR`).

## Part 2. Switch the offline server to PaddleOCR and measure (B1, B2) - 1 hour
1. Set `OCR_ENGINE=paddle-ocr` (or leave `auto`: it tries PaddleOCR first, then the Tesseract + RapidOCR ensemble).
2. `curl http://localhost:8080/v1/models` - every model must show a good checksum.
3. `python tools/run_scenarios.py --engine paddle-ocr` and `python tools/accuracy_gate.py --n 2`. Compare with the ensemble
   numbers in `docs/ACCURACY_REVIEW.md` (91% fully correct, 42% accepted without review, 0 silent wrong).
4. Paddle gives a score per line, so `MIN_CONFIDENCE_UNPROTECTED` (default 0.97) now decides what skips review. Calibrate it on real
   specimens (Part 5), never on synthetic ones. Keep the ensemble as the fallback.
5. Test Arabic: the Iqama scenarios (`--only iqama`) and then real cards. Expect Paddle's Arabic to beat Tesseract's.

## Part 3. Collect real specimens (C1, C2) - 1 to 2 weeks, the long pole
1. `python tools/specimens.py init real-samples` (1,200 specimens planned: 30 per type and country group).
2. Collect with written consent from each holder (or your own documents). Take them the way applicants will: phone photo, scan,
   glare, folded, low light. Include the back of ID cards.
3. For each: `python tools/specimens.py add real-samples photo.jpg --type passport:TD3 --country EGY --condition phone --doc-no ... --birth YYYY-MM-DD --expiry YYYY-MM-DD --surname ... --given ... --nat EGY`.
   Type the values from the PAPER. Do not copy what the service returned. (`specimens.py draft` can pre-fill guesses to save typing; every line it fills must be checked.)
4. `python tools/specimens.py status real-samples` shows what is still missing.
5. Keep the folder off the internet-facing host and out of git. Delete images when the campaign ends.
Priority order: Saudi, Egyptian, Indian, Pakistani, Filipino passports; Schengen visas; green cards; Iqama (new and old); Saudi national ID; then other GCC IDs.

## Part 4. Fine-tune the recognition model (B3) - 2 to 5 days on a GPU server
Start only when you have real labelled lines (aim for 5,000). Fine-tuning on synthetic lines alone makes the model worse on real documents.
1. **Pretrained weights.** On the connected machine, from the PaddleOCR docs, download the training weights (`*_pretrained.pdparams`) that match
   the model you start from (for Arabic: `arabic_PP-OCRv5_mobile_rec`). Clone PaddleOCR at the version that matches those weights.
2. **Labelled lines.** Two sources: reviewer corrections (upload with `consent_training=true`, `TRAINING_STORE_DIR` set, reviewers post to
   `POST /v1/jobs/{id}/corrections`) and PPOCRLabel exports of the real specimens.
3. **Synthetic MRZ lines.** `python training/synth_mrz.py --out data/synth_mrz --count 20000 --fonts /path/OCRB.ttf /usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf`
   Use a real OCR-B font if you can obtain one; the MRZ of every passport is printed in OCR-B. Arabic lines: `training/synth_lines.py`.
4. **Dataset.** `python training/prepare_dataset.py --store-dir <TRAINING_STORE_DIR> --ppocrlabel-dir <exports> --synth-dir data/synth_mrz data/synth_ar --dict <model dictionary> --out data/rec`
   It splits by document, caps synthetic lines at 3x the real ones, and keeps real validation lines separate (`val_real.txt`).
5. **Baseline first.** Predict and evaluate the CURRENT model on `val_real.txt`:
   `python training/predict_lines.py --labels data/rec/val_real.txt --image-dir data/rec --model-name <arch> --model-dir models/<current> --out base_preds.txt`
   `python training/evaluate.py --labels data/rec/val_real.txt --predictions base_preds.txt --out baseline.json`
   `python training/register_model.py --set-metrics <current id> --metrics baseline.json`
6. **Train.** `PADDLEOCR_DIR=... BASE_CONFIG=configs/rec/PP-OCRv5/multi_language/arabic_PP-OCRv5_mobile_rec.yaml PRETRAINED=<weights without .pdparams> DATA_DIR=data/rec OUT_DIR=out/ar1 EPOCHS=30 EVAL_STEP=50 USE_GPU=true bash training/finetune.sh`
   Set the learning rate in the config (about 1e-4 falling to 2e-5 for batch 128). Keep the original dictionary.
7. **Evaluate the new model** the same way as step 5 on the same `val_real.txt`.
8. **Register and activate.** Copy `out/ar1/inference` into `models/ar_rec_ft_<date>`, then
   `python training/register_model.py --id ar_rec_ft_<date> --task text_recognition --lang ar --version ft-<date> --path ar_rec_ft_<date> --arch arabic_PP-OCRv5_mobile_rec --metrics metrics.json --activate`
   It refuses unless CER on real lines beats the active model. Roll back by activating the older id.
9. Repeat monthly with new corrections.

## Part 5. Measure on real specimens and decide (C4, E2)
1. `python tools/run_scenarios.py --real-dir real-samples` (writes `real-samples/report.json`).
2. `python tools/accuracy_gate.py --real-dir real-samples` - applies `config/acceptance.json` (95% fully correct per family, 0 silent wrong,
   Iqama key fields 90%, at least 30 specimens per family). Exit code 1 means not ready.
3. Calibrate confidence: for the accepted-without-review documents, count wrong unprotected fields. If any are wrong, raise
   `MIN_CONFIDENCE_UNPROTECTED` or `AGREEMENT_CONFIDENCE` consequences accordingly. Keep a person sampling accepted documents.
4. Get the business owner's written sign-off on the numbers in `config/acceptance.json` (E2), then change its `status`.
5. For other GCC IDs: after 20+ real front and back images per card, add label wording to `config/id_labels.json`, extend `app/gcc_ids.py`
   with the verified number rule, and re-run step 1 until the family passes.

## If something fails
- Checksum mismatch on a model: re-copy the folder, or re-run `tools/fetch_models.py`; never edit the manifest by hand to silence it.
- `finetune.sh` config key errors: PaddleOCR renamed keys; compare with the `-o` keys in the script and the yaml of your PaddleOCR version.
- New model is worse on real lines: do not use `--force`; add more real lines, lower synthetic share, lower the learning rate.
