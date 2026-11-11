# Training the English / Arabic text model

The service ships a **paddle-ocr** engine (set `OCR_ENGINE=paddle-ocr`) that reads English and Arabic text lines
from local model files only. It never downloads anything. Models are tracked in `models/manifest.json`
(`GET /v1/models` lists them with a checksum status).

## Status: what is and is not verified
- Verified by tests: registry and checksums, the engine wiring (with a fake PaddleOCR), consent-gated corrections,
  dataset building, CER evaluation, and the "only activate if CER improves" rule.
- Verified end to end (PaddleOCR main + paddlepaddle 3.3.1, CPU, tiny run): corrections API -> `prepare_dataset.py` ->
  PaddleOCR `tools/train.py` -> `export_model.py` -> `predict_lines.py` -> `evaluate.py` -> `register_model.py` -> the
  engine loads the exported model. This run used **random initial weights and 2 epochs**, so the model was useless
  (CER 2.6); it proves the plumbing and config keys, not model quality.
- NOT verified: fine-tuning from the real pretrained weights (their download hosts were blocked in the test environment;
  get the `*_pretrained.pdparams` for your model from the PaddleOCR docs on a connected machine), accuracy gains, GPU runs.
- Bug found and fixed in this run: the activation safeguard let a new model replace one that had no recorded CER.
  Now record the baseline first: evaluate it on the same real validation file, then
  `register_model.py --set-metrics <baseline id> --metrics baseline.json`.
- Visual-zone field mapping (name, dates, ID number from Arabic text) is not built. Today the engine returns
  text lines; the MRZ stays the source of structured fields.

## Workflow
1. **Get pretrained models** on an internet-connected machine (PP-OCRv5 mobile detection, English and Arabic
   recognition). Copy the folders into `models/` and fill in `path` in `manifest.json`.
2. **Register** them: `python training/register_model.py ...` computes the sha256. The engine refuses missing
   or mismatched models.
3. **Collect data.** Upload with `consent_training=true` and `TRAINING_STORE_DIR` set. A reviewer then posts line
   boxes and text to `POST /v1/jobs/{id}/corrections`. Without consent or without the store, the call returns 409.
   Alternatively label with PPOCRLabel.
4. **Build the dataset**: `training/prepare_dataset.py` (splits by document, caps synthetic data at 3x real,
   warns about characters outside the dictionary). Synthetic Arabic lines come from `training/synth_lines.py`
   (needs an Arabic-capable font; labels are kept in logical order).
5. **Fine-tune** with `training/finetune.sh` (set `EVAL_STEP` low for small datasets, otherwise no best checkpoint is saved) on a GPU server (PaddlePaddle has no Metal GPU support as far as I
   know, so a Mac would be CPU-only and slow). Aim for at least 5,000 labelled lines when you keep the dictionary.
6. **Evaluate on real validation lines only**: `training/predict_lines.py` then `training/evaluate.py`
   (CER, exact match, split by script; `--normalize-arabic`, `--ascii-digits`).
7. **Activate** with `register_model.py --activate`; it refuses unless CER beats the active model (`--force` overrides).

## Privacy
Images are stored only with explicit per-upload consent, with 0700/0600 permissions. The service
deletes training folders older than `TRAINING_RETENTION_DAYS` (default 30; 0 turns this off) in a background sweep.
For a consent withdrawal or a cron job use `python tools/purge_training.py --store-dir <dir> --job-id <id>` or `--days N`. Keep the training store off the
internet-facing host.

## Automatic retraining

`training/auto_train.py` runs the whole cycle on a schedule (cron or a systemd timer on the training server):

```
0 2 * * 0  cd /opt/visa-ocr && PADDLEOCR_DIR=... BASE_CONFIG=... PRETRAINED=... \
  python training/auto_train.py --store-dir "$TRAINING_STORE_DIR" --work-dir /data/autotrain \
  --lang ar --arch arabic_PP-OCRv5_mobile_rec --dict /models/arabic_dict.txt
```

- It retrains only when at least `--min-new` (500) new corrected lines exist and at least `--min-val` (200) real
  validation lines are available.
- It scores the new model and the model in use on the same real validation lines.
- A new model replaces the current one only if its error rate is lower by `--min-gain`, and only when `--activate`
  is given. Without `--activate` it is registered as a candidate and `report.json` says how to approve it.
- A worse model is never activated. Rolling back is `register_model.py --activate` on the older id.
- No restart is needed after a switch: the PaddleOCR engine notices the changed `models/manifest.json` and loads the
  newly active model on the next read. The models folder must be shared with (or copied to) every worker machine.
- Install the weekly schedule with `sudo sh deploy/install_autotrain.sh` (systemd timer, Sunday 02:00; settings in
  `/etc/visa-ocr/autotrain.env`). Not yet run on a real systemd host.
- Where corrections come from: on the `/ui` page, tick "Applicant consented to training use" before reading. After the
  result, the "Corrections for training" panel shows each located line; the reviewer fixes wrong characters and saves.
  Needs `TRAINING_STORE_DIR` set. MRZ line positions come from the second reader on level pages; for card text lines
  set `INCLUDE_TEXT_LINES=1`.

Tested: the decisions (skip, candidate, activate, refuse) with the training step replaced. Not tested: the real
fine-tuning step with production weights on a GPU. Start without `--activate` until a few cycles have been checked by hand.
