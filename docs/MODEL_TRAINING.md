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
For a consent withdrawal or a cron job use `python tools/purge_training.py --store-dir <dir> --job-id <id> --work-dir <work dir>` or `--days N`. Datasets built for training follow the same rule: `auto_train.py` deletes its line crops, label files and predictions when a run ends and removes run folders older than `--keep-days` (30); everything it writes is owner-only. A model already trained on withdrawn lines is not changed; decide separately whether a withdrawal requires retraining. For folders made before this version run `chmod -R go-rwx <work dir>` once. Keep the training store off the
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
- A worse model is never activated. Rolling back is one command: `python training/register_model.py --rollback --task text_recognition --lang ar`.
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

## Controls on the learning loop (added after the engineering and ML review)

**1. Labels are approved by a second person.** A correction sent to `/v1/jobs/{id}/corrections` is stored as `pending`, one
file per submission under `jobs/<id>/corrections/` (nothing is overwritten). It becomes training data only when a different
person approves it on the server:

```
python tools/approve_corrections.py --store-dir "$TRAINING_STORE_DIR" --list            # least confident readings first
python tools/approve_corrections.py --store-dir "$TRAINING_STORE_DIR" --show JOB_ID --crops-dir ~/crops-JOB_ID
python tools/approve_corrections.py --store-dir "$TRAINING_STORE_DIR" --approve JOB_ID --correction DIGEST --approver "Name"
```

`--show` prints each correction with a digest in square brackets, and writes the image crop each line's box covers to a
private folder: look at the crops, not only at the text and the box numbers. `--approve` decides only the correction whose
digest you give, and only if it is unchanged since you looked; a correction posted afterwards stays pending. The digests of
the crops you were shown are recorded with the approval, and a line whose crop differs is never used. Stored text is shown
with every control or formatting character as a visible escape. Delete the crops folder when you are done.

The submitter cannot approve their own correction. Whatever is approved, the dataset builder still drops a correction whose
MRZ lines fail their own check digits, malformed records, lines in another language than the one being trained, duplicate
crops and any crop that is also in the golden set. `stats.json` and the run report count each kind.

**2. A fixed golden set judges every model.** Create a folder outside the training store with `labels.txt`
(`<image>\t<text>` per line) and the line images. Name the files `<type>__<n>.png` (`passport__0001.png`,
`iqama__0001.png`) to get scores per document type. It is never used for training and is not touched by the retention purge,
so it needs its own legal basis and access control. Pass it with `--golden-dir`. Without it a candidate is built and
registered, but never compared and never activated.

**3. The gate fails closed.** A candidate replaces the model in use only when, on the golden set: its CER is lower by at
least `--min-gain` and by more than the measurement noise; no script and no document type with 20 or more golden lines got
worse; no fewer lines are read exactly right; and both models could be scored. A line with no prediction counts as fully
wrong. The reasons for a refusal are in `report.json` under `gate.reasons`.

**4. The switch is safe and can be undone.** The manifest is replaced whole under a lock. A running server loads the newly
active model, tries it on a blank image, and only then starts using it; if that fails it keeps the model it has and logs
`model switch refused`. To go back: `register_model.py --rollback --task text_recognition --lang ar`. To put a registered
candidate in use by hand: `register_model.py --activate-existing ID`.

**5. The automation cannot run away.** Training that fails twice on the same data is not tried again until the data
changes. A lock left by a dead run is cleared. Checkpoints are deleted after export and only the newest
`--keep-candidates` unused candidates are kept. The systemd unit limits CPU, memory and disk priority. Every run adds a line
to `history.jsonl` with the dataset checksum, seed, settings and package versions.

**6. Learning can carry over.** By default each run starts from the pretrained weights and sees only the corrections still
inside the retention period. With `--continue-from-current` it starts from the checkpoint of the model in use (when this
tool trained it), so gains accumulate. Turn it on only with a golden set in place: the golden set is what detects a model
that forgot older document types.

**7. Check the confidence thresholds against outcomes.** Each correction records the reader's confidence and which fields
the reviewer changed. `python tools/calibration_report.py --store-dir "$TRAINING_STORE_DIR"` prints the error rate per
confidence band and how many readings accepted without review were wrong.

Still not done anywhere: a real fine-tune with production weights on the training server, and shadow scoring of a candidate
on live traffic before activation.
