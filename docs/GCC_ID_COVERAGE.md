# GCC ID coverage

| Document | Status | Evidence |
|---|---|---|
| Saudi Iqama (front) | Supported | 1 real card + 2 synthetic layouts |
| Saudi national ID | Parsed by the same label rules, **not tested** | no specimen |
| Iqama / ID back side | Not supported | no specimen |
| UAE Emirates ID, Kuwait civil ID, Qatar ID, Bahrain CPR | **Recognised from the ID number only** (`app/gcc_ids.py`: Luhn for UAE, birth date and weak check for Kuwait, structure for Qatar and Bahrain). Always `needsReview`, flag `gcc_layout_unverified`. Layouts not tested | no specimens |
| Oman ID | **Not supported** (no public number structure) | no specimens |
| UAE/Qatar IDs with an MRZ on the back | Work through the MRZ path | none |

Every visual-zone result is returned with `needsReview: true` until accuracy is measured on real cards
(`VISUAL_ZONE_AUTO_ACCEPT=false` is the default).

## Adding a new ID layout
1. Collect at least 20 real front and back images (with consent). Do not guess layouts.
2. Add its label wording to `config/id_labels.json` (no code change; or set `ID_LABELS_FILE`).
3. Add the images to a scenario folder with expected values, run `python tools/run_scenarios.py`.
4. Keep review mandatory until the measured wrong-field rate is accepted by the business owner.

Note: the Saudi ID / Iqama check digit (Luhn variant) is not officially published. It is used as a warning signal only.

## Specimen checklist (what to collect, then one command to measure)
For each document type (Saudi national ID, UAE Emirates ID, Kuwait civil ID, Qatar ID, Bahrain CPR, Oman ID, and the
back of the Iqama):
- 20 or more real cards, front and back, from different people and print runs (old and new designs).
- Photos as users will take them: phone camera, angled, glare, scans, low light. Not only clean scans.
- Consent from each holder; store the images on the secured training host only.
- An `expected.csv` with the true values (see `tools/specimen_eval.py` header).

Then: `python tools/specimen_eval.py --dir specimens --expected specimens/expected.csv`.
It prints per-field accuracy, the share sent to review, and silent-wrong cases. Add label wording to
`config/id_labels.json` for any field that is missed, re-run, and repeat. Keep `VISUAL_ZONE_AUTO_ACCEPT=false` until
the silent-wrong rate is accepted in writing. A new country usually also needs code: its date format, ID-number rule
and name layout. That work cannot start without the specimens.
