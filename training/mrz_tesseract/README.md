# The project's own Tesseract model for the MRZ

`models/tessdata/ocrb_int.traineddata` is trained by this project. It replaces a third-party OCR-B model that was tested
earlier and never shipped, because its repository states no licence.

## What it is made from

| Input | Source | Terms |
| --- | --- | --- |
| Base model `eng.traineddata` (float) | github.com/tesseract-ocr/tessdata_best | Apache-2.0 |
| Font `OCRB.ttf` (OCR B, regular) | `ocr-0.3.1.zip` from tsukurimashou.org/ocr.php (Matthew Skala, after Norbert Schwarz); SHA-256 of the zip `58136fccfdee0923cc83a20996a067b98bae054570ee41bf896d7ca8224399bf` | Its documentation says Schwarz first limited it to non-commercial use and "has since released it for unrestricted use and distribution"; Skala makes no copyright claim and gives no legal opinion. The font is NOT shipped: it was only used to draw training lines. If your organisation needs a formal position, ask legal to read section "OCR B" of `ocr.tex` in that zip. |
| Training text | Generated here: valid MRZ lines for passports, visas and ID cards, plus random characters | Synthetic. No real document was used for training. |

## How it was trained (18 Nov 2026)

3,600 generated lines (seed 1) for training and 300 (seed 2) held out; each line drawn in OCR B and degraded like a scan or a
phone photo (low resolution, blur, noise, security-print background, tilt, ink weight, JPEG). Fine-tuned from the base model
for 3,000 iterations at learning rate 0.0005 with Tesseract 5.3.4; best checkpoint at iteration 2,600. `train.sh` repeats it.

## Measured

| | Base model | This model |
| --- | --- | --- |
| Character error on the 300 held-out lines | 55.3% | 0.32% |

On 16 real passports (never used in training), whole service, Tesseract + RapidOCR engine:

| Tesseract model | Fully correct | Accepted without review | Wrongly accepted |
| --- | --- | --- | --- |
| Stock `eng`, no time limit | 13 | 5 | 0 |
| This model, no time limit | 14 | 10 | 0 |
| Stock `eng`, 1.7 s limit | 11 | 3 | 0 |
| This model, 1.7 s limit | 14 | 9 | 0 |

Sixteen documents is a small sample. Score any retrained model on your full specimen set before it replaces this one.
