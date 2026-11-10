"""Turns uploaded images into a result: run the OCR engine, find the document type, parse, validate, apply the
review policy. Returns ("done", result) or ("failed", error). No web framework code in here."""
from __future__ import annotations

import logging


from . import config
from .engines import EngineUnavailable, get_engine_by_name
from .engines.base import Engine, EngineOutput
from .mrz import MrzError, parse_mrz
from .review import Applicant, apply_mrz_policy, compare_applicant
from .validation import validate
from .visual_zone import parse_gcc_id

log = logging.getLogger("visa-ocr")

_DECODE_ERRORS = ("UnidentifiedImageError", "DecompressionBombError", "DecompressionBombWarning")


def analyse_mrz(lines: list[str]) -> dict:
    """Parse and validate MRZ lines that are already text (no OCR involved)."""
    parsed = parse_mrz(lines)
    flags = validate(parsed)
    return {"document": parsed.to_dict(), "flags": [f.to_dict() for f in flags],
            "needsReview": any(f.severity == "error" for f in flags)}


def _fallback_mrz(images: list[bytes], hint: str | None) -> list[str] | None:
    """Second opinion: the MRZ-only engine is good at the bottom band of a passport or visa."""
    try:
        return get_engine_by_name("tesseract-mrz").extract(images, hint).mrz_lines
    except Exception:  # noqa: BLE001 - optional fallback; never fail the job because of it
        return None


def _fields_out(output: EngineOutput) -> dict:
    fields = dict(output.fields or {})
    if not config.include_text_lines():
        fields.pop("textLines", None)          # raw OCR text is personal data; keep it out of stored results
    return fields


def _no_document_message(hint: str | None, had_text: bool) -> str:
    if hint in ("iqama", "gcc_id") and not had_text:
        return ("This ID has no MRZ. Reading it needs the paddle-ocr engine (OCR_ENGINE=paddle-ocr) "
                "with the English and Arabic models installed.")
    return ("No MRZ and no ID-card fields could be read. For a passport or visa, retake the photo flat and sharp with the "
            "two long lines of '<' at the bottom fully visible. For a card, include the whole front.")


_CARD_FIELDS = ("idNumber", "nameAr", "nameEn", "nationality", "birthDate", "expiryDate")


def _fill_from_second_reader(result: dict, images: list[bytes], engine_name: str) -> dict:
    """One reader often misses what another finds (PaddleOCR drops a small date box; Tesseract garbles a name). When the first
    reading of an ID card lacks a key field, read the text again with Tesseract and take only the missing fields from it.
    Filled fields are flagged, and a card read from printed text always goes to a person anyway."""
    doc = result["document"]
    missing = [k for k in _CARD_FIELDS if not doc.get(k)]
    if doc.get("nationality") and not doc.get("nationalityCode"):
        missing.append("nationality")
    if not missing or engine_name == "tesseract-mrz":
        return result
    try:
        from .engines.tesseract_text import read_text_lines
        second = parse_gcc_id(read_text_lines(images))
    except Exception:  # noqa: BLE001 - the second reader is a bonus; its absence must not break the first reading
        return result
    sdoc, filled = second["document"], []
    for k in missing:
        if sdoc.get(k) and (k != "nationality" or sdoc.get("nationalityCode")):
            doc[k] = sdoc[k]
            filled.append(k)
            cal = k.replace("Date", "Calendar")
            if k.endswith("Date") and sdoc.get(cal):
                doc[cal] = sdoc[cal]
    if not filled:
        return result
    if "nationality" in filled:
        doc["nationalityCode"], doc["nationalityName"] = sdoc.get("nationalityCode", ""), sdoc.get("nationalityName", "")
    result["flags"] = [f for f in result["flags"] if not (f["code"].endswith("_not_found") and f["code"].split("_")[0] in filled)]
    result["flags"].append({"code": "filled_from_second_reader", "severity": "warning",
                            "message": "Fields the first reader missed were taken from a second reading: " + ", ".join(filled) + ". Check them."})
    for k in filled:
        doc.setdefault("fieldConfidence", {})[k] = second["document"].get("fieldConfidence", {}).get(k, 0.5)
    return result


def run(images: list[bytes], hint: str | None, applicant: Applicant, engine: Engine) -> tuple[str, dict]:
    try:
        output = engine.extract(images, hint)
        mrz_confidence = output.mrz_confidence
        if not output.mrz_lines and output.engine != "tesseract-mrz":
            output.mrz_lines = _fallback_mrz(images, hint)
            mrz_confidence = None                      # the fallback reports no scores
        text_lines = (output.fields or {}).get("textLines") if isinstance(output.fields, dict) else None

        if not output.mrz_lines and text_lines:
            result = parse_gcc_id(text_lines)
            result = _fill_from_second_reader(result, images, output.engine)
            doc = result["document"]
            found = [k for k in ("idNumber", "nameAr", "nameEn", "nationality", "occupation", "birthDate", "expiryDate", "employer")
                     if doc.get(k)]
            if "idNumber" in doc or len(found) >= 3:
                a_flags, _ = compare_applicant(doc, applicant)
                result["flags"] += [f.to_dict() for f in a_flags]
                result["needsReview"] = any(f["severity"] == "error" for f in result["flags"])
                result["engine"], result["fields"] = output.engine, _fields_out(output)
                return "done", result
        if not output.mrz_lines:
            return "failed", {"code": "mrz_not_found", "message": _no_document_message(hint, bool(text_lines))}

        result = analyse_mrz(output.mrz_lines)
        apply_mrz_policy(result, mrz_confidence, text_lines, applicant)
        result["engine"], result["fields"] = output.engine, _fields_out(output)
        return "done", result
    except EngineUnavailable as exc:
        return "failed", {"code": "engine_unavailable", "message": str(exc)}
    except MrzError as exc:
        return "failed", {"code": "mrz_unsupported", "message": str(exc)}
    except Exception as exc:  # noqa: BLE001 - never leak image data or internals
        if type(exc).__name__ in _DECODE_ERRORS:
            log.warning("unreadable image: %s", type(exc).__name__)
            return "failed", {"code": "invalid_image", "message": "The image could not be decoded. Upload the photo again."}
        log.exception("processing failed")
        return "failed", {"code": "internal_error", "message": "Processing failed"}
