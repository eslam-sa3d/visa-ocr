"""Turns uploaded images into a result: run the OCR engine, find the document type, parse, validate, apply the
review policy. Returns ("done", result) or ("failed", error). No web framework code in here."""
from __future__ import annotations

from datetime import date

import os

import logging


from . import config
from .uploads import shrink_for_ocr
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


def _small_image_note(images: list[bytes]) -> str | None:
    """Advice for the reviewer when a page is too small for its print to be read reliably."""
    import io
    from PIL import Image
    for raw in images:
        try:
            w, h = Image.open(io.BytesIO(raw)).size
        except Exception:  # noqa: BLE001
            continue
        if max(w, h) < 1300:
            return (f"The image is small ({w} x {h} pixels), so the MRZ letters are only a few pixels high and are easily misread. "
                    "Scan at 200 dpi or more, or photograph only the data page so it fills the frame.")
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
    # "ensemble" already took its text from this same Tesseract reader: a second pass would return the same lines at twice the cost
    if not missing or engine_name in ("tesseract-mrz", "ensemble"):
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
    # The first reading was validated before these values existed, so validate them now. With auto-accept on, a filled field
    # always reaches a person: it came from a second reader and nothing else confirms it.
    auto = config.visual_zone_auto_accept()
    codes = {f["code"] for f in result["flags"]}
    result["flags"].append({"code": "filled_from_second_reader", "severity": "error" if auto else "warning",
                            "message": "Fields the first reader missed were taken from a second reading: " + ", ".join(filled) + ". Check them."})
    try:
        if "expiryDate" in filled:
            if doc.get("expiryCalendar") != "gregorian":
                if "expiry_hijri_not_checked" not in codes:
                    result["flags"].append({"code": "expiry_hijri_not_checked", "severity": "error" if auto else "warning",
                                            "message": "Expiry is a Hijri date; expiry check needs the Gregorian date"})
            elif date.fromisoformat(doc["expiryDate"]) < date.today() and "document_expired" not in codes:
                result["flags"].append({"code": "document_expired", "severity": "error", "message": "The document has expired"})
        if "birthDate" in filled and doc.get("birthCalendar") == "gregorian" and date.fromisoformat(doc["birthDate"]) > date.today():
            result["flags"].append({"code": "birth_date_in_future", "severity": "error", "message": "Date of birth is in the future"})
    except ValueError:
        result["flags"].append({"code": "filled_date_invalid", "severity": "error", "message": "A filled date is not a valid date"})
    s_scores = second["document"].get("fieldConfidence", {})
    for k in filled:
        doc.setdefault("fieldConfidence", {})[k] = s_scores.get(k, 0.5)
    low = sorted(k for k in filled if s_scores.get(k, 0.5) < config.min_confidence_visual())
    if low:
        result["flags"].append({"code": "low_confidence_fields", "severity": "error",
                                "message": "Low OCR confidence for filled fields: " + ", ".join(low)})
    return result


def run(images: list[bytes], hint: str | None, applicant: Applicant, engine: Engine) -> tuple[str, dict]:
    try:
        images = [shrink_for_ocr(b) for b in images]     # no cost when the API already did it
        output = engine.extract(images, hint)
        mrz_confidence = output.mrz_confidence
        text_lines = (output.fields or {}).get("textLines") if isinstance(output.fields, dict) else None
        # The MRZ-only search is slow (several seconds when nothing is there). Skip it when the caller said the card has no MRZ and
        # the card text was read, and when the engine already ran that same search itself.
        card_hint = hint in ("iqama", "gcc_id") and bool(text_lines)
        already_searched = output.engine == "ensemble" and not (hint in ("iqama", "gcc_id"))
        if not output.mrz_lines and output.engine != "tesseract-mrz" and not card_hint and not already_searched:
            output.mrz_lines = _fallback_mrz(images, hint)
            mrz_confidence = None                      # the fallback reports no scores

        if not output.mrz_lines and text_lines:
            result = parse_gcc_id(text_lines[:config.max_text_lines()])      # bounded: parsing cost grows with the line count
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
        if not output.mrz_lines and card_hint:
            output.mrz_lines = _fallback_mrz(images, hint)      # told "card", but no card fields were read: look for an MRZ after all
            mrz_confidence = None
        if not output.mrz_lines:
            return "failed", {"code": "mrz_not_found", "message": _no_document_message(hint, bool(text_lines))}

        result = analyse_mrz(output.mrz_lines)
        apply_mrz_policy(result, mrz_confidence, text_lines, applicant, mrz_lines=output.mrz_lines,
                         agreed=(output.fields or {}).get("agreedFields"))
        result["engine"], result["fields"] = output.engine, _fields_out(output)
        if (output.fields or {}).get("readers"):
            result["readers"] = output.fields["readers"]
        result["fields"].pop("readers", None)
        if "optionalData" in (result["fields"].pop("unconfirmedFields", None) or []):
            # The two readers agree on everything else. This field is shown, marked unverified, and does not force a review
            # unless OPTIONAL_DATA_REVIEW=1: the portal does not use a passport's personal number for a decision.
            strict = config.optional_data_review()
            result.setdefault("fieldStatus", {})["optionalData"] = "unverified"
            result["flags"].append({"code": "optional_data_unconfirmed", "severity": "error" if strict else "warning",
                                    "message": "The two readers differ inside the optional data (personal number) at characters its "
                                               "check digit cannot tell apart, such as 6/G. All other fields agree. Do not rely on "
                                               "the optional data without checking it."})
            if strict:
                result["needsReview"] = True
        result["fields"].pop("mrzLine2", None)
        result["fields"].pop("agreedFields", None)
        if result["fields"].pop("numberDisputed", None):
            result["flags"].append({"code": "document_number_readers_differ", "severity": "error",
                                    "message": "The two readers read different document numbers and both pass the check digit. "
                                               "Confirm the number against the document."})
            result["needsReview"] = True
        if result.get("needsReview"):
            small = _small_image_note(images)
            if small:
                result["flags"].append({"code": "image_low_resolution", "severity": "warning", "message": small})
        boxes = result["fields"].get("mrzLineBoxes")
        wanted = config.include_text_lines() or bool(os.environ.get("TRAINING_STORE_DIR"))   # raw text is personal data
        if wanted and boxes and len(boxes) == len(output.mrz_lines):  # the text as read, for the reviewer to correct
            result["fields"]["mrzLineBoxes"] = [{**b, "text": t} for b, t in zip(boxes, output.mrz_lines)]
        else:
            result["fields"].pop("mrzLineBoxes", None)
        return "done", result
    except EngineUnavailable as exc:
        return "failed", {"code": "engine_unavailable", "message": str(exc)}
    except MrzError as exc:
        return "failed", {"code": "mrz_unsupported", "message": str(exc)}
    except Exception as exc:  # noqa: BLE001 - never leak image data or internals
        broken = isinstance(exc, (OSError, SyntaxError, ValueError)) and any(
            w in str(exc).lower() for w in ("truncated", "broken data stream", "cannot identify image", "image file", "decoder"))
        if type(exc).__name__ in _DECODE_ERRORS or broken:     # a cut-off or damaged upload is the caller's file, not a fault here
            log.warning("unreadable image: %s", type(exc).__name__)
            return "failed", {"code": "invalid_image", "message": "The image could not be decoded. Upload the photo again."}
        log.exception("processing failed")
        return "failed", {"code": "internal_error", "message": "Processing failed"}
