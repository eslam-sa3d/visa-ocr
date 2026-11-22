"""Turns uploaded images into a result: run the OCR engine, find the document type, parse, validate, apply the
review policy. Returns ("done", result) or ("failed", error). No web framework code in here."""
from __future__ import annotations

from datetime import date

import logging
import time
import traceback

from . import config, quality, timing
from .uploads import shrink_for_ocr
from .engines import EngineUnavailable, get_engine_by_name
from .engines.base import Engine, EngineOutput
from .mrz import MrzError, parse_mrz, plausible_reading
from .review import Applicant, apply_mrz_policy, compare_applicant
from .validation import validate
from .permits import FOUND_FIELDS, parse_permit
from .visual_zone import gulf_evidence, hijri_expiry_flag, parse_gcc_id

log = logging.getLogger("visa-ocr")

def where(exc: BaseException) -> str:
    """'file.py:123 in function' of the innermost frame. Logged instead of a traceback with messages: an error text from a
    date or number parser can repeat a value read from the document, and logs must not hold personal data."""
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return "unknown"
    f = frames[-1]
    return f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno} in {f.name}"


_DECODE_ERRORS = ("UnidentifiedImageError", "DecompressionBombError", "DecompressionBombWarning")


def analyse_mrz(lines: list[str]) -> dict:
    """Parse and validate MRZ lines that are already text (no OCR involved)."""
    parsed = parse_mrz(lines)
    flags = validate(parsed)
    # "mrz" carries the lines exactly as they were read (or pasted), so the caller can store or re-check them. When a look-alike
    # character was corrected, the corrected value is in "document" and the change is listed in document.correctedFields.
    return {"document": parsed.to_dict(), "mrz": {"format": parsed.format, "lines": list(lines)},
            "flags": [f.to_dict() for f in flags], "needsReview": any(f.severity == "error" for f in flags)}


def _fallback_mrz(images: list[bytes], hint: str | None) -> list[str] | None:
    """Second opinion: the MRZ-only engine is good at the bottom band of a passport or visa."""
    try:
        with timing.step("mrzFallback"):
            return get_engine_by_name("tesseract-mrz").extract(images, hint).mrz_lines
    except Exception as exc:  # noqa: BLE001 - optional fallback; never fail the job because of it
        log.warning("MRZ fallback reader failed: %s at %s", type(exc).__name__, where(exc))
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


def _with_retake(result: dict, clarity) -> dict:
    """Mark a reading the applicant should redo with a clearer photo: an MRZ with a failing check digit, or a card whose ID
    number could not be read or fails its own check. The reading is still returned for the reviewer; the portal shows
    result["retake"]["message"] to the applicant. A reading that passes its checks is never marked, however the photo looks."""
    doc = result.get("document") or {}
    checks = doc.get("checks") or {}
    card = doc.get("format") == "visual"
    unclear = any(v is False for v in checks.values()) or (card and not (doc.get("idNumber") or doc.get("documentNumber")))
    if unclear:
        reasons = list(clarity.doubts) + ["card_unclear" if card else "mrz_unclear"]
        result["retake"] = quality.retake(reasons)
        result["flags"].append({"code": "retake_recommended", "severity": "warning", "message": result["retake"]["message"]})
    return result


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
        with timing.step("secondCardReader"):
            second = parse_gcc_id(read_text_lines(images))
    except Exception as exc:  # noqa: BLE001 - the second reader is a bonus; its absence must not break the first reading
        log.warning("second card reader failed: %s at %s", type(exc).__name__, where(exc))
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
                flag = hijri_expiry_flag(doc, date.today(), auto).to_dict()
                if flag["code"] not in codes:
                    result["flags"].append(flag)
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
    """Read the document. The result says how long each step took ("timings", in seconds)."""
    started = time.monotonic()
    with timing.collect() as steps:
        status, result = _run(images, hint, applicant, engine)
    if status == "done" and isinstance(result, dict):
        result["timings"] = {**steps, "total": round(time.monotonic() - started, 2)}
    return status, result


def _run(images: list[bytes], hint: str | None, applicant: Applicant, engine: Engine) -> tuple[str, dict]:
    try:
        images = [shrink_for_ocr(b) for b in images]     # no cost when the API already did it
        # A photo that is plainly unusable (black, blank, badly blurred, tiny) is turned back at once with what to do about
        # it. Reading it would only take longer to reach the same answer.
        with timing.step("clarityCheck"):
            clarity = quality.clearest(images) if config.quality_check() else quality.Quality()
        if clarity.issues:
            return "failed", {"code": "image_unclear", "message": quality.retake_message(clarity.issues),
                              "retake": quality.retake(clarity.issues), "quality": clarity.to_dict()}
        with timing.step("readers"):
            output = engine.extract(images, hint)
        mrz_confidence = output.mrz_confidence
        text_lines = (output.fields or {}).get("textLines") if isinstance(output.fields, dict) else None
        # The MRZ-only search is slow (several seconds when nothing is there). Skip it when the caller said the card has no MRZ and
        # the card text was read, and when the engine already ran that same search itself.
        card_hint = hint in ("iqama", "gcc_id") and bool(text_lines)
        already_searched = (output.engine == "ensemble" and not (hint in ("iqama", "gcc_id"))) or getattr(output, "mrz_searched", False)
        if not output.mrz_lines and output.engine != "tesseract-mrz" and not card_hint and not already_searched:
            output.mrz_lines = _fallback_mrz(images, hint)
            mrz_confidence = None                      # the fallback reports no scores

        if output.mrz_lines and not plausible_reading(output.mrz_lines):
            log.info("lines shaped like an MRZ were not one (no document type letter or dates); treated as no MRZ")
            output.mrz_lines, mrz_confidence = None, None

        if not output.mrz_lines and text_lines:
            # The front of a European, UK or US residence card, or an Omani card: no MRZ and no check digit on that side.
            # Read from its printed labels and always sent to a person. Gulf cards that name another state, or carry Arabic
            # print without naming Oman, are left to the Gulf reader below.
            named_states, _ = gulf_evidence(text_lines)
            permit = None if (named_states - {"OMN"}) else parse_permit(text_lines[:config.max_text_lines()])
            if permit and sum(1 for k in FOUND_FIELDS if permit["document"].get(k)) >= 3:
                a_flags, _ = compare_applicant(permit["document"], applicant)
                permit["flags"] += [f.to_dict() for f in a_flags]
                permit["engine"], permit["fields"] = output.engine, _fields_out(output)
                return "done", _with_retake(permit, clarity)
        if not output.mrz_lines and text_lines:
            result = parse_gcc_id(text_lines[:config.max_text_lines()])      # bounded: parsing cost grows with the line count
            result = _fill_from_second_reader(result, images, output.engine)
            doc = result["document"]
            found = [k for k in ("idNumber", "nameAr", "nameEn", "nationality", "occupation", "birthDate", "expiryDate", "employer")
                     if doc.get(k)]
            # Without an ID number, labelled fields alone make a card only when the text shows it is a Gulf card (a state's name,
            # a Gulf ID label or Arabic print). The front of a European or American permit has the same English labels.
            named, labelled_gulf = gulf_evidence(text_lines)
            # one stray Arabic letter read out of background noise is not Arabic print: ask for a few words' worth
            arabic = sum("\u0600" <= ch <= "\u06ff" for l in text_lines for ch in str(l.get("text", "")))
            gulf = bool(named or labelled_gulf or arabic >= 8)
            if "idNumber" in doc or (len(found) >= 3 and gulf):
                a_flags, _ = compare_applicant(doc, applicant)
                result["flags"] += [f.to_dict() for f in a_flags]
                result["needsReview"] = any(f["severity"] == "error" for f in result["flags"])
                result["engine"], result["fields"] = output.engine, _fields_out(output)
                for key in ("birthCalendar", "expiryCalendar"):
                    if doc.get(key) == "gregorian":
                        del doc[key]          # the calendar is only stated when it is not the usual one (a Hijri date)
                return "done", _with_retake(result, clarity)
        if not output.mrz_lines and card_hint:
            output.mrz_lines = _fallback_mrz(images, hint)      # told "card", but no card fields were read: look for an MRZ after all
            mrz_confidence = None
        if output.mrz_lines and not plausible_reading(output.mrz_lines):
            output.mrz_lines = None
        if not output.mrz_lines:
            reasons = clarity.doubts + ["not_found"]
            return "failed", {"code": "image_unclear" if clarity.doubts else "mrz_not_found",
                              "message": quality.retake_message(reasons) if clarity.doubts
                              else _no_document_message(hint, bool(text_lines)),
                              "retake": quality.retake(reasons)}

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
        wanted = config.include_text_lines() or bool(config.training_store_dir())   # raw text is personal data
        if wanted and boxes and len(boxes) == len(output.mrz_lines):  # the text as read, for the reviewer to correct
            result["fields"]["mrzLineBoxes"] = [{**b, "text": t} for b, t in zip(boxes, output.mrz_lines)]
        else:
            result["fields"].pop("mrzLineBoxes", None)
        return "done", _with_retake(result, clarity)
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
        log.error("processing failed: %s at %s", type(exc).__name__, where(exc))
        return "failed", {"code": "internal_error", "message": "Processing failed"}
