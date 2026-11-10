"""Engine interface. Vendor and open-source engines plug in behind this."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class EngineUnavailable(RuntimeError):
    """The engine's runtime dependencies (binaries, models, licence) are missing."""


@dataclass
class EngineOutput:
    engine: str
    mrz_lines: list[str] | None = None
    # Extra fields an engine can return directly (visual zone, barcode, etc.).
    fields: dict = field(default_factory=dict)
    confidence: float | None = None
    mrz_confidence: float | None = None   # lowest OCR score among the MRZ lines (None when the engine gives no scores)


class Engine(Protocol):
    name: str

    def extract(self, images: list[bytes], doc_type_hint: str | None) -> EngineOutput:
        """Read one document (one or more images) and return what was found."""
        ...
