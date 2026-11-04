"""Public result types (spec §2.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MoraStatus(str, Enum):
    CORRECT = "correct"
    MISPRONOUNCED = "mispronounced"
    DELETED = "deleted"


class DurationStatus(str, Enum):
    OK = "ok"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"


class PitchStatus(str, Enum):
    OK = "ok"
    TOO_HIGH = "too_high"
    TOO_LOW = "too_low"
    UNKNOWN = "unknown"


class AccentOutcome(str, Enum):
    CORRECT = "correct"
    WRONG_NUCLEUS = "wrong_nucleus"
    FLAT_FOR_ACCENTED = "flat_for_accented"
    FALL_FOR_UNACCENTED = "fall_for_unaccented"
    UNKNOWN = "unknown"


class Warning(str, Enum):
    NO_SPEECH = "no_speech"
    TOO_QUIET = "too_quiet"
    CLIPPING = "clipping"
    LOW_SNR = "low_snr"
    CONTENT_MISMATCH = "content_mismatch"
    PITCH_UNRELIABLE = "pitch_unreliable"
    PARTIAL_READING_UNCERTAIN = "partial_reading_uncertain"
    TRUNCATED = "truncated"


@dataclass
class MoraResult:
    kana: str
    phonemes: list[str]
    start: float | None
    end: float | None
    score: float
    status: MoraStatus
    substituted_by: str | None = None
    duration: DurationStatus | None = None  # only for っ / ん / long vowels
    expected_level: str | None = None  # "H" | "L"
    observed_st: float | None = None  # semitones vs. speaker reference
    pitch_status: PitchStatus | None = None  # only around accent decision points
    devoiced_expected: bool = False


@dataclass
class WordResult:
    surface: str
    reading: str
    score: float
    accent_type: int | None
    accent: AccentOutcome
    start: float | None
    end: float | None
    morae: list[MoraResult] = field(default_factory=list)


@dataclass
class Insertion:
    start: float
    end: float
    content: str | None = None


@dataclass
class ScoreResult:
    overall: float
    accuracy: float | None
    completeness: float | None
    pitch: float | None
    timing: float | None
    confidence: float
    words: list[WordResult] = field(default_factory=list)
    insertions: list[Insertion] = field(default_factory=list)
    warnings: list[Warning] = field(default_factory=list)
    debug: dict | None = None
