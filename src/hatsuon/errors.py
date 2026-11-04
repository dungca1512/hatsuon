"""Hard errors (spec §2.3). Soft conditions are reported as `Warning`s instead."""


class HatsuonError(Exception):
    """Base class for all library errors."""


class AudioError(HatsuonError):
    """Audio cannot be decoded, is empty, or exceeds configured limits."""


class TextError(HatsuonError):
    """Target text is empty/unreadable, or a reading override cannot be aligned."""


class ModelNotFoundError(HatsuonError, FileNotFoundError):
    """The acoustic model has not been exported yet (run `hatsuon-export`)."""
