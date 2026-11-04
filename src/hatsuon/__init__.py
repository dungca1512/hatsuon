"""hatsuon — Japanese pronunciation scoring."""

from .api import score
from .errors import AudioError, HatsuonError, ModelNotFoundError, TextError
from .types import ScoreResult

__version__ = "0.1.1"
__all__ = ["AudioError", "HatsuonError", "ModelNotFoundError", "ScoreResult", "TextError", "__version__", "score"]
