"""Mora segmentation and kana → phoneme mapping (spec §4.1.3).

Phoneme symbols follow the OpenJTalk inventory (as produced by `pyopenjtalk.g2p`)
so that dictionary output and reading overrides share one symbol set:
vowels `a i u e o`, moraic nasal `N`, geminate closure `cl`, consonants such as
`k g s sh z j t ch ts d n h f b p m y r w v` and palatalized `ky gy ny hy by py my ry`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Small kana that merge with the preceding kana into one mora. っ is NOT here.
_ATTACH = frozenset("ャュョァィゥェォヮ")
_HIRA_START, _HIRA_END = 0x3041, 0x3096  # ぁ..ゖ
_KATA_OFFSET = 0x60

SOKUON = "ッ"
HATSUON = "ン"
CHOON = "ー"


def to_katakana(text: str) -> str:
    """Convert hiragana to katakana; other characters are left unchanged."""
    return "".join(
        chr(ord(c) + _KATA_OFFSET) if _HIRA_START <= ord(c) <= _HIRA_END else c for c in text
    )


def _is_katakana(c: str) -> bool:
    return "ァ" <= c <= "ヺ" or c == CHOON


def split_morae(kana: str) -> list[str]:
    """Split a kana string into morae (katakana).

    Small ャュョァィゥェォヮ attach to the preceding kana; ッ, ン and ー are morae on
    their own. Raises ValueError on any non-kana character, since the front-end is
    expected to have converted everything to kana before this point.
    """
    out: list[str] = []
    for c in to_katakana(kana):
        if not _is_katakana(c):
            raise ValueError(f"not a kana character: {c!r} in {kana!r}")
        if c in _ATTACH and out:
            out[-1] += c
        else:
            out.append(c)
    return out


# --- kana → (consonant, vowel) ----------------------------------------------

def _row(cons: str | None, kana: str, vowels: str = "aiueo") -> dict[str, tuple[str | None, str]]:
    return {k: (cons, v) for k, v in zip(kana, vowels)}


_BASIC: dict[str, tuple[str | None, str]] = {
    **_row(None, "アイウエオ"),
    **_row(None, "ァィゥェォ"),
    **_row("k", "カキクケコ"),
    **_row("g", "ガギグゲゴ"),
    **_row("s", "サ ス セソ".replace(" ", ""), "aueo"),
    "シ": ("sh", "i"),
    **_row("z", "ザ ズゼゾ".replace(" ", ""), "aueo"),
    "ジ": ("j", "i"),
    **_row("t", "タテト", "aeo"),
    "チ": ("ch", "i"),
    "ツ": ("ts", "u"),
    **_row("d", "ダデド", "aeo"),
    "ヂ": ("j", "i"),
    "ヅ": ("z", "u"),
    **_row("n", "ナニヌネノ"),
    **_row("h", "ハヒヘホ", "aieo"),
    "フ": ("f", "u"),
    **_row("b", "バビブベボ"),
    **_row("p", "パピプペポ"),
    **_row("m", "マミムメモ"),
    **_row("y", "ヤユヨ", "auo"),
    **_row("y", "ャュョ", "auo"),
    **_row("r", "ラリルレロ"),
    "ワ": ("w", "a"),
    "ヮ": ("w", "a"),
    "ヲ": (None, "o"),
    "ヰ": (None, "i"),
    "ヱ": (None, "e"),
    "ヴ": ("v", "u"),
    "ヵ": ("k", "a"),
    "ヶ": ("k", "e"),
}

# Consonant of a Ci kana → palatalized consonant before ャュョ / ェ.
_PALATAL = {
    "k": "ky", "g": "gy", "n": "ny", "h": "hy", "b": "by", "p": "py", "m": "my", "r": "ry",
    "sh": "sh", "j": "j", "ch": "ch",
}
# Special loanword palatals: テュ, デュ, フュ, ヴュ.
_PALATAL_E_U = {"t": "ty", "d": "dy", "f": "fy", "v": "by"}
# Labialized onsets before ヮ (クヮ, グヮ).
_LABIAL = {"k": "kw", "g": "gw"}


class MoraKind(str, Enum):
    REGULAR = "regular"
    SOKUON = "sokuon"  # っ
    HATSUON = "hatsuon"  # ん
    LONG = "long"  # ー, or a bare vowel repeating the previous vowel (おかあさん)


@dataclass(frozen=True)
class Mora:
    kana: str
    phonemes: tuple[str, ...]
    kind: MoraKind

    @property
    def vowel(self) -> str | None:
        last = self.phonemes[-1] if self.phonemes else None
        return last if last in ("a", "i", "u", "e", "o") else None


def _combine(base: str, small: str) -> tuple[str, ...]:
    bc, bv = _BASIC[base]
    _, sv = _BASIC[small]
    if small == "ヮ":  # クヮ, グヮ (archaic)
        return (_LABIAL[bc], "a") if bc in _LABIAL else tuple(x for x in (bc, "w", "a") if x)
    if small in "ャュョ":
        if bv == "i" and bc in _PALATAL:
            return (_PALATAL[bc], sv)
        if bc in _PALATAL_E_U:
            return (_PALATAL_E_U[bc], sv)
        return tuple(x for x in (bc, "y", sv) if x)
    # small vowel ァィゥェォ
    if bc is None:
        if bv == "u":  # ウィ, ウェ, ウォ
            return ("w", sv)
        if bv == "i" and sv == "e":  # イェ
            return ("y", "e")
        return (sv,)
    if bv == "i" and sv == "e" and bc in _PALATAL:  # シェ, ジェ, チェ, キェ...
        return (_PALATAL[bc], "e")
    return (bc, sv)  # ファ, ティ, ディ, トゥ, ヴァ, ツァ ...


def mora_phonemes(mora: str, prev_vowel: str | None) -> tuple[str, ...]:
    """Phoneme sequence of one mora. ー inherits the previous vowel (or none)."""
    if mora == SOKUON:
        return ("cl",)
    if mora == HATSUON:
        return ("N",)
    if mora == CHOON:
        return (prev_vowel,) if prev_vowel else ()
    if len(mora) == 1:
        cons, vowel = _BASIC[mora]
        return (cons, vowel) if cons else (vowel,)
    if len(mora) == 2:
        return _combine(mora[0], mora[1])
    raise ValueError(f"not a single mora: {mora!r}")


def to_morae(kana: str, prev_vowel: str | None = None) -> list[Mora]:
    """Segment kana into `Mora` objects with phonemes and special-mora kind.

    `prev_vowel` is the vowel the text ends on before `kana` (the previous word),
    so a word-initial ー inherits it: OpenJTalk splits であろう as であろ|う(ー).
    """
    out: list[Mora] = []
    for m in split_morae(kana):
        ph = mora_phonemes(m, prev_vowel)
        if m == SOKUON:
            kind = MoraKind.SOKUON
        elif m == HATSUON:
            kind = MoraKind.HATSUON
        elif m == CHOON or (len(ph) == 1 and ph[0] == prev_vowel):
            kind = MoraKind.LONG
        else:
            kind = MoraKind.REGULAR
        out.append(Mora(m, ph, kind))
        vowel = out[-1].vowel
        # っ / ん break vowel continuity; ー keeps the previous vowel.
        prev_vowel = vowel if m != CHOON else prev_vowel
    return out
