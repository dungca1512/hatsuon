"""Text analysis: words, readings, accent phrases (spec §4.1.1, §4.1.2).

Uses pyopenjtalk (extra `frontend`). Facts relied on, verified against pyopenjtalk 0.4:
- `run_frontend()` yields one NJD node per word with `pron` (pronunciation kana,
  long vowels as ー, particles は/へ/を as ワ/エ/オ) and `read` (orthographic kana).
- A `’` in `pron` follows a mora whose vowel is predicted devoiced (マス’ → m a s U).
- An accent phrase starts at a node with `chain_flag != 1`; following nodes with
  `chain_flag == 1` attach to it. The head node's `acc` is the phrase nucleus after
  accent sandhi (0 = heiban).
- Punctuation and unreadable symbols come back as `pos == "記号"`, `mora_size == 0`.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

from ..errors import TextError
from .morae import Mora, MoraKind, split_morae, to_katakana, to_morae
from .pitch_pattern import MoraPitch, fall_assessable, phrase_pitch

_DEVOICE_MARK = "’"
_QUESTION = ("?", "？")

# Surfaces whose automatic reading is commonly ambiguous (spec §5). Matched on
# the analyzer's word surface (行った is split as 行っ + た).
AMBIGUOUS_SURFACES = frozenset({"今日", "明日", "昨日", "行っ", "方", "上手"})


@dataclass
class Word:
    surface: str
    reading: str  # pronunciation kana (katakana, no devoicing marks)
    morae: list[Mora]
    devoiced: list[bool]  # per mora: vowel expected devoiced
    phrase: int  # index into Utterance.phrases
    is_head: bool  # first word of its accent phrase (carries the phrase outcome)
    uncertain: bool = False  # ambiguous reading, no override supplied


@dataclass
class Phrase:
    words: list[int]
    n_morae: int
    nucleus: int | None  # None = unknown → pitch abstained
    pitch: list[MoraPitch] | None
    assessable: bool  # the expected fall (or its absence) is observable
    question_final: bool = False  # last phrase before a question mark


@dataclass
class Utterance:
    text: str
    words: list[Word]
    phrases: list[Phrase]
    pause_after: set[int] = field(default_factory=set)  # word idx followed by punctuation

    @property
    def reading_uncertain(self) -> bool:
        return any(w.uncertain for w in self.words)

    @property
    def n_morae(self) -> int:
        return sum(len(w.morae) for w in self.words)


def normalize(text: str) -> str:
    """NFKC (full-width digits/latin → half-width, half-width kana → full-width)."""
    return unicodedata.normalize("NFKC", text).strip()


def _parse_pron(pron: str) -> tuple[str, list[bool]]:
    """Strip devoicing marks; return kana and a per-mora devoiced flag."""
    kana = pron.replace(_DEVOICE_MARK, "")
    flags = [False] * len(split_morae(kana))
    prefix = ""
    for chunk in pron.split(_DEVOICE_MARK)[:-1]:
        prefix += chunk
        if prefix:
            flags[len(split_morae(prefix)) - 1] = True
    return kana, flags


def _is_kana_text(s: str) -> bool:
    try:
        to_morae(s)
        return True
    except (ValueError, KeyError):
        return False


@dataclass
class _Node:
    surface: str
    pos: str
    read: str
    pron: str
    acc: int
    mora_size: int
    chain: int
    covered: bool = False  # reading supplied by the caller
    accent_unknown: bool = False  # reading changed → dictionary accent no longer valid


@lru_cache(maxsize=1024)
def _njd(text: str) -> tuple[tuple, ...]:
    """Memoized analyzer output (spec §6.2 front-end caching)."""
    try:
        import pyopenjtalk
    except ImportError as e:  # pragma: no cover - depends on extras
        raise ImportError("hatsuon text front-end needs: pip install 'hatsuon[frontend]'") from e
    return tuple(
        (f["string"], f["pos"], f["read"], f["pron"], int(f["acc"]), int(f["mora_size"]),
         int(f["chain_flag"]))
        for f in pyopenjtalk.run_frontend(text)
    )


def _nodes(text: str) -> list[_Node]:
    return [_Node(*n) for n in _njd(text)]


def analyze(
    text: str,
    reading: str | dict[str, str] | None = None,
    accent: dict[str, int] | None = None,
) -> Utterance:
    """Analyze target text into words, morae and accent phrases.

    `reading` may be a whole-text kana string (words are then segmented on the
    reading itself) or a {surface: kana} mapping for individual words.
    `accent` maps a word surface to its accent nucleus (0 = heiban), overriding the
    dictionary for the phrase that word heads.
    Raises TextError when nothing readable remains or an override cannot be aligned.
    """
    norm = normalize(text)
    if not norm:
        raise TextError("target text is empty")
    if isinstance(reading, str):
        nodes = _nodes_from_reading(reading)
    else:
        nodes = _nodes(norm)
        if reading:
            for key, kana in reading.items():
                nodes = _apply_word_reading(nodes, normalize(key), kana)

    words: list[Word] = []
    phrases: list[Phrase] = []
    pause_after: set[int] = set()
    nuclei: list[int | None] = []
    for n in nodes:
        if n.mora_size == 0 or n.pos == "記号":
            if words:
                pause_after.add(len(words) - 1)
                if n.surface in _QUESTION:
                    phrases[-1].question_final = True
            continue
        kana, devoiced = _parse_pron(n.pron)
        morae = to_morae(kana, words[-1].morae[-1].vowel if words else None)
        head = n.chain != 1 or not phrases or (len(words) - 1) in pause_after
        if head:
            phrases.append(Phrase(words=[], n_morae=0, nucleus=None, pitch=None,
                                  assessable=False))
            nuclei.append(n.acc)
        if n.accent_unknown:
            nuclei[-1] = None
        if head and accent and n.surface in accent:
            nuclei[-1] = accent[n.surface]
        phrase = phrases[-1]
        phrase.words.append(len(words))
        phrase.n_morae += len(morae)
        words.append(Word(
            surface=n.surface, reading=kana, morae=morae, devoiced=devoiced,
            phrase=len(phrases) - 1, is_head=head,
            uncertain=n.surface in AMBIGUOUS_SURFACES and not n.covered,
        ))

    if not words:
        raise TextError(f"no Japanese-readable content in {text!r}")

    for phrase, nucleus in zip(phrases, nuclei):
        if nucleus is not None and 0 <= nucleus <= phrase.n_morae:
            morae = [m for i in phrase.words for m in words[i].morae]
            phrase.nucleus = nucleus
            phrase.pitch = phrase_pitch(morae, nucleus)
            phrase.assessable = fall_assessable(phrase.n_morae, nucleus)
    return Utterance(text=norm, words=words, phrases=phrases, pause_after=pause_after)


def _kana_only(s: str) -> str:
    """Drop punctuation/whitespace; raise TextError if anything non-kana remains."""
    out = "".join(c for c in normalize(s) if unicodedata.category(c)[0] not in "PZS")
    if not _is_kana_text(out):
        raise TextError(f"reading override is not plain kana: {s!r}")
    return to_katakana(out)


def _nodes_from_reading(reading: str) -> list[_Node]:
    """Whole-text kana override: segment and look up accents on the reading itself."""
    target = _kana_only(reading)
    nodes = _nodes(normalize(reading))
    got = "".join(n.read for n in nodes if n.mora_size > 0)
    if got != target:
        raise TextError(f"reading override {reading!r} cannot be aligned (analyzer read {got!r})")
    for n in nodes:
        n.covered = True
    return nodes


def _apply_word_reading(nodes: list[_Node], key: str, kana: str) -> list[_Node]:
    """Replace the reading of the first run of words whose surfaces spell `key`."""
    target = _kana_only(kana)
    for start in range(len(nodes)):
        spelled, end = "", start
        while end < len(nodes) and len(spelled) < len(key):
            spelled += nodes[end].surface
            end += 1
        if spelled != key:
            continue
        run = nodes[start:end]
        pron = "".join(n.pron for n in run)
        unchanged = target in (pron.replace(_DEVOICE_MARK, ""), "".join(n.read for n in run))
        merged = _Node(
            surface=key, pos=run[0].pos, read=target,
            pron=pron if unchanged else target,
            acc=run[0].acc, mora_size=len(split_morae(target)), chain=run[0].chain,
            covered=True, accent_unknown=not unchanged,
        )
        return nodes[:start] + [merged] + nodes[end:]
    raise TextError(f"reading override key {key!r} does not match word boundaries in the text")


def is_long_variant(prev: Mora | None, cur: Mora) -> bool:
    """えい / おう spelled with a bare vowel may be pronounced long (spec §5)."""
    return (
        prev is not None and cur.kind is MoraKind.REGULAR and len(cur.phonemes) == 1
        and (prev.vowel, cur.phonemes[0]) in {("e", "i"), ("o", "u")}
    )
