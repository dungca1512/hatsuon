"""Target lattice for forced alignment, built from the analyzed text (spec §4.2.2, §5).

Token conventions follow the acoustic model's inventory (OpenJTalk phonemes):
long vowels are doubled vowels, devoiced i/u are the separate units I/U, pauses
are pau/sil.
"""

from __future__ import annotations

from dataclasses import dataclass

from .acoustic import Vocab
from .align import Slot
from .frontend.analyze import Utterance, is_long_variant

_DEVOICED = {"i": "I", "u": "U"}
PAUSE_TOKENS = ("pau", "sil")


@dataclass(frozen=True)
class PhoneRef:
    word: int
    mora: int
    phone: int


@dataclass
class Target:
    slots: list[Slot]
    refs: list[PhoneRef | None]  # per slot; None for optional pauses

    def phone_slots(self) -> list[int]:
        return [i for i, r in enumerate(self.refs) if r is not None]


def build_target(utt: Utterance, vocab: Vocab) -> Target:
    ids = vocab.ids

    def tid(p: str) -> int:
        if p not in ids:
            raise ValueError(f"phoneme {p!r} is not in the acoustic model inventory")
        return ids[p]

    pause = Slot(tuple(ids[p] for p in PAUSE_TOKENS), optional=True)
    slots: list[Slot] = [pause]
    refs: list[PhoneRef | None] = [None]
    for wi, w in enumerate(utt.words):
        prev = None
        for mi, (m, dev) in enumerate(zip(w.morae, w.devoiced)):
            long_variant = is_long_variant(prev, m)
            for pi, p in enumerate(m.phonemes):
                is_vowel = pi == len(m.phonemes) - 1 and p == m.vowel
                alts = [p]
                optional = False
                if is_vowel and p in _DEVOICED:
                    if dev:  # expected devoiced: voiceless unit, voiced vowel or nothing
                        alts = [_DEVOICED[p], p]
                        optional = True
                    else:  # devoicing is allophonic, never an error
                        alts.append(_DEVOICED[p])
                if is_vowel and long_variant:
                    alts.append(prev.vowel)  # せい → s e e, おう → o o
                slots.append(Slot(tuple(dict.fromkeys(tid(a) for a in alts)), optional))
                refs.append(PhoneRef(wi, mi, pi))
            prev = m
        slots.append(pause)  # optional pause after every word (incl. the last)
        refs.append(None)
    return Target(slots, refs)
