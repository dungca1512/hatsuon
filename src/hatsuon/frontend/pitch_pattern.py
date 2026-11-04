"""Expected Tokyo pitch pattern per accent phrase (spec §4.1.4)."""

from __future__ import annotations

from dataclasses import dataclass

from .morae import Mora, MoraKind

H, L = "H", "L"

# Kinds that make the first syllable heavy when they occur in mora 2.
_HEAVY_SECOND = frozenset({MoraKind.LONG, MoraKind.HATSUON, MoraKind.SOKUON})


@dataclass(frozen=True)
class MoraPitch:
    level: str  # "H" | "L"
    nucleus: bool  # this mora carries the accent nucleus (last H before the fall)
    soft: bool  # level is only weakly realized; do not penalize a mismatch


def expected_levels(n_morae: int, nucleus: int) -> list[str]:
    """H/L template for an accent phrase of `n_morae` with nucleus `nucleus` (0 = heiban)."""
    if n_morae <= 0:
        return []
    if not 0 <= nucleus <= n_morae:
        raise ValueError(f"nucleus {nucleus} out of range for {n_morae} morae")
    if nucleus == 1:
        return [H] + [L] * (n_morae - 1)
    if nucleus == 0:
        return [L] + [H] * (n_morae - 1)
    return [L] + [H] * (nucleus - 1) + [L] * (n_morae - nucleus)


def phrase_pitch(morae: list[Mora], nucleus: int) -> list[MoraPitch]:
    """Expected pitch for each mora of one accent phrase.

    The first mora is marked soft when the first syllable is heavy (initial lowering
    is weak or absent), unless the phrase is atamadaka (where mora 1 is the H peak).
    """
    levels = expected_levels(len(morae), nucleus)
    heavy_first = len(morae) >= 2 and morae[1].kind in _HEAVY_SECOND
    return [
        MoraPitch(
            level=lvl,
            nucleus=(i + 1 == nucleus),
            soft=(i == 0 and heavy_first and nucleus != 1),
        )
        for i, lvl in enumerate(levels)
    ]


def fall_assessable(n_morae: int, nucleus: int) -> bool:
    """Whether the accent fall is observable inside the phrase.

    Odaka (nucleus == last mora) looks like heiban unless a following mora shows the
    fall; since the phrase already includes attached particles, a phrase-final
    nucleus means there is nothing to observe it on (spec §5, "Odaka vs. heiban").
    """
    return nucleus != n_morae or n_morae == 0
