"""Goodness of Pronunciation on CTC posteriors (spec §4.3.1).

CTC posteriors are peaky: a phone is often emitted on one or two frames with blank
around it, so a per-frame average over its "core" frames is noisy. We use the
posterior *mass* over the whole aligned span (core + its share of blanks):

    m_q   = Σ_t P(q | t)            for every phone q, t in the span
    GOP   = log m_target − log max_q m_q     (≤ 0; 0 = target is the best phone)

Accepted alternatives (devoiced unit, long-vowel variant) count as the target.
The strongest non-accepted phone is the substitution candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .align import Segment, Slot

_EPS = 1e-8


@dataclass
class PhoneGop:
    gop: float | None  # None when an optional slot was skipped (nothing to judge)
    competitor: int | None  # best non-accepted phone over the span
    margin: float  # log m_competitor − log m_target (> 0 = competitor stronger)
    core_logp: float  # mean log P(token) on emitted frames (alignment confidence)


def phone_gop(logp: np.ndarray, seg: Segment, slot: Slot, phones: np.ndarray) -> PhoneGop:
    if seg.skipped:
        return PhoneGop(None, None, 0.0, 0.0)
    span = np.exp(logp[seg.start : seg.end].astype(np.float64))
    mass = span[:, phones].sum(axis=0)
    accepted = np.isin(phones, slot.alts)
    m_target = float(mass[accepted].max()) if accepted.any() else 0.0
    rivals = np.where(accepted, -1.0, mass)
    j = int(rivals.argmax())
    m_rival = float(rivals[j])
    best = max(m_target, m_rival)
    a, b = seg.core
    return PhoneGop(
        gop=float(np.log(m_target + _EPS) - np.log(best + _EPS)),
        competitor=int(phones[j]) if m_rival > 0 else None,
        margin=float(np.log(m_rival + _EPS) - np.log(m_target + _EPS)),
        core_logp=float(logp[a:b, seg.token].mean()),
    )
