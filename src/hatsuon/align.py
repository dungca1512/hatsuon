"""CTC forced alignment and free decode (spec §4.2.2–§4.2.4), numpy only.

The target is a list of `Slot`s. Each slot accepts any of several token ids
(pronunciation variants as parallel paths) and may be optional (pauses, devoiced
vowels). The lattice is the usual CTC one — blank, token, blank, token, … — with
extra skip arcs over optional slots.

One deliberate departure from CTC collapse semantics: two adjacent slots with the
same token may follow each other without a blank in between. A long vowel `o o`
is often emitted as one continuous `o` run; forcing a blank inside it would put
the boundary on noise. Such merged runs are split evenly between the two slots.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

NEG = -1e30


@dataclass(frozen=True)
class Slot:
    alts: tuple[int, ...]  # accepted token ids; alts[0] is the canonical one
    optional: bool = False


@dataclass
class Segment:
    token: int | None  # chosen alternative; None when an optional slot was skipped
    start: int  # frame span after blank redistribution, end exclusive
    end: int
    core: tuple[int, int]  # frames on which the token itself was emitted

    @property
    def skipped(self) -> bool:
        return self.token is None


@dataclass
class Alignment:
    segments: list[Segment]
    score: float  # total path log-probability
    n_frames: int


class AlignmentError(ValueError):
    pass


def _graph(slots: list[Slot]):
    """Predecessor table for states 0..2K (even = blank, odd = token of slot (s-1)/2)."""
    k_n = len(slots)
    n = 2 * k_n + 1
    preds: list[set[int]] = [{s} for s in range(n)]
    for s in range(1, n):
        preds[s].add(s - 1)
    for k in range(1, k_n):
        preds[2 * k + 1].add(2 * k - 1)  # token → token without blank
    # skip arcs: exits of slot j → entries of slot k when slots j+1..k-1 are optional
    for k in range(k_n + 1):  # k == k_n: the final blank
        j = k - 2
        while j >= -1 and slots[j + 1].optional:
            entries = [2 * k] if k == k_n else [2 * k, 2 * k + 1]
            exits = [0] if j == -1 else [2 * j + 1, 2 * j + 2]
            for e in entries:
                preds[e].update(exits)
            j -= 1
    start = {0, 1}
    for k in range(k_n):
        if not slots[k].optional:
            break
        start.update({2 * k + 2, 2 * k + 3} if k + 1 < k_n else {2 * k + 2})
    end = {n - 1, n - 2} if k_n else {0}
    for k in range(k_n - 1, -1, -1):
        if not slots[k].optional:
            break
        end.update({2 * k - 1, 2 * k} if k else {0})
    width = max(len(p) for p in preds)
    table = np.full((n, width), -1, dtype=np.int64)
    for s, p in enumerate(preds):
        table[s, : len(p)] = sorted(p)
    return table, sorted(start), sorted(end)


def _emissions(logp: np.ndarray, slots: list[Slot], blank: int) -> np.ndarray:
    t_n = logp.shape[0]
    e = np.empty((t_n, 2 * len(slots) + 1), dtype=np.float64)
    e[:, 0::2] = logp[:, blank : blank + 1]
    for k, sl in enumerate(slots):
        e[:, 2 * k + 1] = logp[:, list(sl.alts)].max(axis=1)
    return e


def force_align(logp: np.ndarray, slots: list[Slot], blank: int) -> Alignment:
    """Best path of (T, V) log-posteriors through the slot lattice."""
    t_n = logp.shape[0]
    if not slots:
        return Alignment([], float(logp[:, blank].sum()), t_n)
    if t_n < sum(not s.optional for s in slots):
        raise AlignmentError(f"{t_n} frames cannot hold {len(slots)} target tokens")
    table, start, end = _graph(slots)
    em = _emissions(logp, slots, blank)
    n = table.shape[0]
    valid = table >= 0
    idx = np.where(valid, table, 0)

    delta = np.full(n, NEG)
    delta[start] = em[0, start]
    back = np.zeros((t_n, n), dtype=np.int32)
    for t in range(1, t_n):
        cand = np.where(valid, delta[idx], NEG)
        j = cand.argmax(axis=1)
        back[t] = idx[np.arange(n), j]
        delta = cand[np.arange(n), j] + em[t]
    last = max(end, key=lambda s: delta[s])
    score = float(delta[last])
    if score <= NEG / 2:
        raise AlignmentError("no valid alignment path")
    path = np.empty(t_n, dtype=np.int64)
    path[-1] = last
    for t in range(t_n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return Alignment(_segments(path, logp, slots), score, t_n)


def _segments(path: np.ndarray, logp: np.ndarray, slots: list[Slot]) -> list[Segment]:
    segs: list[Segment] = []
    for k, sl in enumerate(slots):
        frames = np.flatnonzero(path == 2 * k + 1)
        if frames.size == 0:
            segs.append(Segment(None, -1, -1, (-1, -1)))
            continue
        a, b = int(frames[0]), int(frames[-1]) + 1
        tok = max(sl.alts, key=lambda q: float(logp[a:b, q].sum()))
        segs.append(Segment(tok, a, b, (a, b)))
    present = [s for s in segs if not s.skipped]
    for prev, nxt in pairwise(present):
        if prev.end == nxt.start and prev.token == nxt.token:
            # merged identical run (long vowel): split it evenly
            lo, hi = prev.core[0], nxt.core[1]
            mid = (lo + hi + 1) // 2
            prev.core, nxt.core = (lo, mid), (mid, hi)
            prev.end, nxt.start = mid, mid
        elif nxt.start > prev.end:  # blank gap: give each side half
            mid = (prev.end + nxt.start + 1) // 2
            prev.end, nxt.start = mid, mid
    return segs


# --- free decode and edit-distance comparison (§4.2.4) ---------------------------------


def greedy_decode(logp: np.ndarray, blank: int, drop: frozenset[int] = frozenset()):
    """Best-path decode → list of (token, first_frame, last_frame_exclusive)."""
    best = logp.argmax(axis=1)
    out: list[tuple[int, int, int]] = []
    prev = blank
    for t, q in enumerate(best.tolist()):
        if q != prev and q != blank and q not in drop:
            out.append((q, t, t + 1))
        elif q == prev and out and q != blank and out[-1][0] == q and out[-1][2] == t:
            out[-1] = (q, out[-1][1], t + 1)
        prev = q
    return out


@dataclass
class EditResult:
    matched: list[int | None]  # per slot: index into hyp, or None (deleted / skipped)
    insertions: list[int]  # hyp indices with no target counterpart
    match_ratio: float  # matched required slots / max(required slots, hyp tokens)


def edit_align(hyp: list[int], slots: list[Slot]) -> EditResult:
    """Levenshtein alignment of decoded tokens to target slots.

    A hyp token matches a slot if it is one of the slot's alternatives. Skipping an
    optional slot is free; substitutions cost 1, as do insertions and deletions.
    Optional slots cannot be substituted: a wrong token there is an insertion.

    `match_ratio` divides by the longer of target and hypothesis: with free
    insertions a short target finds most of its phones scattered through a long,
    unrelated utterance, so dividing by the target alone misses wrong sentences.
    """
    k_n, h_n = len(slots), len(hyp)
    cost = np.zeros((k_n + 1, h_n + 1))
    cost[0, 1:] = np.arange(1, h_n + 1)
    for k in range(1, k_n + 1):
        cost[k, 0] = cost[k - 1, 0] + (0 if slots[k - 1].optional else 1)
    for k in range(1, k_n + 1):
        sl = slots[k - 1]
        skip = 0 if sl.optional else 1
        for h in range(1, h_n + 1):
            sub = 0 if hyp[h - 1] in sl.alts else (np.inf if sl.optional else 1)
            cost[k, h] = min(cost[k - 1, h - 1] + sub, cost[k - 1, h] + skip, cost[k, h - 1] + 1)
    matched: list[int | None] = [None] * k_n
    used: set[int] = set()
    k, h = k_n, h_n
    while k > 0 or h > 0:
        sl = slots[k - 1] if k else None
        hit = bool(k and h and hyp[h - 1] in sl.alts)
        if k and h and (hit or not sl.optional) and cost[k, h] == cost[k - 1, h - 1] + (not hit):
            if hit:
                matched[k - 1] = h - 1
            used.add(h - 1)  # substitution consumes the hyp token too
            k, h = k - 1, h - 1
        elif k and cost[k, h] == cost[k - 1, h] + (0 if sl.optional else 1):
            k -= 1
        else:
            h -= 1
    required = [i for i, s in enumerate(slots) if not s.optional]
    hits = sum(matched[i] is not None for i in required)
    ratio = hits / max(len(required), h_n) if required else 1.0
    return EditResult(matched, sorted(set(range(h_n)) - used), ratio)
