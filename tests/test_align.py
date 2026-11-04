"""Forced alignment and free decode on synthetic posteriors (spec §4.2)."""

import numpy as np
import pytest

from hatsuon.align import (
    AlignmentError,
    Slot,
    edit_align,
    force_align,
    greedy_decode,
)

V, BLANK = 8, 0
A, B, C, D, PAU = 1, 2, 3, 4, 7


def post(frames: str) -> np.ndarray:
    """One char per frame: '_' blank, digits = token ids. Peak prob 0.9."""
    lp = np.full((len(frames), V), np.log(0.1 / (V - 1)))
    for t, ch in enumerate(frames):
        lp[t, BLANK if ch == "_" else int(ch)] = np.log(0.9)
    return lp


def spans(al):
    return [None if s.skipped else (s.start, s.end) for s in al.segments]


def test_basic_midpoint_split():
    al = force_align(post("11__2222__3"), [Slot((A,)), Slot((B,)), Slot((C,))], BLANK)
    assert [s.token for s in al.segments] == [A, B, C]
    assert spans(al) == [(0, 3), (3, 9), (9, 11)]
    assert al.segments[1].core == (4, 8)


def test_optional_pause_taken_and_skipped():
    slots = [Slot((A,)), Slot((PAU,), optional=True), Slot((B,))]
    with_pause = force_align(post("1__777__2"), slots, BLANK)
    assert with_pause.segments[1].token == PAU
    without = force_align(post("1____2"), slots, BLANK)
    assert without.segments[1].skipped and spans(without)[2] is not None


def test_optional_at_edges():
    slots = [Slot((PAU,), True), Slot((A,)), Slot((PAU,), True)]
    assert force_align(post("__1__"), slots, BLANK).segments[0].skipped
    al = force_align(post("77_1_77"), slots, BLANK)
    assert [s.token for s in al.segments] == [PAU, A, PAU]


def test_alternatives_pick_best_variant():
    # devoiced vowel: canonical voiceless unit (5) or the voiced vowel (A)
    slots = [Slot((B,)), Slot((5, A), optional=True), Slot((C,))]
    assert force_align(post("2_1_3"), slots, BLANK).segments[1].token == A
    assert force_align(post("2_5_3"), slots, BLANK).segments[1].token == 5
    assert force_align(post("2___3"), slots, BLANK).segments[1].skipped


def test_merged_long_vowel_split_evenly():
    # "o o" emitted as one continuous run without blank
    al = force_align(post("2_111111_3"), [Slot((B,)), Slot((A,)), Slot((A,)), Slot((C,))], BLANK)
    a1, a2 = al.segments[1], al.segments[2]
    assert a1.core == (2, 5) and a2.core == (5, 8) and a1.end == a2.start


def test_required_token_never_skipped():
    al = force_align(post("1____3"), [Slot((A,)), Slot((B,)), Slot((C,))], BLANK)
    assert not al.segments[1].skipped  # forced onto some frame even if absent


def test_too_short():
    with pytest.raises(AlignmentError):
        force_align(post("1_"), [Slot((A,)), Slot((B,)), Slot((C,))], BLANK)


def test_greedy_decode_and_drop():
    hyp = greedy_decode(post("11_1_22__77_3"), BLANK, drop=frozenset({PAU}))
    assert [h[0] for h in hyp] == [A, A, B, C]
    assert hyp[0][1:] == (0, 2)


def test_edit_align():
    slots = [Slot((A,)), Slot((PAU,), True), Slot((B,)), Slot((C,))]
    r = edit_align([A, B, C], slots)
    assert r.match_ratio == 1.0 and r.insertions == [] and r.matched == [0, None, 1, 2]
    r = edit_align([A, D, D, C], slots)  # B substituted, one extra token
    assert r.matched[2] is None and len(r.insertions) == 1
    assert r.match_ratio == pytest.approx(2 / 4)  # 2 hits / max(3 required, 4 decoded)
    assert edit_align([], slots).match_ratio == 0.0


def test_deterministic():
    lp = np.log(np.random.default_rng(0).dirichlet(np.ones(V), size=50))
    slots = [Slot((A,)), Slot((B,), True), Slot((C,)), Slot((D,))]
    assert spans(force_align(lp, slots, BLANK)) == spans(force_align(lp, slots, BLANK))
