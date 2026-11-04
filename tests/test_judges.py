"""Accent-phrase and special-mora judgements on synthetic values."""

from hatsuon.frontend.morae import MoraKind
from hatsuon.pitch import PitchOptions, judge_phrase
from hatsuon.timing import TimingBounds, judge, mean_mora_duration
from hatsuon.types import AccentOutcome as A
from hatsuon.types import DurationStatus

STRICT = PitchOptions(peak_delay=0)
OPTS = PitchOptions()  # dead zone 1.5 st, peak delay 1


def j(vals, nucleus, opts=OPTS, question=False):
    return judge_phrase(vals, nucleus, True, question, opts)


def test_heiban():
    assert j([-3.0, 1.0, 1.2, 0.8], 0) is A.CORRECT
    assert j([-3.0, 1.0, -2.0, -2.5], 0) is A.FALL_FOR_UNACCENTED


def test_accented_fall_at_nucleus():
    assert j([2.0, -2.0, -3.0], 1) is A.CORRECT  # atamadaka
    assert j([-3.0, 2.0, 2.5, -2.0], 3) is A.CORRECT  # nakadaka
    assert j([-3.0, 1.0, 1.2, 1.0], 2) is A.FLAT_FOR_ACCENTED


def test_peak_delay_tolerance():
    # peak lags into mora 2 (native atamadaka): strict rule misses it, default accepts
    vals = [0.6, 3.7, 0.5, -3.9]
    assert j(vals, 1, STRICT) is A.WRONG_NUCLEUS
    assert j(vals, 1) is A.CORRECT
    # a fall two morae late is still wrong
    assert j([1.0, 1.2, 1.1, -3.0], 1) is A.WRONG_NUCLEUS


def test_null_morae():
    # devoiced nucleus: bridged to the next voiced mora
    assert j([2.0, None, -3.0], 1, STRICT) is A.CORRECT
    assert j([None, None, 1.0, None], 2) is A.UNKNOWN  # coverage too low
    assert judge_phrase([1.0, -3.0], 1, False, False, OPTS) is A.UNKNOWN  # not assessable
    assert judge_phrase([1.0, -3.0], None, True, False, OPTS) is A.UNKNOWN


def test_question_final_mora_ignored():
    # heiban + final rise: the rise must not matter; a final fall is masked too
    assert j([-3.0, 1.0, 1.0, -4.0], 0, question=True) is A.CORRECT


def test_timing_bounds():
    bounds = {MoraKind.SOKUON: TimingBounds(0.6, 3.0)}
    assert judge(MoraKind.SOKUON, 10, 10.0, bounds).status is DurationStatus.OK
    assert judge(MoraKind.SOKUON, 3, 10.0, bounds).status is DurationStatus.TOO_SHORT
    assert judge(MoraKind.SOKUON, 40, 10.0, bounds).status is DurationStatus.TOO_LONG
    assert judge(MoraKind.SOKUON, 10, 10.0, bounds, reliable=False).status is None
    assert judge(MoraKind.REGULAR, 10, 10.0, bounds).status is None
    assert mean_mora_duration([(0, 10), None, (10, 30)]) == 15.0
    assert mean_mora_duration([None]) is None
