"""Expected Tokyo pitch templates (spec §4.1.4)."""

import pytest

from hatsuon.frontend.morae import to_morae
from hatsuon.frontend.pitch_pattern import expected_levels, fall_assessable, phrase_pitch


@pytest.mark.parametrize(
    ("n", "nucleus", "expected"),
    [
        (3, 1, "HLL"),  # atamadaka: いのち
        (3, 0, "LHH"),  # heiban: さくら
        (4, 2, "LHLL"),  # nakadaka
        (3, 3, "LHH"),  # odaka looks like heiban inside the word
        (4, 3, "LHHL"),  # odaka word + particle: fall on the particle
        (1, 1, "H"),
        (1, 0, "L"),
    ],
)
def test_expected_levels(n, nucleus, expected):
    assert "".join(expected_levels(n, nucleus)) == expected


def test_nucleus_out_of_range():
    with pytest.raises(ValueError):
        expected_levels(3, 4)


def test_heavy_first_syllable_is_soft():
    # トーキョー (heiban): mora 2 is ー → initial lowering is soft
    p = phrase_pitch(to_morae("トーキョー"), 0)
    assert p[0].soft and not any(x.soft for x in p[1:])
    # サクラ: light first syllable → not soft
    assert not phrase_pitch(to_morae("サクラ"), 0)[0].soft
    # nakadaka with heavy first syllable: soft as well
    assert phrase_pitch(to_morae("コーヒー"), 3)[0].soft
    # atamadaka keeps mora 1 as a real H peak even when heavy
    assert not phrase_pitch(to_morae("ホンヤ"), 1)[0].soft


def test_nucleus_flag():
    p = phrase_pitch(to_morae("ハナガ"), 2)
    assert [x.nucleus for x in p] == [False, True, False]


def test_fall_assessable():
    assert fall_assessable(4, 3)  # fall shows on the particle
    assert not fall_assessable(3, 3)  # phrase-final odaka
    assert fall_assessable(3, 0)
