"""Kana → phoneme mapping and special-mora kinds (spec §4.1.3)."""

import pytest

from hatsuon.frontend.morae import MoraKind, to_morae


def phonemes(kana: str) -> list[str]:
    return [p for m in to_morae(kana) for p in m.phonemes]


@pytest.mark.parametrize(
    ("kana", "expected"),
    [
        ("トーキョー", "t o o ky o o"),
        ("キッテ", "k i cl t e"),
        ("シンブン", "sh i N b u N"),
        ("チャ", "ch a"),
        ("ジュ", "j u"),
        ("ツヅク", "ts u z u k u"),
        ("フジ", "f u j i"),
        ("ファイル", "f a i r u"),
        ("ティー", "t i i"),
        ("ディスク", "d i s u k u"),
        ("テュ", "ty u"),
        ("シェフ", "sh e f u"),
        ("ウィ", "w i"),
        ("イェ", "y e"),
        ("ヴァ", "v a"),
        ("ヲ", "o"),
        ("フュ", "fy u"),
        ("クヮ", "kw a"),
    ],
)
def test_phonemes(kana, expected):
    assert phonemes(kana) == expected.split()


@pytest.mark.parametrize(
    ("kana", "kinds"),
    [
        ("オカアサン", "R R L R H"),
        ("トーキョー", "R L R L"),
        ("ガッコウ", "R S R R"),  # コウ is a variant (ou), not marked long here
        ("オバサン", "R R R H"),
        ("オバアサン", "R R L R H"),
        ("ニッポン", "R S R H"),
    ],
)
def test_kinds(kana, kinds):
    code = {"R": MoraKind.REGULAR, "L": MoraKind.LONG, "S": MoraKind.SOKUON,
            "H": MoraKind.HATSUON}
    assert [m.kind for m in to_morae(kana)] == [code[k] for k in kinds.split()]
