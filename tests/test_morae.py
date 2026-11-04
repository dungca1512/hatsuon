"""Mora segmentation rules (spec §4.1.3)."""

import pytest

from hatsuon.frontend.morae import split_morae


@pytest.mark.parametrize(
    ("kana", "expected"),
    [
        # small ャュョ attach to the preceding kana
        ("キョウ", ["キョ", "ウ"]),
        ("シュッパツ", ["シュ", "ッ", "パ", "ツ"]),
        # vowel doubling: 4 morae, 2 syllables
        ("トウキョウ", ["ト", "ウ", "キョ", "ウ"]),
        ("オカアサン", ["オ", "カ", "ア", "サ", "ン"]),
        # ー is its own mora
        ("トーキョー", ["ト", "ー", "キョ", "ー"]),
        # っ and ん are their own morae
        ("キッテ", ["キ", "ッ", "テ"]),
        ("ガッコウ", ["ガ", "ッ", "コ", "ウ"]),
        ("ホン", ["ホ", "ン"]),
        # loanword small vowels attach (ファ, ティ, ヴァ, ウィ)
        ("ファイル", ["ファ", "イ", "ル"]),
        ("ティー", ["ティ", "ー"]),
        ("ヴァイオリン", ["ヴァ", "イ", "オ", "リ", "ン"]),
        ("ウィキ", ["ウィ", "キ"]),
        ("クヮ", ["クヮ"]),
        # hiragana input is normalized to katakana
        ("きゃく", ["キャ", "ク"]),
        ("がっこう", ["ガ", "ッ", "コ", "ウ"]),
        # edge cases
        ("", []),
        ("ャ", ["ャ"]),  # leading small kana: own mora, no crash
    ],
)
def test_split_morae(kana, expected):
    assert split_morae(kana) == expected


@pytest.mark.parametrize("bad", ["東京", "abc", "キョウ。"])
def test_split_morae_rejects_non_kana(bad):
    with pytest.raises(ValueError):
        split_morae(bad)
