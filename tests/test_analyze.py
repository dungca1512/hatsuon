"""Text analysis with pyopenjtalk (spec §4.1, §5)."""

import pytest

pytest.importorskip("pyopenjtalk")

from hatsuon.errors import TextError
from hatsuon.frontend.analyze import analyze, normalize


def readings(text, **kw):
    return [w.reading for w in analyze(text, **kw).words]


def test_basic_sentence():
    u = analyze("東京に行きます。")
    assert [w.surface for w in u.words] == ["東京", "に", "行き", "ます"]
    assert readings("東京に行きます。") == ["トーキョー", "ニ", "イキ", "マス"]
    assert len(u.phrases) == 2
    assert u.phrases[0].nucleus == 0 and u.phrases[0].n_morae == 5  # heiban + particle
    assert u.phrases[1].nucleus == 3  # イキマ'ス
    assert "".join(p.level for p in u.phrases[1].pitch) == "LHHL"
    assert u.n_morae == 9


def test_devoicing_flags():
    u = analyze("行きました")
    flags = [(m.kana, d) for w in u.words for m, d in zip(w.morae, w.devoiced)]
    assert ("シ", True) in flags
    assert all(not d for k, d in flags if k != "シ")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("今日は", ["キョー", "ワ"]),  # particle は → ワ
        ("学校へ", ["ガッコー", "エ"]),  # particle へ → エ
        ("本を", ["ホン", "ヲ"]),  # を kept as ヲ; phoneme is /o/
    ],
)
def test_particles(text, expected):
    assert readings(text) == expected


@pytest.mark.parametrize(
    ("text", "joined"),
    [
        ("三本", "サンボン"),
        ("3本", "サンボン"),  # NFKC + number handling
        ("六本", "ロッポン"),
        ("一本", "イッポン"),
        ("八百", "ハッピャク"),
    ],
)
def test_counters(text, joined):
    assert "".join(readings(text)) == joined


def test_normalize_halfwidth():
    assert normalize("ｶﾀｶﾅ　ＡＢＣ１") == "カタカナ ABC1"


def test_punctuation_is_phrase_boundary():
    u = analyze("はい、行きます")
    assert 0 in u.pause_after
    assert u.words[1].is_head


def test_question_flag():
    u = analyze("行きますか？")
    assert u.phrases[-1].question_final


def test_phrase_final_odaka_not_assessable():
    # a one-word phrase with nucleus on the last mora cannot show its fall
    u = analyze("花", accent={"花": 2})
    assert u.phrases[0].nucleus == 2 and not u.phrases[0].assessable


@pytest.mark.parametrize("bad", ["", "   ", "。、！", "😀"])
def test_unreadable_text_raises(bad):
    with pytest.raises(TextError):
        analyze(bad)


def test_ambiguous_reading_flag():
    assert analyze("今日は晴れ").reading_uncertain
    assert not analyze("東京").reading_uncertain


def test_word_reading_override():
    u = analyze("今日は晴れ", reading={"今日": "こんにち"})
    w = u.words[0]
    assert w.reading == "コンニチ" and not w.uncertain
    assert u.phrases[w.phrase].nucleus is None  # accent abstained for changed reading


def test_word_override_equal_to_analyzer_keeps_accent():
    u = analyze("今日は晴れ", reading={"今日": "きょう"})
    assert u.words[0].reading == "キョー"
    assert u.phrases[0].nucleus is not None and not u.reading_uncertain


def test_override_spanning_words():
    u = analyze("学校へ行った", reading={"行った": "いった"})
    assert u.words[-1].surface == "行った" and u.words[-1].reading == "イッタ"


def test_whole_text_reading():
    u = analyze("今日は晴れ", reading="こんにちははれ")
    assert "".join(w.reading for w in u.words).startswith("コンニチワ")
    assert not u.reading_uncertain


@pytest.mark.parametrize(
    ("reading", "err"),
    [({"明日": "あした"}, "does not match"), ({"今日": "today"}, "not plain kana"),
     ("today", "not plain kana")],
)
def test_bad_overrides(reading, err):
    with pytest.raises(TextError, match=err):
        analyze("今日は晴れ", reading=reading)
