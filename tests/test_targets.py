"""Target lattice built from analyzed text (no acoustic model needed)."""

import pytest

pytest.importorskip("pyopenjtalk")

from hatsuon.acoustic import Vocab
from hatsuon.frontend.analyze import analyze
from hatsuon.targets import build_target

_TOKENS = ["PAD", "pau", "sil", *"aiueoIUN", "cl", *"kgsztdnhbpmyrw", "sh", "ch", "ts", "ky", "j"]
VOCAB = Vocab({t: i for i, t in enumerate(_TOKENS)})
TOK = VOCAB.tokens


def slots(text):
    tg = build_target(analyze(text), VOCAB)
    return [(tuple(TOK[a] for a in s.alts), s.optional, r is None)
            for s, r in zip(tg.slots, tg.refs)]


def test_pause_slots_around_words():
    s = slots("東京に行きます。")
    pauses = [x for x in s if x[2]]
    assert len(pauses) == 1 + 4  # utterance start + after each word
    assert all(opt for _, opt, _ in pauses)


def test_devoiced_vowel_is_optional():
    # 行きました: シ is devoiced → [I, i], optional
    s = [x for x in slots("行きました") if not x[2]]
    assert (("I", "i"), True, False) in s
    # a voiced i also accepts the devoiced unit, but stays required
    assert (("i", "I"), False, False) in s


def test_long_variant_alternative():
    # an explicit せい reading: the い may be realised as a long e
    tg = build_target(analyze("先生", reading="せんせい"), VOCAB)
    alts = [tuple(TOK[a] for a in s.alts) for s in tg.slots]
    assert ("i", "I", "e") in alts


def test_word_initial_choon_inherits_previous_vowel():
    # OpenJTalk splits であろう as であろ|う(ー): the ー must carry the o
    u = analyze("観察されるであろう。")
    morae = [m for w in u.words for m in w.morae]
    assert all(m.phonemes for m in morae)
    assert morae[-1].phonemes == ("o",)


def test_unknown_phoneme_raises():
    small = Vocab({"PAD": 0, "pau": 1, "sil": 2, "a": 3})
    with pytest.raises(ValueError, match="inventory"):
        build_target(analyze("かさ"), small)
