import hatsuon


def test_public_api():
    assert hatsuon.__version__
    assert callable(hatsuon.score)
