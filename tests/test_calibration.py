"""The shipped calibration file carries a record for every value (spec A.4)."""

from hatsuon.calibration import load_calibration

FIELDS = {"value", "status", "range", "dataset", "metric", "date"}


def _within(value, rng) -> bool:
    if isinstance(rng[0], list):  # [lo, hi] bounds, each with its own range
        return all(_within(v, r) for v, r in zip(value, rng))
    lo, hi = rng
    return lo <= value <= hi


def test_every_param_has_a_record():
    cal = load_calibration()
    for name, p in cal.raw["params"].items():
        assert FIELDS <= p.keys(), name
        assert p["status"] in {"provisional", "calibrated"}, name
        if p["status"] == "calibrated":
            assert p["dataset"] not in {"", "none"}, name


def test_numeric_values_inside_search_range():
    for name, p in load_calibration().raw["params"].items():
        if isinstance(p["range"], list) and not isinstance(p["value"], dict):
            assert _within(p["value"], p["range"]), name


def test_gop_norms_are_usable():
    cal = load_calibration()
    for phone, n in cal.raw["gop_norms"].items():
        if phone.startswith("_"):
            continue
        assert n["sd"] > 0, phone
    assert cal.gop_norm("no-such-phone").sd == 1.0
