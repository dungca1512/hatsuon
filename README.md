# hatsuon

Japanese pronunciation scoring for read-aloud speech, on the CPU. It scores
accuracy per mora, the special morae (っ, ん, long vowels) and Tokyo pitch accent.

> Status: **alpha (0.1)**. Thresholds were tuned on native speech only (JSUT/JVS),
> and no L2 learner data has been used yet. Treat the scores as feedback, not as a
> grade. Design and tuning records: [docs/SPEC.md](docs/SPEC.md) and
> [`calibration/default.json`](src/hatsuon/calibration/default.json).

## Install

```bash
pip install "hatsuon[all]"
```

The acoustic model is **not** shipped in the package. Convert it once on your
machine. This step needs PyTorch and transformers only for the conversion itself:

```bash
pip install "hatsuon[export]"
hatsuon-export            # writes ~/.cache/hatsuon/models/... (ONNX, int8)
```

`hatsuon-export --out DIR` writes the model elsewhere; set `HATSUON_MODEL_DIR=DIR`
to use it. `--no-quantize` keeps FP32 weights.

## Use

```python
import hatsuon

r = hatsuon.score("learner.wav", "東京に行きます。")
print(r.overall, r.accuracy, r.completeness, r.pitch, r.timing, r.confidence)
for w in r.words:
    print(w.surface, w.reading, w.score, w.accent.value)
    for m in w.morae:
        print("  ", m.kana, m.status.value, round(m.score), m.duration, m.pitch_status)
print([x.value for x in r.warnings])
```

- `audio` can be a path, encoded bytes, or a `(samples, sample_rate)` tuple.
- Ambiguous readings: `reading="きょう"` (whole text) or `reading={"今日": "きょう"}`.
- Accent overrides: `accent={"箸": 1}` (0 = heiban, n = fall after mora n).
- `pitch=False` or `timing=False` skips those stages.
- `debug=True` attaches F0, the free decode and per-phone GOP to `r.debug`.
- Soft problems such as `no_speech`, `content_mismatch` or `pitch_unreliable`
  come back in `r.warnings` and lower `r.confidence`.
- Hard errors raise `hatsuon.AudioError` or `hatsuon.TextError`.

## Known limitations (0.1)

- **Native-only calibration.** We have not yet measured mispronunciation recall
  on learners. On 20 held-out JVS speakers, 0.6% of correct native morae fall
  below the `mispronounced` threshold. Synthetic one-mora substitutions (the
  text is changed, the audio is not) are caught 94% of the time: 98% for
  consonants and 91% for vowels. Real learner errors are subtler than these.
- **Pitch accent:**
  - Native speakers often place the F0 peak one mora late, so a fall up to one
    mora after the nucleus is accepted. As a result, a learner who puts the
    nucleus one mora too late is not caught.
  - About 9% of native heiban phrases are flagged with a false fall.
- **Readings come from OpenJTalk.** When it misreads a word (proper nouns,
  命 → イノチ when the speaker says めい), a correct reading shows up as
  `deleted` morae. Pass `reading={"命": "めい"}` in those cases.
- Read speech only. Not for spontaneous speech or singing.

## Development

```bash
pip install -e ".[all,dev]"
pytest && ruff check .
bash tools/fetch_data.sh                  # JSUT, JVS, jsut-label into data/ (see DATA_SOURCES.md)
python tools/eval_accent.py --n 300       # pitch-accent accuracy on JSUT
python tools/calibrate.py collect && python tools/calibrate.py fit --write
```

## License

MIT. Corpora and the acoustic model have their own licences and are never
redistributed. See [DATA_SOURCES.md](DATA_SOURCES.md).
