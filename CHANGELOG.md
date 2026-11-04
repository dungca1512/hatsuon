# Changelog

## 0.1.2 — 2026-11-04

### Fixed
- `truncated` no longer fires on recordings that stop right after speech has
  decayed. Speech must also still be loud at the file edge
  (`AudioOptions.truncation_edge_frac`, calibrated: test false-alarm 1.0 %, recall
  86 % on synthetic cuts). On a large L2 practice set the flag rate fell from
  56 % to 17 %.
- More than 8 audio channels now raises `AudioError` before decoding, since every
  channel is decoded before the downmix.
- The M4A/AAC fallback decoder (audioread) reads the sample rate, channel count
  and duration from the metadata, and rejects bad input before decoding.

### Docs
- README "Known limitations": the `mispronounced` flag rate on unlabelled
  non-native read speech.

## 0.1.1

### Fixed
- Audio size limits are checked before decoding and resampling. Before this fix,
  an over-long file was fully decoded, and a waveform with an absurd sample rate
  (for example `(x, 1)`) was resampled, before the 60 s limit applied. Either
  case could exhaust memory.
- A sample rate outside 8–192 kHz now raises `AudioError`.

### Changed
- Releases are published from GitHub Actions with PyPI Trusted Publishing and
  carry provenance attestations.
- The package metadata now links to the GitHub repository.

## 0.1.0

First public release (alpha).
