# Japanese Pronunciation Scoring Library — Behavioral Specification

Status: draft v0.1
Provenance: written from public literature and public tools only. No proprietary
source code, configuration, data or tuning was consulted. All numeric values are
starting points drawn from general phonetics knowledge and must be re-calibrated
on public data (see §7).

---

## 1. Problem statement and scope

### 1.1 Problem
Given a recording of a learner reading a known Japanese sentence, tell the learner
(a) how close their pronunciation is to standard (Tokyo) Japanese, overall and per
word/mora, and (b) *where* and *what kind* of errors occurred, so a front-end can
render actionable feedback.

### 1.2 In scope
- **Read-aloud / repeat-after-me** tasks: the target text is known in advance.
- Short to medium utterances (one phrase up to a few sentences, roughly ≤ 30 s).
- Segmental accuracy at the mora and phoneme level.
- Special morae: sokuon (っ), moraic nasal (ん), long vowels (ー / vowel doubling).
- Lexical pitch accent of standard Tokyo Japanese, at accent-phrase level.
- Basic timing / fluency indicators (speaking rate, long pauses).
- CPU-only inference with acceptable latency.

### 1.3 Explicitly not in scope (v1)
- Free / spontaneous speech scoring (unknown target text).
- Grammar, vocabulary, content or task-achievement scoring.
- Dialectal accent systems other than Tokyo standard.
- Natural-language feedback generation, localization, UI.

### 1.4 Design principles
1. **Abstain rather than mislead.** When the system is not confident (noisy audio,
   failed alignment, unreliable F0), it must say so and suppress the affected
   judgement instead of reporting an error the learner did not make. A false
   "you said this wrong" is worse for learners than a missing hint.
2. **Relative, speaker-normalized acoustics.** Never compare absolute Hz or
   absolute durations across speakers; compare shapes and ratios.
3. **Every score is explainable.** Each aggregate must be traceable to mora-level
   evidence returned in the output.
4. **Deterministic.** Same audio + same text + same version ⇒ identical output.

---

## 2. Public API behavior

### 2.1 Inputs
| Input | Required | Description |
|---|---|---|
| Audio | yes | A file path, bytes, or a float waveform + sample rate. Any common container the decoder supports (WAV, FLAC, MP3, M4A/AAC, OGG). Internally resampled to 16 kHz mono float32. |
| Target text | yes | Japanese in normal orthography (kanji/kana mixed). May contain punctuation, full-width/half-width digits, latin letters. |
| Reading override | no | Kana reading for the whole text or per-word (furigana), used to resolve ambiguous readings (e.g. 今日, 行った, 方). If supplied it takes precedence over automatic reading. |
| Accent override | no | Per-word accent nucleus position, to override dictionary predictions. |
| Reference audio | no | A native recording of the same text. When provided, pitch and timing can additionally be compared against it (§4.4.6). |
| Options | no | Feature toggles (pitch accent on/off, timing on/off), strictness level, and whether to return frame-level debug data. |

### 2.2 Outputs
A single result object (serializable to JSON), containing:

**Sentence level**
- `overall` — 0–100.
- Sub-scores, each 0–100 or `null` if abstained:
  - *segmental accuracy* (phoneme/mora correctness),
  - *completeness* (fraction of target morae actually produced),
  - *pitch accent* (accent-phrase correctness),
  - *timing* (special-mora duration and speaking rhythm).
- `confidence` — 0–1, how much the system trusts its own judgement for this input.
- `warnings` — list of machine-readable codes (see §2.3).

**Word level** (one entry per word in the target text, in order)
- surface form, kana reading, word score, expected accent type, observed accent
  judgement, time span, list of morae.

**Mora level** (one entry per mora)
- kana, phoneme sequence, start/end time (or `null` if deleted),
- segmental score 0–100 and status: `correct` | `mispronounced` | `deleted`,
  plus, when detectable, the most likely substituted phoneme,
- for special morae: duration judgement `ok` | `too_short` | `too_long`,
- pitch: expected level (H/L), observed relative level (semitones vs. speaker
  reference) or `null` if unvoiced/unreliable, and a per-mora pitch status,
- flag whether the vowel is expected to be devoiced.

**Optional debug block**: per-frame F0 (semitones), frame posteriors summary,
alignment path. Off by default (size).

**Insertions** (extra speech not in the target) are reported as a separate list
of time spans with the recognized content when available; they reduce the timing
/ fluency score but not segmental accuracy of target morae.

### 2.3 Error cases
Distinguish **hard errors** (raise an exception, no score) from **soft conditions**
(score returned with warnings / abstentions).

Hard errors:
- Audio cannot be decoded, has zero length, or exceeds the configured maximum
  duration/size.
- Target text is empty after normalization, or contains no Japanese-readable
  content (e.g. only punctuation or emoji).
- Reading override cannot be aligned to the text.

Soft conditions (return result, set warning, lower `confidence`):
- `no_speech` — VAD finds no speech. All sub-scores = 0 (not `null`), so a silent
  submission is never rewarded. Mora statuses = `deleted`.
- `too_quiet`, `clipping`, `low_snr` — quality estimates below/above thresholds.
- `content_mismatch` — the learner evidently said something different (see
  §4.2.4). Segmental scoring still runs but completeness is low and pitch/timing
  are abstained.
- `pitch_unreliable` — too few voiced frames or tracker confidence too low:
  pitch sub-score `null`, pitch statuses suppressed.
- `partial_reading_uncertain` — automatic reading of one or more words was
  ambiguous and no override was supplied.
- `truncated` — speech starts or ends at the file boundary (likely cut-off).

### 2.4 Python binding (v0.1)
```python
hatsuon.score(audio, text, *, reading=None, accent=None, reference_audio=None,
              pitch=True, timing=True, debug=False) -> ScoreResult
```
- `audio`: path, encoded bytes, or a `(samples, sample_rate)` tuple. A bare array
  is rejected (`TypeError`) because its sample rate would be a guess.
- `reading`: kana for the whole text, or `{word_surface: kana}` for single words.
- `accent`: `{word_surface: nucleus}`; 0 = heiban, n = fall after mora n.
- `reference_audio`: accepted but not used yet (§4.4.6 is post-v0.1).
- Result types (`ScoreResult`, `WordResult`, `MoraResult`, `Insertion`, the status
  enums and `Warning`) live in `hatsuon.types`; `MoraResult.pitch_status` is set
  only on morae adjacent to an accent decision point.
- `debug=True` adds a dict with at least: `f0_st` (per-frame semitones),
  `free_decode`, `match_ratio`, `alignment` (frame path), `phones` (per target
  phone: token, word, mora, raw GOP, core log-posterior, deleted) and
  `timing_ratios` (special-mora duration / mean mora). Its keys are diagnostic and
  may change between minor versions.
- Hard errors: `hatsuon.AudioError`, `hatsuon.TextError`; a missing acoustic model
  raises `ModelNotFoundError` (create it once with `hatsuon-export`).

---

## 3. Processing pipeline overview

```
 audio ──► [A] Audio preprocessing ──► waveform 16k mono, speech region, quality flags
                                              │
 text ───► [B] Text front-end ──► words → morae → phonemes, accent phrases,
                                  expected H/L per mora, devoicing flags
                                              │
                 ┌────────────────────────────┴───────────────┐
                 ▼                                            ▼
     [C] Acoustic model (frame posteriors)          [F] F0 extraction
                 │                                   (semitone, voicing prob.)
                 ▼                                            │
     [D] Alignment: forced (target) + free decode             │
         → phone/mora time spans, mismatch check              │
                 │                                            │
                 ▼                                            │
     [E] Segmental scoring (GOP) + duration checks            │
                 │                                            │
                 └─────────────►  [G] Pitch accent assessment ◄┘
                                              │
                                              ▼
                                [H] Aggregation + calibration
                                              │
                                              ▼
                                         Result object
```

Data passed between stages:
- A→C/F: one waveform (same array for both; never re-load or re-resample).
- B→D: canonical phoneme sequence with mora and word boundaries, plus permitted
  pronunciation variants (§5).
- C→D/E: frame-level log-posteriors over the model's token set.
- D→E/G: time span per phoneme and per mora, alignment confidence per segment.
- F→G: per-frame F0 (semitones relative to speaker reference), voicing
  probability.
- E,G→H: per-mora scores and statuses with confidence.

---

## 4. Per-stage specification

### 4.0 Audio preprocessing (stage A)
- Decode, downmix to mono, resample to 16 kHz with a high-quality resampler.
- DC removal; peak-normalize for analysis only (store scale factor; do not use
  normalized loudness as a score feature).
- **Voice activity detection** (energy-based or a small neural VAD such as
  Silero VAD) to find the speech region; trim leading/trailing silence with a
  small safety margin (≈100–200 ms) so plosive onsets and final devoiced vowels
  are not cut.
- Quality estimates: clipping ratio (fraction of samples near full scale), RMS
  level of speech region, a rough SNR (speech-region energy vs. non-speech energy).
  Thresholds are product decisions; expose them as options.
- **Do not denoise by default.** Spectral denoisers create artifacts that lower
  GOP for clean speech and distort F0. Offer it as an option only.

### 4.1 Text front-end (stage B)

#### 4.1.1 Normalization
- Unicode NFKC (full-width digits/latin → half-width, half-width katakana →
  full-width).
- Strip punctuation and symbols for scoring, but keep their positions — sentence
  and clause punctuation are prosodic phrase boundaries (§4.4).
- Digits: convert to kana reading with counter-aware rules (三本 → さんぼん,
  一分 → いっぷん, 八百 → はっぴゃく). Sound changes such as rendaku and gemination
  after numbers are frequent; prefer the front-end's own number handling and add
  tests for common counters.
- Latin letters / acronyms: either read as katakana letter names or mark as
  unscorable spans (configurable). Unscorable spans are excluded from scores
  but still shown in output.

#### 4.1.2 Reading and accent prediction
Recommended public tool: **pyopenjtalk** (Python binding of OpenJTalk, which bundles
a morphological analyzer and an accent-annotated dictionary). It provides:
- word segmentation, part-of-speech and kana reading per word,
- phoneme sequence (`g2p`), where **devoiced vowels are marked** (upper-case vowel
  symbols) — use this as the devoicing prediction,
- full-context labels, from which accent-phrase boundaries, mora position within
  the accent phrase and the **accent nucleus position** can be read,
- built-in accent sandhi rules for compounds and attached particles.

Alternatives / complements: UniDic-based analyzers (UniDic carries accent type and
accent-connection fields), and the public Japanese accent dictionaries derived from
NHK-style data if license-compatible. Treat any dictionary prediction as
*expected but approximate*.

Reading override handling: if the caller supplies furigana, align it to the
surface text at word/character level, replace the analyzer reading, and re-derive
morae and phonemes. Accent for overridden words: look up the reading in the
dictionary if possible; otherwise mark accent as unknown (pitch judgement for that
word is abstained, not guessed).

#### 4.1.3 Mora segmentation
Rules (operate on kana reading after converting katakana to hiragana or vice versa
consistently):
- A mora is one kana, **except** that a small ゃゅょ / ぁぃぅぇぉ / ゎ attaches to the
  preceding kana (きゃ = 1 mora).
- っ (sokuon) is **its own mora**.
- ん (moraic nasal) is **its own mora**.
- ー (long-vowel mark) is **its own mora**; a doubled vowel that forms a long vowel
  (おかあさん, とうきょう → [toːkjoː]) is also two morae.
- Note the distinction between morae and syllables: とうきょう has 4 morae but
  2 syllables. Pitch accent is defined over morae, but the accent nucleus can
  only fall on the *first* mora of a heavy syllable (§5).

Each mora is mapped to its phoneme sequence (consonant + optional glide + vowel,
or a single special phoneme for っ/ん/ー), with the vowel identity of ー inherited
from the previous mora.

#### 4.1.4 Expected pitch pattern (Tokyo Japanese)
For an accent phrase of *N* morae with nucleus position *n* (0 = unaccented):
- *n = 1* (atamadaka): H L L … L
- *n = 0* (heiban): L H H … H, and a following particle stays H
- *1 < n ≤ N*: L H … H (up to mora *n*) then L for the rest; if *n = N* (odaka)
  the word itself looks like heiban and the **fall only shows on the following
  particle** — so heiban vs. odaka is only assessable when a particle follows.
- Initial lowering (first mora L) is weak or absent when the first syllable is
  heavy (first two morae in one syllable, e.g. long vowel or ん/っ in mora 2).
  Mark the first mora's expected level as "soft" in that case.
- Across accent phrases within an intonation phrase, **downstep** lowers the pitch
  range after an accented phrase; do not compare absolute levels across phrases.

Output of the front-end per mora: expected level H/L, whether it is the nucleus,
whether the level is "soft", accent-phrase id.

### 4.2 Acoustic alignment (stages C, D)

#### 4.2.1 Acoustic model
Requirements: frame-level posteriors over a phoneme or kana inventory that covers
Japanese, robust to non-native speech, runnable on CPU.

Options (all public; verify license and current availability before adopting):
- A multilingual **phoneme-recognition CTC** model fine-tuned from wav2vec2/XLS-R
  (e.g. models fine-tuned on IPA/espeak phone targets). Advantage: phone-level
  posteriors usable directly for GOP, and substitution diagnosis across languages
  (L1 interference shows up as non-Japanese phones).
- A Japanese **kana- or phoneme-level CTC** fine-tune of wav2vec2/XLS-R or
  similar self-supervised models, trained on public Japanese corpora (Common
  Voice ja, JSUT, CSJ where licensed, large web-derived corpora).
- A general ASR model (e.g. Whisper family) is useful for the *free-decode
  mismatch check* (§4.2.4) but its attention-based outputs are **not** good
  frame-level posteriors for GOP; do not use it as the GOP model.

Recommendation: one CTC model over Japanese phonemes (≈ 40–45 units incl. long
vowel, geminate closure, moraic nasal) is the simplest viable core. If training
your own, include a separate unit for geminate closure and for moraic nasal, and
either a length mark or distinct long-vowel units — otherwise duration-based
special mora errors become invisible to the posteriors.

#### 4.2.2 Forced alignment
- **CTC forced alignment** (Viterbi over the target token sequence with blanks),
  e.g. as available in torchaudio's CTC alignment utilities, or a hand-written
  Viterbi.
- Allow **optional tokens**: optional short pause between words, optional
  devoiced vowels (§5), pronunciation variants from the front-end as parallel
  paths (lattice) so a valid variant is never scored as an error.
- Known issue: CTC alignments are **peaky** — a token may occupy one frame with
  blanks around it. Segment *boundaries* (and hence durations) from naïve CTC
  alignment are unreliable. Mitigations:
  - assign blank frames to the neighbouring token by posterior or split at the
    midpoint,
  - prefer models trained with a blank penalty / "less peaky" CTC variants,
  - for duration-critical decisions (sokuon, long vowels) refine boundaries with
    acoustic cues: energy dip (closure), voicing on/off, spectral stability.
  - For offline evaluation of boundary quality, compare against an HMM-based
    aligner (e.g. Montreal Forced Aligner with a public Japanese model).

#### 4.2.3 Alignment confidence
Per segment, record the average log-posterior of the aligned token and the
alignment cost relative to an unconstrained (free) path over the same frames.
Very low values for many consecutive segments indicate the target was not said
there.

#### 4.2.4 Content mismatch detection
Run an unconstrained decode (CTC greedy/beam, or a general ASR model) and compare
its mora/phoneme sequence to the target with an edit-distance alignment.
- If the match ratio is very low (the learner read another sentence, spoke another
  language, or only noise), set `content_mismatch`. Match ratio = matched target
  phones / max(target phones, decoded phones): dividing by the target alone lets a
  short target match scattered phones of a long unrelated utterance.
- Use the edit alignment to label target morae as **deleted** (no counterpart and
  forced alignment confidence near floor) and to find **insertions**.
- Do not use free-decode text as the primary segmental score — ASR language
  models "correct" learner errors toward plausible words, hiding exactly the
  mistakes we want to detect. It is a sanity check and a deletion/insertion
  detector only.

### 4.3 Phoneme/mora-level scoring (stage E)

#### 4.3.1 Goodness of Pronunciation
Use **Goodness of Pronunciation (GOP)** (Witt & Young 2000) adapted to CTC
posteriors. Conceptually, for each aligned target phone *p* over its frames:

  GOP(p) = mean over frames of [ log P(p | frame) − max_q log P(q | frame) ]

restricted to non-blank frames or with blank mass handled explicitly. Variants
worth evaluating:
- alignment-free / CTC-aware GOP that compares the likelihood of the target
  sequence vs. the sequence with *p* replaced or deleted (more robust with peaky
  CTC);
- GOP features fed to a small classifier (logistic regression / gradient boosting)
  trained on labelled L2 data, instead of a single threshold.

Substitution diagnosis: the phone *q* with highest posterior mass where *p* was
expected is reported as the likely substitution when its margin over *p* is clear.

#### 4.3.2 From GOP to a 0–100 score
Raw GOP is not comparable across phones (natives get systematically lower GOP for
some phones, e.g. short or reduced ones). Calibrate **per phone**:
- Collect GOP distributions for each phone on native read speech.
- Map a learner's GOP to a score via its position in the native distribution
  (e.g. percentile or z-score → logistic squashing). A phone produced as well as a
  typical native ≈ 100; below the native lower tail ≈ low score.
- Status thresholds (`correct` / `mispronounced`) are chosen on labelled L2 data to
  hit a target precision for the "mispronounced" label (favour precision, per
  design principle 1).

Mora score = combination of its phone scores (min or weighted mean; vowel phones
usually carry more perceptual weight for Japanese learners than onsets — validate).
v0.1 uses the min (worst phone): with the mean, a fully wrong onset plus a correct
vowel scored ≈ 50 and was never flagged.

#### 4.3.3 Special-mora timing
Duration is the key cue for っ, ー/long vowels and ん, and posteriors alone often
miss these errors.
- Estimate **local speaking rate** as morae per second over the accent phrase or
  utterance (excluding pauses), then express each special mora's duration in
  units of the speaker's average mora duration.
- Rough expectations from Japanese phonetics literature (calibrate!):
  - long vowel ≈ 1.5–2.5× a short vowel of the same speaker;
  - geminate closure (っ) clearly longer than a singleton closure, commonly in the
    range ≈ 1.5–3× singleton;
  - ん ≈ about one mora, with place assimilated to the following consonant.
- Judgement: `too_short` if below a lower ratio bound, `too_long` if above an upper
  bound; otherwise `ok`. Bounds should be learned from native data at several
  speaking rates.
- Common learner errors to target: missing sokuon (きて vs. きって), shortened long
  vowels (おばさん vs. おばあさん), ん merged into neighbouring vowel or realized
  as full [n]+vowel.
- If boundary confidence for the relevant segment is low, abstain on duration.

### 4.4 Pitch accent assessment (stages F, G)

#### 4.4.1 F0 extraction
- Use a robust tracker: **pYIN** (Mauch & Dixon 2014; reference implementation in
  librosa), or a neural tracker (CREPE, RMVPE, FCPE) if CPU budget allows.
- Search range: wide enough for adult male to child voices but not so wide as to
  invite octave errors; a two-pass approach works well — first pass with a wide
  range, then re-run or constrain around the speaker's estimated median.
- Frame hop ≈ 10–16 ms.
- Post-processing: discard frames with low voicing probability, remove octave
  jumps (frame-to-frame jump > ~half an octave that reverts quickly), light median
  smoothing. Do not interpolate across long unvoiced spans for scoring; small gaps
  (≲ 50 ms) can be interpolated for display.
- **Compute F0 once** per request and share it with all consumers.

#### 4.4.2 Speaker normalization
Convert Hz to **semitones relative to a speaker reference** (median F0 over the
utterance's voiced frames): st = 12·log2(f / f_ref). All pitch comparisons are on
this scale, so a low-voiced and a high-voiced speaker with the same contour get the
same result.

#### 4.4.3 Per-mora pitch value
For each mora, take a robust statistic (median, or the value at the mora's
vowel midpoint/last third) of voiced semitone frames **within the vowel portion**.
Consonant regions and the first few frames after a voiceless consonant are
excluded (micro-prosodic perturbation). Morae with too few voiced frames
(devoiced vowels, っ closure, whispered) get `null`.

#### 4.4.4 Judging the accent pattern
Work **per accent phrase** and **use local relations, not absolute levels**:
- Compute deltas between consecutive non-null morae.
- Classify each delta as rise / fall / level using a dead-zone (start around
  1–2 semitones; Tokyo accent falls are typically several semitones, but learners
  and fast speech reduce them).
- The **primary cue is the location of the main fall** (the step from the nucleus
  mora to the next). Secondary cue: the initial rise from mora 1 to mora 2
  (unless marked "soft").
- **Peak delay:** native F0 peaks often lag into the mora after the nucleus (JSUT:
  ~53% of atamadaka, ~24% of nakadaka phrases), so the fall may also be found on
  the step starting up to `peak_delay` morae later (calibrated value: 1). With
  mora-level values this means nucleus *k* and *k+1* cannot be told apart; the
  evaluation reports that false-accept rate separately.
- Per accent phrase outcome:
  - `correct` — fall found where expected (or no fall for heiban, with following
    particle level/high);
  - `wrong_nucleus` — a clear fall exists but at a different mora;
  - `flat_for_accented` — expected a fall but contour stays level or rises;
  - `fall_for_unaccented` — clear fall where heiban expected;
  - `unknown` — insufficient voiced morae near the decision points, or accent
    type unknown.
- Global declination: either fit and remove a linear trend per intonation phrase
  before judging, or rely only on local deltas around decision points (simpler
  and usually sufficient).
- Per-mora pitch status (`ok` / `too_high` / `too_low` / `unknown`) can be derived
  by comparing observed relative level to the expected H/L template, but only
  around decision points; elsewhere mark `ok` or `unknown` to avoid noisy hints.

#### 4.4.5 Confidence gate for pitch
If the fraction of morae with reliable F0, or the segmental/alignment confidence
of the phrase, is too low, the whole pitch judgement for that phrase (or
utterance) is set to `unknown` and excluded from the pitch sub-score. Rationale:
pitch errors reported on top of misaligned or unrecognized speech are nearly
always wrong.

#### 4.4.6 Optional: comparison with reference audio
When a native reference recording is supplied:
- Align learner and reference with **DTW** over MFCC (or self-supervised)
  features, or better, force-align both to the same target and map mora to mora.
- Compare semitone contours (each normalized to its own speaker reference) per
  mora: correlation of shapes and mean absolute difference in semitones.
- Use reference-based comparison to *confirm* dictionary-based judgements and to
  cover words whose dictionary accent is uncertain. Report which source was used.

### 4.5 Score aggregation (stage H)

#### 4.5.1 Mora → word
- Word segmental score = mean of its mora scores, with deleted morae counted as 0.
- Word timing score = share of its special morae judged `ok` (null if none).
- Word pitch = outcome of its accent phrase (a word may share an accent phrase
  with particles; attribute the phrase outcome to its content word).

#### 4.5.2 Word → sentence
- Segmental accuracy = mora-count-weighted mean over words (excluding unscorable
  spans).
- Completeness = produced morae / target morae.
- Pitch = fraction of assessable accent phrases judged `correct`, possibly with
  partial credit (e.g. `wrong_nucleus` worse than a small misplaced rise). `null`
  if none assessable.
- Timing = combination of special-mora outcomes and pause behaviour (long
  intra-phrase pauses, many insertions).

#### 4.5.3 Overall
- Overall = weighted combination of sub-scores, with weights exposed as
  configuration. The library should not hard-wire a policy; provide a sensible
  default and document it.
- Recommended default: completeness acts as a multiplier or cap (you cannot score
  highly on half a sentence); abstained sub-scores are dropped and weights
  renormalized, never treated as 100.
- **Calibration**: fit the final mapping (e.g. isotonic or logistic regression from
  sub-scores to human ratings) on rated data so that score bands mean consistent
  things. Re-fit when the acoustic model changes.
- `no_speech` ⇒ all sub-scores 0, overall 0.

---

## 5. Japanese-specific edge cases

| Case | Expected handling |
|---|---|
| **Devoiced vowels** (i/u between voiceless consonants, e.g. ～です, ～ます, きく, した) | Front-end marks them. In alignment, the vowel is optional (may be absent or very short). In segmental scoring, presence *or* absence is acceptable. In pitch, the mora has no F0 → `null`; if the nucleus is on a devoiced mora, the fall appears shifted to the next voiced mora — accept that. |
| **Sokuon っ** | Separate mora with a closure (stops) or lengthened frication (s, sh). Score mainly by duration (§4.3.3), secondarily by posterior of the closure unit. No F0 → `null` pitch. Word-final っ (exclamations) — treat as glottal stop; usually unscorable. |
| **Moraic nasal ん** | Separate mora; place of articulation assimilates ([m] before p/b/m, [ŋ] before k/g, nasalized vowel before vowels/semivowels/fricatives). Accept all allophones; do not penalize the assimilated variant. Common error: realized as [n]+vowel (extra mora) or dropped. |
| **Long vowels** | Two morae. Both ー and vowel doubling (えい often pronounced [eː], おう → [oː]) must allow the spoken long-vowel variant. えい read as [ei] is also acceptable in careful speech — allow both. |
| **Particles は/へ/を** | Read as wa / e / o. Front-ends normally handle this; add tests. |
| **Ambiguous readings** (今日, 明日, 行った, 方, 上手) | Use override if given; otherwise use the analyzer's choice, set `partial_reading_uncertain`, and if the free decode clearly matches an alternative reading, accept it rather than marking errors. |
| **Numbers and counters** | Gemination and rendaku (いっぽん, さんぼん, ろっぽん). Test tables required. |
| **Katakana loanwords** | Long vowel and sokuon density is high; dictionary accent for loanwords is often antepenultimate-mora based; coverage of new loanwords may be missing → accent `unknown`. |
| **Heavy first syllable** | Initial rise is weak/absent; do not penalize missing L→H on the first mora. |
| **Odaka vs. heiban** | Indistinguishable without a following particle; mark `unknown` when phrase-final. |
| **Compounds and accent sandhi** | Rely on front-end sandhi rules; treat as approximate; reference audio (§4.4.6) is the fallback. |
| **ガ行鼻濁音 ([ŋ] for medial g)** | Both [g] and [ŋ] acceptable. |
| **Sentence-final rising intonation** (questions) | Utterance-final rise can mask accent fall on the last phrase; exclude the final mora(s) of a question from accent judgement or model the boundary tone. |
| **Very fast or very slow speech** | Duration ratios use local speaking rate; pitch dead-zone may need to scale with phrase pitch range. |
| **Repetitions / self-corrections** | The learner says a word twice. Forced alignment must allow optional repetitions or the free-decode diff must detect insertions; score the *last* complete attempt of a word (document this choice). |

---

## 6. Performance requirements and deployment constraints (CPU)

### 6.1 Targets (suggested)
- Commodity CPU, 2–4 cores per request worker, no GPU.
- Latency target: under ~1–2 s for a 5–10 s utterance end-to-end; real-time
  factor ≲ 0.2.
- Memory: models + runtime ≲ 1–2 GB per worker.

### 6.2 What makes it acceptable
- **Model size**: a base-size self-supervised encoder (~95M params) fine-tuned with
  CTC is usually enough; large (300M+) models give better L2 robustness but cost
  3–4× CPU. Benchmark both on your data.
- **Export and quantize**: ONNX export with dynamic INT8 quantization typically
  gives 2–4× CPU speed-up with small accuracy loss; verify GOP calibration after
  quantization (posterior sharpness changes).
- **Single pass per stage**: decode and resample once, run the acoustic model once
  (forced alignment and free decode share the same posteriors when using the same
  CTC model), compute F0 once.
- **F0 tracker cost**: probabilistic trackers with Viterbi decoding are much slower
  than plain YIN/autocorrelation; restrict the frequency search range, restrict
  analysis to the speech region, and consider a vectorized implementation. Profile
  — this stage can dominate if done naïvely.
- **Thread control**: cap intra-op threads per worker (BLAS/ONNX runtime) so
  concurrent requests don't oversubscribe cores; set a concurrency limit for
  CPU-heavy DSP stages.
- **Warm-up**: load models once at process start and run a dummy inference to avoid
  first-request latency spikes.
- **Front-end caching**: memoize text analysis per target text (texts repeat
  heavily in lesson-based usage).
- **Determinism**: fixed seeds, deterministic ops, fixed thread counts where
  numerics depend on reduction order; add a determinism test (same input twice ⇒
  byte-identical output).

---

## 7. Evaluation methodology

### 7.1 Data
- **Native read speech** (e.g. JSUT, JVS, Common Voice ja) — for GOP calibration
  per phone, duration-ratio norms, and accent-detection sanity checks (natives
  should score high; measure false-alarm rate).
- **L2 read speech** with human annotation — e.g. publicly available non-native
  Japanese read-speech corpora (check availability/licensing via the relevant
  speech resource consortia), or a small in-house annotated set collected under
  explicit consent for the open-source project.
- **Synthetic error injection** for unit-level tests: delete っ, shorten long vowels
  (time-scale a segment), swap phonemes via TTS, shift pitch nucleus via pitch
  manipulation (e.g. PSOLA). Useful for regression tests, not for final metrics.

### 7.2 Human labels
- Multiple native raters (≥ 3) per item; labels at mora level (correct /
  mispronounced / deleted, duration error), accent-phrase level (correct / wrong
  nucleus / flat), and a holistic 1–5 or 0–100 rating.
- Report inter-rater agreement (Cohen's/Fleiss' κ for labels, Pearson/Spearman
  or ICC for holistic). Machine-vs-human agreement should be judged relative to
  human-vs-human agreement.

### 7.3 Metrics
- Mispronunciation detection: precision, recall, F1 at mora level; report
  **precision at fixed recall** since false alarms are the primary harm.
- Special-mora errors: per-category precision/recall.
- Pitch accent: accent-phrase accuracy vs. labels; confusion matrix of outcomes;
  abstention rate (and accuracy on non-abstained items).
- Holistic: Pearson and Spearman correlation of `overall` with mean human rating;
  MAE after calibration.
- Native false-alarm rate: fraction of native utterances given any
  `mispronounced` label.
- Robustness: metrics stratified by SNR bucket, device type, speaker L1, gender.
- Performance: p50/p95 latency and RTF on the target CPU; memory.
- Determinism: repeat runs must be identical.

### 7.4 Comparison against a commercial API
If benchmarking against a commercial pronunciation-assessment service that
supports Japanese:
- Same audio set, same reference texts, default settings on both sides.
- Compare each system's holistic score with human ratings (correlation), and
  system-vs-system agreement on flagged words/morae.
- Report cost per 1,000 requests and latency for both, with the deployment
  assumptions spelled out.
- Respect the service's terms of use regarding benchmarking and data retention;
  publish only aggregate numbers.

### 7.5 Regression suite
Keep a frozen set of audio + expected outputs (with tolerances) covering every row
of §5; run it on every model or threshold change.

---

## 8. Known limitations and open problems

1. **L2 acoustic mismatch**: acoustic models trained on native speech produce lower
   posteriors for accented but intelligible speech, inflating error rates for
   some L1 groups. Mitigation: include L2 data in fine-tuning; per-L1 calibration.
   *Open question*: whether per-L1 calibration is fair or desirable for a learning
   product.
2. **Peaky CTC boundaries** make duration measurement fragile (§4.2.2). *Open
   question*: is a separate boundary-refinement model worth its CPU cost?
3. **Dictionary accent is approximate**, especially for compounds, loanwords,
   proper nouns and numbers. Without reference audio, accent feedback on these is
   low-confidence.
4. **Accent vs. intonation**: question intonation, focus and emphasis legitimately
   change pitch. v1 does not model focus; it may flag emphatic but correct speech.
5. **Devoiced nuclei** hide the accent fall; judgement may shift or abstain.
6. **Children and very high voices**: F0 trackers and acoustic models degrade.
7. **Very short utterances** (one or two morae): too little context for speaking
   rate and speaker F0 reference; timing and pitch should usually abstain.
8. **Repetitions / partial reads**: the "score the last attempt" policy is a design
   choice that some users will find unfair. *Open question*.
9. **Weighting of pitch in the overall score**: whether pitch accent should lower
   the overall score or be reported separately is pedagogically debatable.
   *Open question*; keep it configurable, default to reporting separately with low
   weight.
10. **GOP threshold transfer**: thresholds learned on read speech in quiet rooms do
    not transfer cleanly to phone microphones in noisy environments; quality-aware
    calibration is an open item.
11. **Front-end reading errors show up as deletions.** If the analyzer misreads a
    word (on JVS: 命 read イノチ but spoken めい, 大西洋 read オーニシヨー, 乾 read イヌイ,
    鋼製 read ハガネセー), the correct native reading is scored against the wrong
    target, and the word's morae come out `deleted`.
    - In v0.1 the only remedy is the caller's `reading` override.
    - The §5 rule "accept a free decode that clearly matches an alternative
      reading" is not implemented yet.
12. **Pitch peak delay (v0.1).** A fall up to one mora after the nucleus is
    accepted (§4.4.4), because native speakers often place the F0 peak that late.
    The cost is that a nucleus placed one mora too late cannot be detected.

---

## 9. Out of scope for the open-source version

- Any learner-facing feedback text, coaching messages, tips on articulation, and
  their localization.
- Lesson/curriculum content, task types and their product-specific scoring
  policies (e.g. how scores map to pass/fail or levels).
- LLM-based judging or feedback rewriting.
- Web service layer: authentication, rate limiting, storage/retention of user
  audio, tracing, deployment manifests.
- UI rendering contracts (charts, colour coding) beyond the neutral data in §2.2.
- Any pretrained weights or thresholds derived from non-public data.

---

## Appendix A. Tuning playbook

Every tunable value in this spec must be derived with the procedure below, on
public or consented data, and recorded (value, dataset, date, metric) in a
versioned calibration file shipped with the library. Ranges given are search
ranges for a sweep, not answers.

### A.1 General procedure
1. Split data by **speaker** (never by utterance) into tune / test sets.
2. For each parameter, sweep a grid over the search range while holding others at
   defaults; then do one joint refinement pass for strongly interacting pairs
   (marked ⇄ below).
3. Optimize the stated metric on the tune set; report on the test set only once.
4. Prefer the value at the **edge of the plateau** that favours abstention /
   precision, not the single best point (robustness to new devices).
5. Re-run the whole playbook whenever the acoustic model, F0 tracker, or
   quantization changes.

### A.2 Parameter table

| Parameter | Controls | Data | Metric to optimize | Search range |
|---|---|---|---|---|
| VAD trim margin | Not cutting onsets / final devoiced vowels | Native + L2 | Deletion false-alarm rate on first/last mora | 50–300 ms |
| Low-SNR / too-quiet / clipping thresholds | When to warn and lower confidence | L2 + noise-augmented native (MUSAN-style noise at several SNRs) | Point where native false-alarm rate starts rising sharply | SNR 5–20 dB |
| F0 search range (pass 1) | Coverage vs. octave errors | Native, both genders, several ages | Gross pitch error vs. a reference tracker or laryngograph data if available | lower 50–80 Hz, upper 400–700 Hz |
| F0 pass-2 window around median | Octave error suppression | Same | Gross pitch error | ±0.7–1.2 octave |
| Voicing decision | Which frames count as voiced | Same | Voicing decision error; share of voiced morae left without F0 | pYIN HMM voiced flag (v0.1: a raw-probability cut-off at 0.3 left 45 % of morae without F0, because pYIN's per-frame probability is uncalibrated; the HMM decision left 10.5 %) — revisit only with a reference tracker |
| Octave-jump filter | Spurious jumps | Same | Gross pitch error without over-smoothing real falls | jump > 5–8 st reverting within 2–5 frames |
| Pitch dead-zone ⇄ main-fall threshold | Rise/fall/level classification | Native with known accent types (dictionary-checked) + L2 labelled | Accent-phrase accuracy on native ≥ target, then L2 F1 | 0.5–3 st |
| Pitch confidence gate (min fraction of morae with reliable F0, min alignment confidence) | When to abstain on pitch | L2 labelled | Accuracy on non-abstained phrases vs. abstention rate (choose knee of curve) | 30–80 % |
| GOP per-phone calibration | GOP → 0–100 | Native read speech, ≥ a few hundred tokens per phone | Native mean score ≈ ceiling, stable across speakers | percentile/z-score mapping; logistic slope swept. With a peaky CTC model, native GOP is a point mass at 0 (≈ 98.6 % of tokens), so a per-phone z-score is degenerate. v0.1 uses one uniform scale for all phones, in closed form: with min-combination, "mora < threshold" ⇔ "worst GOP < scale·z_thr", so scale = the native per-mora worst-GOP quantile at the target false-alarm rate (0.5 %), divided by z_thr. Check it with synthetic one-mora substitutions on the same audio (B.4): detection rate on held-out speakers. Before fitting, exclude words the front-end misreads: a word deleted by many speakers of the same prompt is a reading error. |
| `mispronounced` threshold (per phone or global) | Error flagging | L2 labelled | Max recall s.t. precision ≥ target (e.g. 0.8–0.9) | over calibrated score 20–70. v0.1 (no L2 data): fixed at 30, with the GOP scale fitted so that native morae fall below it at 0.5 % |
| Substitution-report margin | When to name the substituted phone | L2 labelled with substitution labels | Precision of named substitutions | 0.5–3 nats log-posterior margin |
| Long-vowel ratio bounds | `too_short` / `too_long` | Native at slow/normal/fast rate + L2 | Per-category F1, native false-alarm ≤ target | lower 0.4–1.6×, upper 2.5–4× mean mora. (The lower range was 1.1–1.6× in the first draft. The ratio's denominator is a whole mora, consonant included, and on the JSUT HTS labels a whole native long vowel is 1.06 mean morae (p5 0.66), while a single vowel, which is what a shortened long vowel becomes, is 0.44 (p95 1.04).) |
| Sokuon closure ratio bounds | Missing / exaggerated っ | Same | Same | lower 1.2–2×, upper 3–5× singleton closure |
| Content-mismatch match ratio | When to declare `content_mismatch` | L2 + deliberately wrong-sentence recordings | Balanced error between "wrong sentence missed" and "bad but correct attempt flagged" | 0.2–0.7 |
| Deletion decision (forced-align confidence floor ⇄ free-decode diff) | `deleted` status | L2 + synthetic deletions | Deletion F1 | sweep jointly |
| Aggregation weights | Overall score | L2 with holistic human ratings | Correlation with mean human rating, then MAE after isotonic calibration | weights on simplex, coarse grid |

### A.3 Checks after quantization or model swap
- Re-fit GOP calibration (posterior sharpness changes); compare native score
  distributions before/after — they should overlap closely after re-fit.
- Re-check special-mora boundaries: blank/peak behaviour can change with
  quantization.
- Confirm determinism and latency targets (§6).

### A.4 What to record per tuned value
Value, search range, dataset(s) and split, metric and its value on tune/test,
model + tracker version, date. Values without this record must not ship.

---

## Appendix B. Data sources

Researched 2026-11. Licence summaries are short paraphrases for planning, **not
legal advice**. Re-read each official licence page before downloading, and again
before any commercial use or redistribution.

### B.1 Summary

| Need | Best freely obtainable source | Ease |
|---|---|---|
| Native read speech, clean, single speaker | JSUT | Easy (direct download) |
| Native read speech, many speakers | JVS; Common Voice ja | Easy (CV needs an account) |
| Native speech at scale (model fine-tuning) | ReazonSpeech corpus | Easy (gated HF, accept terms) |
| Phoneme / accent labels for native speech | jsut-label (manual), jsut-lab (OpenJTalk-generated) | Easy |
| Native speech with manual prosody / accent labels | CSJ-Core (X-JToBI) | Hard (paid, apply to NINJAL) |
| Word pitch-accent dictionary | Kanjium accent data; pyopenjtalk dictionary | Easy |
| Noise for augmentation / SNR tests | MUSAN | Easy |
| Reference aligner for boundary checks | Montreal Forced Aligner `japanese_mfa` model | Easy |
| **L2 (non-native) Japanese with human labels** | UME-JRF; I-JAS; Korean-learner L2 corpus (LREC-COLING 2024) | **Hard — no open-licence option found** |

Main gap: **no openly licensed, freely downloadable, annotated L2 Japanese
read-speech corpus was found.** All tuning steps in Appendix A that say "L2
labelled" depend on one of the restricted sources in B.3; until access is
granted, use the synthetic fallback in B.4.

### B.2 Native speech and labels

**JSUT** — ~10 h, one female speaker, 48 kHz, read speech covering daily-use
kanji readings. Official page: Takamichi lab site (Google Sites, `jsut`). Paper:
arXiv:1711.00354.
- Text: CC BY-SA 4.0. Audio: academic, non-commercial research (including inside
  companies) and personal use; commercial use needs the author's permission;
  redistribution not allowed (small excerpts allowed).
- Use for: F0 tracker sanity checks, duration norms for one speaker, regression
  fixtures that stay local (do not commit audio to the public repo).

**JVS** — 100 professional speakers (49 M / 51 F), ~30 h, normal / whisper /
falsetto; 22 h parallel normal speech. Official page: Takamichi lab site
(`jvs_corpus`). Paper: arXiv:1908.06248.
- Tags: CC BY-SA 4.0. Audio: same style of terms as JSUT (research / non-commercial,
  no redistribution).
- Use for: **per-phone GOP calibration on natives**, duration-ratio norms across
  speakers, native false-alarm rate, F0 range across voices. The whisper subset is
  a useful "no F0" stress test for the pitch confidence gate.

**Common Voice — Japanese** (Mozilla) — v25.0 (build 2026-03): ~725 h recorded,
~372 h validated, ~7.8k speakers, read sentences, crowd-recorded on varied devices.
- Licence: CC0, but since Oct 2025 downloads go through the **Mozilla Data
  Collective** with an account and terms of service (no attempts to identify
  speakers; some versions also forbid re-hosting).
- Use for: device / noise robustness, many-speaker native GOP distributions,
  low-quality-audio thresholds. Validated clips only for calibration.

**ReazonSpeech corpus** — >35,000 h of Japanese broadcast audio. Hugging Face
dataset `reazon-research/reazonspeech` (gated).
- Licence: CDLA-Sharing-1.0, **restricted to uses permitted under Article 30-4 of
  the Japanese Copyright Act** (information analysis). Its code / models are under
  separate (Apache-2.0) licences.
- Use for: fine-tuning a Japanese phoneme/kana CTC model (§4.2.1). Get legal
  advice before shipping weights trained on it in a commercial setting.

**jsut-label** (GitHub `sarulab-speech/jsut-label`) — HTS-style labels for JSUT
basic5000 with **manually annotated phonetic and prosodic information**; times
auto-estimated. Does not include devoiced vowels or POS. CC BY-SA 4.0.
- Use for: ground-truth accent nucleus per phrase on basic5000 → evaluate the
  front-end accent prediction (§4.1.2) and native accent-phrase accuracy.

**jsut-lab** (GitHub `r9y9/jsut-lab`) — OpenJTalk-generated full-context labels +
automatic alignments for all of JSUT. Alignments may contain errors. Confirm the
licence on the repo before use.
- Use for: quick bootstrap of phone boundaries; not ground truth.

**CSJ (Corpus of Spontaneous Japanese)** — 662 h spontaneous monologue; the 44 h
**CSJ-Core** has segment labels and X-JToBI prosodic labels. Distributed by
NINJAL; paid licence, application required (terms not found online; contact
NINJAL).
- Use for: the best available manual prosody labels; only if budget allows.

### B.3 L2 (non-native) Japanese — restricted

**UME-JRF** — read Japanese speech by foreign students, 16 kHz / 16-bit WAV.
Distributed by the NII Speech Resources Consortium (NII-SRC); **application
required**. Speaker count, L1s, annotation depth and fees were not found online —
check the NII-SRC page and ask before planning around it.

**I-JAS (International Corpus of Japanese as a Second Language)** — NINJAL;
learners with 12 different L1s (~1,000 participants), text and audio released
since 2020. Mostly task-based / interview speech rather than read sentences, so it
fits fluency and robustness work better than read-aloud GOP calibration. Check the
NINJAL I-JAS page for audio access terms; citation of I-JAS is required.

**Korean learners' L2 speech corpus of seven languages** (Han, Kim & Chung,
LREC-COLING 2024) — ~200 h of Japanese read by Korean learners, with
pronunciation transcription and **expert scores for pronunciation accuracy and
prosodic fluency**. Closest public match to our task, but only one L1 (Korean).
Distribution channel and access terms were not confirmed — check the paper and
contact the authors.

**Japanese read-word database for non-native speakers** (Yamakawa, Amano &
Kondo, O-COCOSDA 2014) — 343 words read by 129 non-native speakers (L1: Korean,
Chinese, Thai, Vietnamese, French, English) plus 18 native speakers; the word list
targets **singleton vs. geminate stops, short vs. long vowels, voiced vs.
voiceless consonants**; phoneme labels for part of the data. Ideal for tuning the
special-mora duration rows of Appendix A. Distribution channel not found — contact
the authors.

**Polite Japanese audio dataset** (Griffith University, listed on Research Data
Australia) — polite / non-polite sentences (です/ます) by 18 native and 11
non-native speakers; CC BY-ND. Small; details not verified (page access blocked
during research). Possibly useful as a few extra L2 test items only.

### B.3.1 Access priority (no self-collection)
Send these requests in parallel at project start; expect weeks, not days:
1. Korean-learner L2 corpus (largest, has expert scores) — email the authors.
2. Non-native read-word database (best for っ / long-vowel tuning) — email the
   authors.
3. UME-JRF — apply via NII-SRC.
4. I-JAS — check NINJAL audio access (fluency / robustness only).
State the purpose (open-source research library, only derived statistics and
thresholds will be published, no audio redistributed) — this is what most of these
licences allow.

### B.4 Fallback while L2 access is pending (no recording needed)
- **Synthetic error injection on native data** (JVS / JSUT): delete or shorten
  closures (っ), time-compress long vowels, swap phonemes with TTS, move the accent
  nucleus with an accent-controllable TTS or pitch manipulation (PSOLA). Gives
  labelled errors for free; good for duration and pitch rows of Appendix A, weak for
  realistic L1-interference substitutions.
- **speechocean762** (English, CC BY 4.0) — validates that the GOP implementation
  is correct by reproducing published GOP baselines before trusting it on Japanese.
- Native false-alarm metrics (§7.3) need only native data and can be finalized now.
- Ship v0.x with L2-dependent thresholds marked `provisional` in the calibration
  file.

### B.5 Supporting resources
- **Kanjium** (GitHub `mifunetoshiro/kanjium`) — accent nucleus positions for
  ~124k words (raw `accents.txt`); CC BY-SA 4.0 (share-alike applies to the data,
  check the asterisked exceptions in its README). Use to cross-check or extend the
  front-end's accent dictionary.
- **MUSAN** (OpenSLR SLR17) — ~109 h music / speech / noise, 16 kHz; listed as
  CC BY 4.0 with per-file licences (check before commercial use). Use for noise
  augmentation and SNR-threshold sweeps.
- **MFA `japanese_mfa` acoustic model** (Hugging Face
  `MontrealCorpusTools/japanese_mfa`) — CC BY 4.0 model card; its training data
  includes corpora with their own terms. Use offline only, to judge CTC boundary
  quality (§4.2.2).
- English-L2 benchmarks **L2-ARCTIC** and **speechocean762** — not Japanese, but
  their annotation schemes are good templates for B.4, and speechocean762 is a
  handy sanity check that the GOP pipeline is implemented correctly.

### B.6 Bootstrap tasks for the new repository
1. Write `DATA_SOURCES.md` in the new repo from this appendix; for each source,
   paste the official licence URL and the date it was read.
2. Write download scripts that fetch into a git-ignored data directory; never
   commit audio. Gated sources (Common Voice, ReazonSpeech, UME-JRF, CSJ) must
   prompt the user to accept terms themselves — scripts only check that the files
   exist.
3. Build loaders that output a common manifest (audio path, text, kana reading,
   speaker id, split, labels if any).
4. First calibration run: JVS + Common Voice (native) → GOP per-phone norms,
   duration norms, F0 range (Appendix A rows that need native data only).
5. First accent evaluation: jsut-label basic5000 → front-end accent accuracy and
   native accent-phrase accuracy.
6. Tune "L2 labelled" rows of Appendix A on B.4 synthetic data first and mark them
   `provisional`; re-tune on real L2 data as soon as a B.3 source is granted.
7. Draft the access-request emails for B.3.1 (purpose, scope, what will be
   published) for the maintainer to send.
