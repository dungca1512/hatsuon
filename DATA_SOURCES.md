# Data sources

Corpora used for calibration and evaluation (SPEC Appendix B). Nothing listed here
is redistributed by this repository: audio lives in the git-ignored `data/`
directory and is fetched by `tools/fetch_data.sh`. Licence summaries are
paraphrases for planning, not legal advice — re-read the official page before use.

| Source | Official page | Licence (summary) | Read on | Used for |
|---|---|---|---|---|
| JSUT v1.1 | https://sites.google.com/site/shinnosuketakamichi/publication/jsut | Text CC BY-SA 4.0. Audio: academic / non-commercial research (incl. inside companies) / personal use; no redistribution (small excerpts allowed); commercial use by contacting the authors. | 2026-11-04 | F0 and duration sanity checks, local regression fixtures |
| JVS v1 | https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus | Tags CC BY-SA 4.0. Audio: same terms as JSUT (non-commercial research, no redistribution). | 2026-11-04 | Per-phone GOP calibration, duration norms, native false-alarm rate |
| jsut-label | https://github.com/sarulab-speech/jsut-label | CC BY-SA 4.0 | 2026-11-04 | Ground-truth accent nucleus for basic5000 (front-end accent accuracy) |

## Models

| Model | Page | Licence | Read on | Notes |
|---|---|---|---|---|
| japanese-hubert-base-phoneme-ctc-v4 | https://huggingface.co/prj-beatrice/japanese-hubert-base-phoneme-ctc-v4 | Apache-2.0 | 2026-11-04 | Fine-tuned from rinna/japanese-hubert-base on ReazonSpeech (CDLA-Sharing-1.0, Art. 30-4 restricted — see SPEC B.2). Not redistributed; users export it locally with `hatsuon-export`. |
