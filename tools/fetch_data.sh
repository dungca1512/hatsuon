#!/usr/bin/env bash
# Download public corpora into the git-ignored data/ directory (SPEC Appendix B.6).
# By running this you accept each corpus licence (see DATA_SOURCES.md).
# Usage: tools/fetch_data.sh [jsut-label] [jsut] [jvs]   (default: all three)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
mkdir -p "$DATA/raw"

fetch() {  # fetch URL OUT : resumable download (-C -), fail on HTTP error
  echo ">> $2"
  curl -L --fail -C - --retry 3 -o "$2" "$1"
}

unpack() {  # unpack ZIP DEST : extract once, marker file prevents re-extraction
  [ -f "$2/.done" ] && return 0
  mkdir -p "$2"
  unzip -q -o "$1" -d "$2"
  touch "$2/.done"
}

get_jsut_label() {
  fetch https://github.com/sarulab-speech/jsut-label/archive/refs/heads/master.zip \
    "$DATA/raw/jsut-label.zip"
  unpack "$DATA/raw/jsut-label.zip" "$DATA/jsut-label"
}

get_jsut() {
  fetch http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip "$DATA/raw/jsut_ver1.1.zip"
  unpack "$DATA/raw/jsut_ver1.1.zip" "$DATA/jsut"
}

get_jvs() {
  # Google Drive only; confirm=t skips the "large file" interstitial.
  fetch "https://drive.usercontent.google.com/download?id=19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt&export=download&confirm=t" \
    "$DATA/raw/jvs_ver1.zip"
  unpack "$DATA/raw/jvs_ver1.zip" "$DATA/jvs"
}

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(jsut-label jsut jvs)
for t in "${targets[@]}"; do
  case "$t" in
    jsut-label) get_jsut_label ;;
    jsut) get_jsut ;;
    jvs) get_jvs ;;
    *) echo "unknown target: $t" >&2; exit 2 ;;
  esac
done
echo "done: ${targets[*]}"
