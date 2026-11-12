#!/bin/bash
# setup.sh DATA [options]: everything the engine needs besides itself, in the folder DATA (run build.sh first):
#
#   DATA/strata-v0.2.9    Strata's server and tools at v0.2.9, with their Python environment (.venv)
#   DATA/models           the model's two IQ3_S shards and the picture projector (85 GB), checked against SHA-256
#   DATA/prep/tokenizer   the tokenizer folder, from the GGUF's metadata (Strata's tools/strata_tokenizer.py)
#   DATA/prep/mtp-rt      the MTP draft layer: 5.2 GB read from Qwen's checkpoint and packed (Strata's tools/mtp_*.py)
#   DATA/strata-v0.2.9/build-vision/bin/strata-vision   the picture encoder (not with --no-pictures)
#   DATA/profile.json     the run config (profile/fill_profile.py); an existing one is kept
#
# A step whose result is already there is skipped, so after an interruption it is simply run again.
#
#   bash setup.sh /data/flash-next
#   bash setup.sh /data/flash-next --check                    # the checks and what is still to do; changes nothing
#   bash setup.sh /data/flash-next --models /mnt/models --no-pictures --route-prior 0.15
#
# Options:
#   --models DIR     the model files somewhere else (for example already downloaded); missing ones are downloaded there
#   --strata DIR     an existing Strata checkout at v0.2.9 instead of DATA/strata-v0.2.9
#   --no-pictures    no projector download, no encoder build, a profile without pictures
#   --route-prior X  FNX_ROUTE_PRIOR in the profile: 0 (default) is exact; 0.15 is our production setting, faster, lossy
#   --check          only check and report
# Environment: HF_TOKEN (sent to Hugging Face if set), CUDA, HOSTCXX (as for build.sh), JOBS (encoder build, default 6).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
say() { echo "setup.sh: $*"; }
die() { echo "setup.sh: $*" >&2; exit 1; }

STRATA_REPO=https://github.com/Niko1221/Strata.git
STRATA_TAG=v0.2.9
STRATA_COMMIT=c62d5e3a15b40ea51c96aae9e8d36928965f53ab
HF_REPO=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
HF_REV=ed59f92082b1e93c0e96d60a8b11aab089b52f09        # the revision Strata v0.2.9 and v0.2.10 pin
SHARD1=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
SHARD2=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
MMPROJ=mmproj-Qwen3.8-Flash-Next-BF16.gguf
declare -A SIZE=([$SHARD1]=54817524224 [$SHARD2]=28800138432 [$MMPROJ]=907543008)
declare -A SHA=([$SHARD1]=4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3
                [$SHARD2]=316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113
                [$MMPROJ]=b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0)

[ $# -ge 1 ] && [ "${1#-}" = "$1" ] || die "usage: bash setup.sh DATA [--models DIR] [--strata DIR] [--no-pictures] [--route-prior X] [--check]"
DATA=$1
shift
MODELS='' STRATA='' PICTURES=1 PRIOR=0 CHECK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --models) MODELS=${2:?--models DIR}; shift 2 ;;
    --strata) STRATA=${2:?--strata DIR}; shift 2 ;;
    --no-pictures) PICTURES=0; shift ;;
    --route-prior) PRIOR=${2:?--route-prior X}; shift 2 ;;
    --check) CHECK=1; shift ;;
    *) die "unknown option $1" ;;
  esac
done
abspath() { case "$1" in /*) echo "$1" ;; *) echo "$PWD/$1" ;; esac; }
DATA=$(abspath "$DATA")
MODELS=$(abspath "${MODELS:-$DATA/models}")
STRATA=$(abspath "${STRATA:-$DATA/strata-v0.2.9}")
PREP=$DATA/prep
EXE=$HERE/build/split/strata-split-2
GG=$HERE/third_party/llama.cpp
NEEDED=("$SHARD1" "$SHARD2")
[ "$PICTURES" = 1 ] && NEEDED+=("$MMPROJ")
TODO=0
todo() { TODO=$((TODO + 1)); say "to do: $*"; }
ok() { say "ok: $*"; }
gb() { awk -v b="$1" 'BEGIN {printf "%.1f GB", b / 1e9}'; }

# ------------------------------------------------------------------ checks
[ -x "$EXE" ] || die "no $EXE: run 'bash build.sh' first"
[ -d "$GG/gguf-py/gguf" ] || die "no $GG/gguf-py: run 'bash build.sh' first (it fetches llama.cpp)"
for t in git curl sha256sum python3; do command -v $t >/dev/null || die "$t is needed"; done
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "Python 3.10 or newer is needed"

GPU_LIST=0,1
if command -v nvidia-smi >/dev/null; then
  p100=$( (nvidia-smi --query-gpu=index,name --format=csv,noheader || true) | awk -F', ' '/P100/ && n < 2 {print $1; n++}' | paste -sd, -)
  if [ "$(echo "$p100" | tr ',' '\n' | grep -c .)" = 2 ]; then
    GPU_LIST=$p100
    ok "two Tesla P100: cards $GPU_LIST"
  else
    say "WARNING: nvidia-smi does not show two Tesla P100 (found: ${p100:-none}); the profile names cards 0,1"
  fi
else
  say "WARNING: no nvidia-smi; the profile names cards 0,1"
fi
mem_gib=$(awk '/^MemTotal:/ {print int($2 / 1048576)}' /proc/meminfo)
if [ "$mem_gib" -lt 88 ]; then
  say "WARNING: $mem_gib GiB of RAM; this engine needs 96 GB (about 85 GiB once loaded, more while loading)"
else
  ok "$mem_gib GiB of RAM"
fi

# disk: what the missing model files and the MTP preparation still need
need=0
for f in "${NEEDED[@]}"; do
  have=$(stat -L -c %s "$MODELS/$f" 2>/dev/null || stat -L -c %s "$MODELS/$f.part" 2>/dev/null || echo 0)
  need=$((need + ${SIZE[$f]} - have))
done
[ -f "$PREP/mtp-rt/experts.bin" ] || need=$((need + 7 * 1024 * 1024 * 1024))
avail_on() { local d=$1; while [ ! -d "$d" ]; do d=$(dirname "$d"); done; df -Pk "$d" | awk 'NR == 2 {print $4 * 1024}'; }
if [ "$need" -gt 0 ]; then
  for d in "$MODELS" "$PREP"; do
    free=$(avail_on "$d")
    [ "$free" -ge "$need" ] || say "WARNING: about $(gb "$need") still to write, $(gb "$free") free at $d"
  done
fi

# ------------------------------------------------------------------ Strata v0.2.9 and its Python environment
strata_ok() { git -C "$STRATA" rev-parse -q --verify HEAD >/dev/null 2>&1; }
if strata_ok && [ -f "$STRATA/.venv/.setup-ok" ]; then
  ok "Strata at $STRATA"
elif [ "$CHECK" = 1 ]; then
  todo "Strata $STRATA_TAG into $STRATA, and its .venv"
else
  if ! strata_ok; then
    [ ! -e "$STRATA" ] || [ -z "$(ls -A "$STRATA")" ] || die "$STRATA exists and is not a git checkout"
    say "cloning Strata $STRATA_TAG into $STRATA"
    git clone -q --depth 1 --branch "$STRATA_TAG" "$STRATA_REPO" "$STRATA"
  fi
  say "Strata's Python environment"
  python3 -m venv "$STRATA/.venv"
  "$STRATA/.venv/bin/pip" install -q -r "$STRATA/requirements.txt"
  touch "$STRATA/.venv/.setup-ok"
fi
if strata_ok; then
  head=$(git -C "$STRATA" rev-parse HEAD)
  [ "$head" = "$STRATA_COMMIT" ] || die "$STRATA is at $head, not Strata $STRATA_TAG ($STRATA_COMMIT)"
fi

# ------------------------------------------------------------------ the model files
verified() {   # name: already checked at this size and time
  local f=$MODELS/$1
  grep -qxF "$1 $(stat -L -c '%s %Y' "$f")" "$MODELS/.verified" 2>/dev/null
}
check_sha() {   # name path: the file's SHA-256 against the table (a few minutes for a shard)
  say "checking $1 (SHA-256)"
  [ "$(sha256sum "$2" | cut -d' ' -f1)" = "${SHA[$1]}" ]
}
for f in "${NEEDED[@]}"; do
  p=$MODELS/$f
  if [ -f "$p" ] && [ "$(stat -L -c %s "$p")" = "${SIZE[$f]}" ] && verified "$f"; then
    ok "$f"
    continue
  fi
  if [ "$CHECK" = 1 ]; then
    if [ -f "$p" ]; then todo "check $p"; else todo "download $f ($(gb "${SIZE[$f]}"))"; fi
    continue
  fi
  mkdir -p "$MODELS"
  if [ ! -f "$p" ]; then
    part=$p.part
    if [ "$(stat -L -c %s "$part" 2>/dev/null || echo 0)" != "${SIZE[$f]}" ]; then
      say "downloading $f ($(gb "${SIZE[$f]}")) from $HF_REPO"
      auth=()
      [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
      curl -L --fail --retry 5 --retry-delay 20 -C - ${auth[@]+"${auth[@]}"} -o "$part" \
        "https://huggingface.co/$HF_REPO/resolve/$HF_REV/$f" || die "download of $f failed (run again to resume)"
    fi
    [ "$(stat -L -c %s "$part")" = "${SIZE[$f]}" ] || die "$part has $(stat -L -c %s "$part") bytes, not ${SIZE[$f]}"
    check_sha "$f" "$part" || die "$part does not match its SHA-256; delete it and run again"
    mv "$part" "$p"
  else
    [ "$(stat -L -c %s "$p")" = "${SIZE[$f]}" ] || die "$p has $(stat -L -c %s "$p") bytes, not ${SIZE[$f]}"
    check_sha "$f" "$p" || die "$p does not match its SHA-256 (delete it to download it again)"
  fi
  echo "$f $(stat -L -c '%s %Y' "$p")" >> "$MODELS/.verified"
  ok "$f"
done

# ------------------------------------------------------------------ the tokenizer folder and the MTP draft layer
run_tool() { (cd "$STRATA" && STRATA_GGUF_PY="$GG/gguf-py" .venv/bin/python "$@"); }
if [ -f "$PREP/tokenizer/vocab.json" ] && [ -f "$PREP/tokenizer/chat_template.jinja" ]; then
  ok "tokenizer at $PREP/tokenizer"
elif [ "$CHECK" = 1 ]; then
  todo "the tokenizer folder $PREP/tokenizer"
else
  say "the tokenizer folder"
  mkdir -p "$PREP"
  run_tool tools/strata_tokenizer.py --gguf "$MODELS/$SHARD1" --out "$PREP"
fi

if [ -f "$PREP/mtp-rt/dense.bin" ] && [ -f "$PREP/mtp-rt/dense.txt" ] && [ -f "$PREP/mtp-rt/experts.bin" ]; then
  ok "MTP draft layer at $PREP/mtp-rt"
elif [ "$CHECK" = 1 ]; then
  todo "the MTP draft layer $PREP/mtp-rt (5.2 GB from Qwen's checkpoint)"
else
  say "the MTP draft layer: its tensors from Qwen's checkpoint (5.2 GB, resumable)"
  mkdir -p "$PREP/mtp-bf16"
  run_tool tools/mtp_fetch.py inventory --out "$PREP/mtp-bf16"
  run_tool tools/mtp_fetch.py fetch --out "$PREP/mtp-bf16"
  run_tool tools/mtp_fetch.py verify --out "$PREP/mtp-bf16"
  if [ ! -f "$PREP/mtp-q2_0.gguf" ]; then
    run_tool tools/mtp_pack.py --src "$PREP/mtp-bf16" --experts q2_0 --out "$PREP/mtp-q2_0.tmp.gguf"
    mv "$PREP/mtp-q2_0.tmp.gguf" "$PREP/mtp-q2_0.gguf"
  fi
  rm -rf "$PREP/mtp-rt.tmp"
  run_tool tools/mtp_rt.py --gguf "$PREP/mtp-q2_0.gguf" --out "$PREP/mtp-rt.tmp"
  mv "$PREP/mtp-rt.tmp" "$PREP/mtp-rt"
  say "the MTP draft layer is in $PREP/mtp-rt; $PREP/mtp-bf16 and $PREP/mtp-q2_0.gguf (6 GB) can be deleted"
fi

# ------------------------------------------------------------------ the picture encoder (as ours: CUDA build, run on the CPU)
VISION=$STRATA/build-vision/bin/strata-vision
if [ "$PICTURES" = 0 ]; then
  ok "no pictures"
elif [ -x "$VISION" ]; then
  ok "picture encoder at $VISION"
elif [ "$CHECK" = 1 ]; then
  todo "build the picture encoder $VISION"
else
  . "$HERE/scripts/toolchain.sh"
  pick_toolchain
  HOSTCC=$(dirname "$HOSTCXX")/$(basename "$HOSTCXX" | sed 's/g++/gcc/')
  [ -x "$HOSTCC" ] || HOSTCC=$(command -v gcc || true)
  [ -n "$HOSTCC" ] || die "no C compiler for the picture encoder (install gcc-13)"
  say "building the picture encoder (nvcc $CUDA_VER, $HOSTCXX)"
  cmake -S "$STRATA/tools/vision" -B "$STRATA/build-vision" -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR="$GG" \
    -DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DCMAKE_CUDA_HOST_COMPILER="$HOSTCXX" -DCMAKE_C_COMPILER="$HOSTCC" -DCMAKE_CXX_COMPILER="$HOSTCXX"
  nice -n 10 cmake --build "$STRATA/build-vision" --target strata-vision -j "${JOBS:-6}"
  [ -x "$VISION" ] || die "the build gave no $VISION"
fi

# ------------------------------------------------------------------ the run config
CFG=$DATA/profile.json
if [ -f "$CFG" ]; then
  ok "run config $CFG (kept; delete it to write it again)"
elif [ "$CHECK" = 1 ]; then
  todo "the run config $CFG"
else
  mkdir -p "$DATA/logs"
  vis=()
  [ "$PICTURES" = 1 ] && vis=(--vision-exe "$VISION")
  python3 "$HERE/profile/fill_profile.py" --models "$MODELS" --mtp "$PREP/mtp-rt" --tokenizer "$PREP/tokenizer" \
    ${vis[@]+"${vis[@]}"} --logs "$DATA/logs" --route-prior "$PRIOR" --gpu "$GPU_LIST" --exe "$EXE" --out "$CFG"
fi

if [ "$CHECK" = 1 ]; then
  say "$TODO step(s) to do"
else
  say "done. Start it with: bash $HERE/run.sh $DATA"
fi
