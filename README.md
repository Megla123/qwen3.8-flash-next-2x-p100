# Qwen3.8-Flash-Next on two Tesla P100

This is a port of Strata's all-cards split engine (`strata-split-2`) to the NVIDIA Tesla P100 (Pascal, sm_60). It runs
**Qwen3.8-Flash-Next** in ISTA-DASLab's **GSQ-RCO IQ3_S** GGUF on **two P100 16 GB** cards with a 524,288-token
context, MTP drafting and pictures. On our server it writes **about 67 tokens a second** and reads a long prompt at
**about 920 tokens a second**.

> **Currently built for one model file only: ISTA-DASLab's IQ3_S GGUF, on Tesla P100 cards.** Other quantizations, such
> as Unsloth's UD-Q4_K_XL, do not use this port's P100 kernels and run slower than Strata itself, or not at all (see
> "Known limitations"). On other cards, or with other files, use [Strata](https://github.com/Niko1221/Strata).

The engine is one executable. [Strata](https://github.com/Niko1221/Strata)'s Python server (v0.2.9) drives it and gives
you the OpenAI-compatible API, the tokenizer and the chat template. This repository holds the engine's sources exactly
as they run in our production, the build, an example run config, and the benchmark we measure with. The model is not
included: you download it yourself (see "Prepare the model files").

- What is ours and what is Strata's: [NOTICE.md](NOTICE.md), file by file in [docs/PROVENANCE.md](docs/PROVENANCE.md),
  every change in [docs/upstream-v0.2.10.diff](docs/upstream-v0.2.10.diff).
- Engine switches: [docs/SWITCHES.md](docs/SWITCHES.md). How the port came about: [docs/HISTORY.md](docs/HISTORY.md).

## Our numbers

Measured through Strata's server with the tests of [bench/bench.py](bench/bench.py) on a freshly loaded engine, with the
example profile at `FNX_ROUTE_PRIOR=0` (exact, see below). Writing is at temperature 0 unless marked. The rounds named
(`splitNN`) are in our measurement logs, which are not part of this release (they are in Slovenian and mixed with our
private setup); the numbers and findings in this README and in `docs/` are taken from them.

| Test | Result | Measured in |
|---|---|---|
| Fresh writing: 12 new prompts, 300 tokens each | **66.7-67.1** tok/s | rounds `split69`, `split71`, `split72` |
| Edit: the model repeats a 5,000-character Markdown text with one word replaced (1,228 tokens) | **115.9** tok/s | the same rounds |
| Edit: the model repeats a 4.6 KB Python file with a variable renamed (1,388 tokens) | **108.8-109.1** tok/s | the same rounds (private input file; the benchmark uses a public one) |
| Cold read of 36,424 tokens | **924** tok/s | `split73` (one round; the build before read 903-918 in four rounds) |
| Writing right after that read | 56-60 tok/s | `split69`-`split73` |
| New messages on a running conversation: 1,719 / 3,460 / 4,582 / 7,181 / 10,795 / 13,840 / 20,787 tokens | 659 / 810 / 834 / 805 / 909 / 868 / 883 tok/s | `split73` |
| Picture (96 x 96 PNG, encoder on the CPU): time to the answer | 2.6-2.7 s | `split69` |
| Writing after the picture (sampled) | 52.4-52.6 tok/s | `split69` |
| Load, until the server reports loaded | 305-335 s | `split69`-`split73` |

- **The released binary** (`5ef3a6ae`) ran one round, `split73`: the long read, the writing right after it, and the
  medium reads. Its last change touched the prompt path only, so the other writing rows come from the build before it
  (`2ef67b0e`), whose writing code is the same.
- **Deep contexts**, measured on an older build (`c16e3b6d`, 11 Nov): reading 0 → 134k tokens 835 tok/s,
  131k → 281k 800 tok/s, 279k → 505k 725 tok/s. Later builds only made reading faster.
- **With `FNX_ROUTE_PRIOR=0.15`** (our production setting, lossy) on the build of 11 Nov (`b1438fe1`, `split67`):
  fresh 68.3, edit py 113.1, md 117.5 tok/s, against 66.6, 105.3 and 111.5 at 0 in the same round.
- Two requests at once (`--batch 2`, an older build): about 32 tok/s each, 63 together, so no gain over one.

### The machine

| | |
|---|---|
| Server | HPE ProLiant DL380 Gen10 |
| CPUs | 2 x Intel Xeon Bronze 3106 (8 cores, 1.7 GHz, no Hyper-Threading), 2 NUMA nodes |
| RAM | 96 GB DDR4 RDIMM (6 x 16 GB, 3 channels per CPU); Linux sees 91.8 GiB |
| GPUs | 2 x Tesla P100-PCIE-16GB, one on each CPU, each in a PCIe Gen3 x16 slot; power limit 185 W (default 250 W) |
| Host to card | 11.9 GB/s per card, the same from either CPU's memory |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-38, kernel parameter `iommu=pt`, transparent huge pages `madvise` |
| Driver, toolkit | NVIDIA 580.178.04; CUDA 12.4.131 (Ubuntu's package), host compiler g++ 13, CMake 4.2.3 |
| Server side | Strata v0.2.9's `serve/` on Python 3.14.4 |

Our server runs Strata v0.2.9 with two local changes that are not on the speed path: a `/slots` endpoint for our own app,
and tool-call parsing fixes backported from Strata 0.2.12. Plain v0.2.9 is what you need.

## Requirements

- **Linux, x86-64.**
- **Two Tesla P100 16 GB (PCIe).** The engine is built for exactly two cards of compute capability 6.0 and refuses
  others. It never uses peer-to-peer access or NVLink: the cards exchange their sums through pinned host memory.
- **PCIe Gen3 x16 for each card.** Experts that are not in VRAM are copied from RAM while it runs (see "Other machines").
- **96 GB of RAM.** The engine keeps a pinned RAM copy of every expert (about 47 GiB) and the model's 26.8 GiB lookup
  table (the second shard) in RAM. Once loaded it holds 82-85 GiB of the 91.8 GiB Linux sees on our server; during the
  load it needs 8.4 GiB more, freed at the end. **64 GB is not enough** for this engine, even though Strata's own engine
  runs IQ3_S in 64 GB.
- **Disk:** 83.6 GB for the model's two shards, 0.9 GB for the picture projector, 0.8 GB for the MTP draft layer (another
  6 GB while you prepare it). The load reads 48 GB from disk: about 100 s of the 5 minutes on ours (0.5 GB/s).
- **CUDA 12.x** (CUDA 13 dropped sm_60) with a host compiler it accepts (CUDA 12.4: gcc 13 at most), CMake 3.22 or newer,
  git, and a driver that supports both CUDA 12 and Pascal.
- **Python 3.10 or newer** for Strata's server and tools.
- **`iommu=pt`** (or the IOMMU off) on Intel boards with VT-d: with translation on, our host-to-card copies ran at
  7.6-8 GB/s instead of about 12.

## Install

Three commands, after the requirements above. `/data/flash-next` is any folder with about 95 GB free; everything but the
engine goes there.

```bash
git clone https://github.com/Megla123/qwen3.8-flash-next-2x-p100.git && cd qwen3.8-flash-next-2x-p100
bash build.sh                          # the engine: build/split/strata-split-2, about 20 minutes
bash setup.sh /data/flash-next         # Strata v0.2.9, the model (85 GB download), tokenizer, MTP layer, pictures, run config
bash run.sh /data/flash-next           # the server on http://127.0.0.1:8080/v1, loaded after about 5 minutes
```

- `bash setup.sh /data/flash-next --check` only reports what is there and what is still to do.
- `setup.sh` skips every step whose result is already there, so after an interruption you run it again; downloads
  resume. `--models DIR` uses model files you already have (they are checked against SHA-256), `--no-pictures` leaves
  out the picture encoder and its 0.9 GB projector, `--route-prior 0.15` writes our production setting (lossy, see
  "Exact and lossy"). It finds the two P100 with `nvidia-smi` and names them in the run config.
- `bash run.sh /data/flash-next --port 8080 --host 127.0.0.1` checks the engine's libraries, free RAM and the cards
  before it starts the server. Stop it with Ctrl+C and check with `nvidia-smi` that the cards are free.
- With the server running, `python3 bench/bench.py --strata-dir /data/flash-next/strata-v0.2.9 --wait` measures it
  (see "Benchmark and compare").

The sections below say what each script does and how to do the same by hand. They use these names:

```bash
P100=$HOME/qwen3.8-flash-next-2x-p100  # this repository
DATA=/data/flash-next                  # the folder setup.sh fills
STRATA=$DATA/strata-v0.2.9             # Strata's server and tools, at v0.2.9
MODELS=$DATA/models                    # the downloaded model files
PREP=$DATA/prep                        # the files made from them (tokenizer, MTP draft layer)
```

## Build

```bash
cd "$P100" && bash build.sh            # nvcc from PATH; or: CUDA=/usr/local/cuda-12.4 JOBS=8 bash build.sh
```

`build.sh`:

1. fetches llama.cpp at `3cf03257f219afbe7334045ff7c6a06ac68c627d`, the commit Strata v0.2.9 and v0.2.10 both pin, into
   `third_party/llama.cpp` (only its ggml-cuda headers are used; ggml itself is not built);
2. regenerates `engine/src/iq3s_x.cuh` from that checkout and stops if it differs from the shipped one;
3. configures and builds `build/split/strata-split-2`, with production's settings: Release, `sm_60`, 2 cards,
   `-DSTRATA_SPLIT_PF_JOB=64`, `--use_fast_math --expt-relaxed-constexpr`, and `engine/p100_compat.h` force-included in
   every CUDA source;
4. refuses a binary that can call a CUDA peer-access function, and warns if `ldd` cannot find the CUDA libraries.

It stops early, with a message, if the toolkit is not CUDA 12 or the host compiler is newer than that toolkit accepts
(set `HOSTCXX=g++-13`). A full build took about 20 minutes on our two Xeon Bronze 3106 (`kernels_p2.cu` alone about 17).
Each compile job takes 1-2 GB of RAM. Do not build on the machine while the engine is loaded: it pins most of the RAM,
and a build beside it once hung our server.

The rebuilt binary will not be byte-identical to ours, because the source path is compiled in (the default location of
`engine/data/`). Its code is the same: [docs/PROVENANCE.md](docs/PROVENANCE.md) lists the SHA-256 of every source file of
our binary, and the copies here match them.

## Prepare the model files (what setup.sh does)

Strata v0.2.9 is the server, and its tools prepare the files. `setup.sh` clones it into `$STRATA`; by hand:

```bash
git clone https://github.com/Niko1221/Strata.git "$STRATA"
cd "$STRATA" && git checkout v0.2.9          # commit c62d5e3a15b40ea51c96aae9e8d36928965f53ab
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export STRATA_GGUF_PY=$P100/third_party/llama.cpp/gguf-py     # llama.cpp's gguf-py, fetched by build.sh
```

The commands in this section are Strata's tools as their own help describes them in v0.2.9. Our files were made in
October with an older Strata; we have not run these commands from v0.2.9 ourselves. Run them from `$STRATA`.

**The model.** From [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (the one Strata pins), into `$MODELS`:

| File | Bytes | SHA-256 |
|---|---:|---|
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | 54,817,524,224 | `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` | 28,800,138,432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (pictures only) | 907,543,008 | `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0` |

`setup.sh` downloads them with `curl` (resumable; set `HF_TOKEN` if Hugging Face asks for one) and checks them against
the table. By hand, for example with the Hugging Face CLI, one file at a time, then `sha256sum`:

```bash
hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
    --revision ed59f92082b1e93c0e96d60a8b11aab089b52f09 --local-dir "$MODELS"
```

**The tokenizer folder** (vocabulary, merges and chat template, taken from the GGUF's metadata):

```bash
.venv/bin/python tools/strata_tokenizer.py --gguf "$MODELS"/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --out "$PREP"
# writes $PREP/tokenizer/
```

**The MTP draft layer.** The GGUF has no MTP head. Strata's tools read its 31 tensors (5.2 GB) from Qwen's BF16
checkpoint with HTTP range requests (at a pinned revision, checked against SHA-256), pack them and write the engine's
runtime files:

```bash
.venv/bin/python tools/mtp_fetch.py inventory --out "$PREP"/mtp-bf16
.venv/bin/python tools/mtp_fetch.py fetch --out "$PREP"/mtp-bf16
.venv/bin/python tools/mtp_fetch.py verify --out "$PREP"/mtp-bf16
.venv/bin/python tools/mtp_pack.py --src "$PREP"/mtp-bf16 --experts q2_0 --out "$PREP"/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf "$PREP"/mtp-q2_0.gguf --out "$PREP"/mtp-rt
# $PREP/mtp-rt holds dense.bin, dense.txt and experts.bin; $PREP/mtp-bf16 and mtp-q2_0.gguf can go afterwards
```

The SHA-256 of our MTP and tokenizer files are in [docs/HISTORY.md](docs/HISTORY.md#our-prepared-files), to compare.

**The draft vocabulary.** The draft layer may only propose tokens from a subset. `engine/data/draft_vocab_sl.bin` is
the one we run and measured with: sFNX's English and code subset (`draft_vocab_en.bin`, 40,525 tokens) plus 8,621
Slovenian word pieces, 49,146 in all. For English only, `fill_profile.py --draft-vocab "$P100"/engine/data/draft_vocab_en.bin`
takes sFNX's subset; for another language build one with Strata's `tools/draft_vocab.py --corpus` (see its help). The
choice changes speed, never the answers: every drafted token is checked by the full model.

**The picture encoder** (optional). Strata's `strata-vision`, built from the v0.2.9 checkout against the same llama.cpp:

```bash
CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13 cmake -S tools/vision -B build-vision -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_DIR="$P100"/third_party/llama.cpp -DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-13
cmake --build build-vision --target strata-vision -j 6        # $STRATA/build-vision/bin/strata-vision
```

These are the settings our encoder was built with. The profile runs it on the CPU (`"gpu": false`), so the cards' memory
stays with the model.

## Run (what run.sh does)

`setup.sh` writes the run config to `$DATA/profile.json` with `profile/fill_profile.py` and keeps it on later runs, so
you can edit it. By hand, the script checks that every file it names exists:

```bash
cd "$P100"
python3 profile/fill_profile.py --models "$MODELS" --mtp "$PREP"/mtp-rt --tokenizer "$PREP"/tokenizer \
    --vision-exe "$STRATA"/build-vision/bin/strata-vision --logs "$DATA"/logs --out "$DATA"/profile.json
# without --vision-exe: no pictures; --route-prior 0.15: our production setting (lossy); --gpu 2,3 if the P100 are
# not cards 0 and 1; --exe if you built elsewhere than build/; --draft-vocab for another draft vocabulary
```

`run.sh` then starts Strata's server with it from the v0.2.9 checkout, which by hand is:

```bash
cd "$STRATA"
.venv/bin/python serve/server.py --engine strata --config "$DATA"/profile.json --port 8080 --host 127.0.0.1
```

`LD_LIBRARY_PATH` must lead to the CUDA 12 libraries the engine was built against (ours is unset: Ubuntu's package puts
them in `/usr/lib/x86_64-linux-gnu`). The engine logs to the config's `"log"`. `GET /health` answers `"loaded": true`
after about 5 minutes; then the API is at `http://127.0.0.1:8080/v1`.

**What the profile sets** (every switch in [docs/SWITCHES.md](docs/SWITCHES.md)):

| Key or argument | Value | Meaning |
|---|---|---|
| `"gpu"` | `"0,1"` | the two P100 as `nvidia-smi` numbers them (`setup.sh` finds them) |
| `--native` | shard 1 | the model; shard 2 is found beside it |
| `--mtp`, `--draft-vocab` | the MTP folder, `draft_vocab_sl.bin` | the draft layer and the tokens it may propose |
| `--max-context` | 524288 | with `FNX_ROPE_SCALE=2 FNX_YARN_ORIG_CTX=262144` (YaRN x2); longer requests get HTTP 400 |
| `--spec`, `--lookup` | 4, 15 | MTP draft depth; prompt-lookup drafts of up to 15 tokens (edits) |
| `--conversation-cache-mib` | 4000 | parked conversations in RAM, so returning to one does not read it again |
| `--batch` | 1 | one request at a time |
| `--vision` | | pictures through the encoder |

The `env` block holds 37 `FNX_*` switches. The ones that matter most for speed: `FNX_EXPERT_RAM_GIB=29.0` (how much of
the experts stays in RAM only), `FNX_P100_DMA=1` with `FNX_DMA_SPLIT=2` and `FNX_PREFETCH=1` (copying missing experts),
`FNX_PF_JOBV=7`, `FNX_PF_STAGE_ALL=1` and `FNX_PF_F16ACC=1` (reading), `FNX_P100_ATTN_DEC=1` (attention while writing).
To stay within the trained 262,144 positions, edit the written JSON: `--max-context 262144`, and remove `FNX_ROPE_SCALE`
and `FNX_YARN_ORIG_CTX`.

## Benchmark and compare

With the server running:

```bash
cd "$P100"
python3 bench/bench.py --strata-dir "$STRATA" --url http://127.0.0.1:8080 --wait --out results.json
python3 bench/bench.py --strata-dir "$STRATA" --tests fresh,edit,long,medium,image --out results.json   # with a picture
```

It runs our tests (fresh writing, two edits, a long read, medium reads, optionally a picture), under 10 minutes on our
machine, and prints your numbers beside ours. Some prompts are in Slovenian, as in our runs. The texts are read from the
Strata v0.2.9 checkout, so every run reads the same ones. Before you compare:

- **Start the engine fresh and run the benchmark once.** The expert cache learns what it reads, so a second run on the same
  engine is faster.
- Comparable as they are: fresh writing (the same 12 prompts), the Markdown edit (the same text: v0.2.9's
  `docs/SECOND_GPU.md` is the file we used) and the picture.
- Not the same text as ours: the long and medium reads (we read an older Strata checkout with a local change; the
  benchmark reads v0.2.9's `docs/*.md` and `src/**/*.cpp`, the same sizes in characters) and the Python edit (ours was a
  private 4.6 KB script; the benchmark uses Strata's `tools/collect_profile_traces.py`, 4.1 KB). Expect the same ballpark,
  not the same numbers.
- Our runs differed by up to 2 % between rounds on the same build (long read 903-918), more right after a change of
  settings.
- Keep `FNX_PROF=0` and nothing else on the cards. Our cards ran at 1,328 MHz under load; the clock follows recent load,
  so short tests can differ by 4-5 %.

## Exact and lossy

Every change in this port was checked against the build before it: at temperature 0 the first-token logits gave KL 0
(we dump them with `FNX_DUMP_FIRST_LOGITS`) and the answers were the same to the character. "Exact" in this repository
means that: the same bits as our previous build.

- **`FNX_ROUTE_PRIOR`** is the one lossy switch. Above 0, past each token's top 2 experts the router prefers experts that
  are already in VRAM. At 0.15 the first-token KL was 0.0095 and a wider check (8 edits, 10 programs, 10 calculations,
  5 JSON tasks, 3 hidden-data tasks, 6 Slovenian answers) came out the same as at 0. The same prompt at temperature 0 no
  longer always gives the same text, because the routing depends on what the cache holds. At 0.3 a long Markdown edit
  made only 2 of 6 replacements, so we stopped at 0.15. The template defaults to 0.
- **Against other engines** the answers are close, not identical: the P100 kernels compute in FP16 where Strata's use
  other arithmetic (`FNX_PF_F16ACC`, the HFMA2 expert kernels). Against our earlier engine on the same cards (Strata
  0.2.9's layer-split engine with our P100 kernels) the first token matched in 21 of 21 requests. The YaRN setting
  changes the position encoding at every length (see "Run" to turn it off).
- The model is IQ3_S: ISTA-DASLab's quantization, not the full model (Strata's `docs/QUANTS.md` has its numbers).

## Other machines

We have one machine. The points below are what the design implies, with the measurements on our machine they rest on;
none of them was measured on another machine.

- **One CPU instead of two:** should work the same. The engine needs no peer access; it pins each card's host thread to
  the CPU the card hangs from (from `/sys`), which on one CPU is simply that CPU. On our server each card reads both
  CPUs' memory at the same 11.9 GB/s.
- **x8/x8 slots (or a x16 slot wired x8), PCIe Gen3:** about 6 GB/s per card instead of 12. About 60 % of the experts
  live only in RAM, so traffic over PCIe is part of the work: missing experts are copied in for each decode window, and a
  long read stages every missing expert of each layer (up to 568 MiB a layer) for each chunk. Expect reading to slow
  down clearly and writing to slow down less (fewer copies per token). Gen2 x16 is like Gen3 x8.
- **More RAM** does not make it faster; the cards' PCIe links are the limit. **Less** than 96 GB does not load this
  profile.
- **Power limit:** at 250 W instead of our 185 W, long reads were at most 1-1.5 % faster on our cards.
- **P100 SXM2:** the same chip and should run; NVLink is not used. Not tried.
- **Volta and newer cards** (sm_70 and up): the P100 kernels are used only below sm_70, so a build for those cards is
  Strata v0.2.10's split engine with a profile tuned for 16 GB P100s, older than current Strata. Use Strata there.
- **Other Pascal cards** (P40, GTX 10x0, sm_61): not supported. The engine refuses an architecture it was not built for,
  and its kernels are built on FP16 arithmetic, which those chips run at a small fraction of the P100's rate.
- **Three or four cards:** Strata's code has them; this port has only run on two.

## Known limitations

- **Only the IQ3_S file.** The port's P100 expert kernels cover the expert types IQ3_S, IQ3_XXS, IQ2_S, IQ4_XS, IQ4_NL,
  Q4_0, IQ2_XXS and IQ1_M. Files whose experts are K-quants, such as Unsloth's UD-Q4_K_XL (Q4_K, Q5_K, Q6_K, Q8_0), get
  none of them: writing falls back to Strata's original kernels, and reading a prompt stops with
  `p100 moe: expert type N not handled` (with `FNX_PF_MOE_Q8=1` it reads through the writing kernels, slowly). A tester
  ran UD-Q4_K_XL on this port beside current Strata on the same machine, and the port was slower on every test; on IQ3_S
  the tester got the same results as ours. These files also hold more expert bytes (UD-Q4_K_XL about 72 GiB against 47), so
  less of them fits on the cards. Of ISTA's other files, IQ2_XS ran on an earlier build of this port (7-8 Nov, about
  65 tok/s fresh) but not on this one; Q2_0 and IQ3_XXS have not been run.
- Two P100 16 GB and Linux only.
- One request at a time is what we tuned for (`--batch 1`).
- The engine refuses `logprobs`.
- Loading takes about 5 minutes; the first answers after a load are slower while the expert cache settles.
- With a long context and the 4 GB conversation cache, RAM is tight: our server has 6-9 GiB free once loaded. A parked
  conversation takes about 21 KB a token; bigger ones are read again instead of restored.
- The sources' comments refer to our test tools (`mikro/...`), measurement rounds (`splitNN`), ideas by number and our own
  apps; those are not part of this release. A few `FNX_PROF` log lines are in Slovenian. We kept the files exactly as
  they run.
- `history/patches/` holds the last patch scripts we applied, with our build machine's paths. They are a record: their
  changes are already in `engine/`.
- This folder's `build.sh` and top-level `CMakeLists.txt` reproduce our build's settings and were checked against it by
  reading (same sources, same flags), but this exact folder has not been compiled yet. `setup.sh` and `run.sh` were run
  against stand-ins (a fake Strata checkout and tools, small model files), not yet against the real downloads; the
  header generator, the profile writer and the benchmark (against a stand-in server) were run as they are.

## Credits and license

- **Strata** by Niko1221 and the Strata contributors (MIT): the engine this is built from, its server and tools.
- **sFNX** by John Fink (MIT, https://github.com/johnfink8/sfnx): the split-every-layer engine Strata ported.
- **ggml / llama.cpp** by the ggml authors (MIT): the GPU block formats and dot products; `engine/src/iq3s_x.cuh` is made
  from its `vecdotq.cuh`.
- **ISTA-DASLab** for the GSQ-RCO model files, **Qwen** for Qwen3.8-Flash-Next.
- Ideas taken from others: the 16-bit `dp4a` form (`fnx_dp4a_xmad` in `p100_compat.h`) from KaneQuah/Kmic-68's llama.cpp
  fork (commit `99eb48176d`); the copy-engine thread for missing experts (`FNX_P100_DMA`) from DeepInfer; the register
  constraint in `fnx_dp4a_vmad` from Strata PR #4274; the expert pool floor from Strata 0.2.12 (commit `20885f7a`).

MIT, the same license as Strata: see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). The licenses of sFNX and ggml are in
[third_party/](third_party/).
