# Qwen3.8-Flash-Next on two Tesla P100

This is a port of Strata's all-cards split engine (`strata-split-2`) to the NVIDIA Tesla P100 (Pascal, sm_60). It runs
**Qwen3.8-Flash-Next** on **two P100 16 GB** cards with MTP drafting and pictures, from either of two GGUF files. On our
server, with this repository's benchmark (see "Our numbers"):

| Model file and profile | Context | RAM | Fresh writing | Edits (py / md) | Reading 38,862 tokens |
|---|---:|---|---:|---:|---:|
| ISTA-DASLab's GSQ-RCO **IQ3_S** (`setup.sh DATA`) | 524,288 | 96 GB | 66.6 tok/s | 111.0 / 116.0 | 954 tok/s |
| the same in the **64 GB mode** (`setup.sh DATA --ram 64`) | 262,144 | 64 GB | 63.3 | 93.6 / 101.9 | 768 |
| Unsloth's **UD-Q4_K_XL** (`setup.sh DATA --model q4_k_xl`) | 524,288 | 96 GB | 50.3 | 62.6 / 63.5 | 765 |

> **Built for these two files, on Tesla P100 cards.** Files with other expert types (Q6_K, Q3_K, Q2_K, ...) do not use
> this port's P100 kernels and run slower than Strata itself, or not at all (see "Known limitations"). On other cards,
> or with other files, use [Strata](https://github.com/Niko1221/Strata). IQ3_S is the faster of the two: UD-Q4_K_XL is
> the larger quantization (72 GiB of experts against 49), so fewer of its experts fit on the cards, edits wait for
> PCIe, and 96 GB of RAM is tight for it (see "Unsloth's UD-Q4_K_XL").

**Coder** (`setup.sh DATA --coder`, either file): the router picks only from the half of the experts that ISTA-DASLab's
Coder release keeps. On our server fresh writing and code edits ran 9-11 % faster on IQ3_S, and code edits 43 % faster
on UD-Q4_K_XL. But it is a different, smaller model: it is worse outside code, and in our tests it ignored an
instruction given in Slovenian (see "Coder").

The engine is one executable. [Strata](https://github.com/Niko1221/Strata)'s Python server (v0.2.9) drives it and gives
you the OpenAI-compatible API, the tokenizer and the chat template. This repository holds the engine's sources exactly
as our Coder build (`40ef6d53`) was built from them: our production engine for UD-Q4_K_XL plus the Coder switch, which
when off gives our IQ3_S production engine's bits (see "Our numbers"). It also has the build, run configs for both
files, and the benchmark we measure with. The model is not included: you download it yourself (see "Prepare the model
files").

- What is ours and what is Strata's: [NOTICE.md](NOTICE.md), file by file in [docs/PROVENANCE.md](docs/PROVENANCE.md),
  every change in [docs/upstream-v0.2.10.diff](docs/upstream-v0.2.10.diff).
- Engine switches: [docs/SWITCHES.md](docs/SWITCHES.md). How the port came about: [docs/HISTORY.md](docs/HISTORY.md).

## Our numbers

### This release, with the benchmark

[bench/bench.py](bench/bench.py) through Strata v0.2.9's server, on 13 Nov 2026. The IQ3_S columns ran on the engine that
this repository's `build.sh` built on our server (`f85a4aba`), in a test window in the morning; the UD-Q4_K_XL columns in a
second window that day (11:33-11:53), with the UD-Q4_K_XL profiles as they are now in `profile/`, on our production engine
for UD-Q4_K_XL (`c213636c`), which gives this build's bits (below). Each profile was written by `profile/fill_profile.py`
at `FNX_ROUTE_PRIOR=0` (exact, see "Exact and lossy") and run once, on a freshly loaded engine. Writing is at temperature 0
unless marked. These are the benchmark's own inputs, so your run compares with them directly (it prints them beside
yours).

| Test | IQ3_S, 512k | IQ3_S, 64 GB mode, 256k | UD-Q4_K_XL, 512k | UD-Q4_K_XL, 256k |
|---|---:|---:|---:|---:|
| Fresh writing: 12 new prompts, 300 tokens each (tok/s) | **66.6** | **63.3** | **50.3** | **51.1** |
| Edit: the model repeats Strata's `tools/collect_profile_traces.py` with a variable renamed (1,197 tokens) | **111.0** | **93.6** | **62.6** | **63.0** |
| Edit: the model repeats 5,000 characters of Strata's `docs/SECOND_GPU.md` with a word replaced (1,228 tokens) | **116.0** | **101.9** | **63.5** | **63.7** |
| Cold read of 38,862 tokens | **954** | **768** | **765** | 901 (not cold, see below) |
| Writing right after that read | 61.8 | 61.4 | 48.5 | 48.0 |
| New messages on a running conversation: 2,029 / 3,586 / 5,259 / 7,142 / 11,081 / 14,212 / 20,243 tokens | 700 / 815 / 853 / 803 / 941 / 876 / 904 | 613 / 717 / 759 / 710 / 850 / 783 / 817 | 540 / 681 / 721 / 687 / 813 / 732 / 781 | 603 / 755 / 793 / 752 / 887 / 805 / 853 |
| Picture (96 x 96 PNG, encoder on the CPU): time to the answer; writing after it (sampled) | off in this run | off | 4.2 s; 42.6 | 4.1 s; 43.2 |
| Load, until the server reports loaded | 312 s | 249 s | about 210 s | about 210 s |

- **The same bits as our production.** Our IQ3_S production engine (`bf9c7f1f`) with the same profile gave the same
  first-token logits in 27 of 27 requests and the same speeds (66.9, 111.0 / 116.0, 953). The 64 GB mode gave the IQ3_S
  profile's logits in 27 of 27 (for that comparison with the same YaRN setting as the 512k profile; speed does not depend
  on it). In the morning window this build with the UD-Q4_K_XL profile of then and our production engine for it
  (`c213636c`) gave the same logits in 29 of 29 requests, the picture's included. This folder's sources since then
  (`40ef6d53`, `c213636c` plus `FNX_EXPERT_MASK`) with the switch unset gave our IQ3_S production engine's logits in 27 of
  27 requests (KL 0) and its speeds (our test chain: fresh 66.96 against 66.86, a 36k read 943.9 against 942.5).
- **The 64 GB mode** ran in a memory cgroup of 58 GiB on our 96 GB server (see "64 GB of RAM").
- **UD-Q4_K_XL:** 10 GiB of RAM stayed available at the lowest in both runs, and no swap was used. The 256k profile ran
  right after the 512k one, so most of the lookup table was still in the page cache: its "cold" read is warm (the 512k
  profile's 765 is the cold one). The morning's run of the UD-Q4_K_XL profile of then (256k, `FNX_EXPERT_DYN_GIB=14`,
  six guessed experts a row in the next-layer copies) gave fresh 41.4, edits 45.3 / 43.9, a cold read 726, and ran short
  of RAM (4-7 GiB available, the swap file full, the picture's first answer 48.4 s). See "Unsloth's UD-Q4_K_XL".

### Our rounds

The rounds named (`splitNN`, `q4kN`) are in our measurement logs, which are not part of this release (they are in
Slovenian and mixed with our private setup); the numbers and findings in this README and in `docs/` are taken from them.
They ran our own chain of tests (the benchmark's fresh prompts and Markdown edit, other texts for the reads and the
Python edit), IQ3_S at `FNX_ROUTE_PRIOR=0`:

| Test | Result | Measured in |
|---|---|---|
| Fresh writing: 12 new prompts, 300 tokens each | **66.7-67.2** tok/s | rounds `split69`, `split71`, `split72`, `split74` |
| Edit: the model repeats a 5,000-character Markdown text with one word replaced (1,228 tokens) | **114.7-116.2** tok/s | the same rounds |
| Edit: the model repeats a 4.6 KB Python file with a variable renamed (1,388 tokens) | **108.5-109.1** tok/s | the same rounds (private input file; the benchmark uses a public one) |
| Cold read of 36,424 tokens | **961** tok/s | `split74`, `split75` (with `FNX_PF_SP`; 920-924 without) |
| Writing right after that read | 56-60 tok/s | `split69`-`split73` |
| New messages on a running conversation: 1,719 / 3,460 / 4,582 / 7,181 / 10,795 / 13,840 / 20,787 tokens | 656 / 808 / 839 / 807 / 946 / 900 / 917 tok/s | `split75` |
| A read deep in a conversation: 58,555 tokens | 941 tok/s | `split75` |
| Picture (96 x 96 PNG, encoder on the CPU): time to the answer | 2.6-2.7 s | `split69` |
| Writing after the picture (sampled) | 52.4-52.6 tok/s | `split69` |
| Load, until the server reports loaded | 305-335 s | `split69`-`split75` |

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

## 64 GB of RAM

The default profile holds 82-85 GiB once loaded: a pinned RAM copy of every expert (48.8 GiB) and the 26.8 GiB lookup
table (`per_layer_token_embd`, the second shard) copied into RAM. `setup.sh --ram 64` (or `fill_profile.py --ram 64`)
writes a profile for a machine with 64 GB, IQ3_S only:

- `FNX_PLE_RAM=0`: the lookup table is read from its file through the page cache, as Strata's own engine does; each
  decode window or prompt chunk asks for its rows' pages at once.
- `FNX_EXPERT_DYN_GIB=13`: of the experts in VRAM only the coldest 13 GiB keep a RAM copy and take part in the expert
  cache; the other 6.8 GiB stay in VRAM for good, with no copy. RAM then holds 42.0 GiB of experts instead of 48.8.
- `--max-context 262144` (the trained length, no YaRN) and a 2,000 MiB conversation cache. At 524,288 positions the
  K/V pool needs nearly every expert slot to be one it can borrow, so there is nothing to keep in VRAM for good; the load
  says so and stops if the two do not fit together.

We have no 64 GB machine. We ran the profile on our 96 GB server inside a memory cgroup of 58 GiB without swap
(`systemd-run --user --scope -p MemoryMax=58G -p MemorySwapMax=0`), which counts the pinned copies, the engine's own
memory and the page cache, after dropping the lookup table's file from the page cache (as after a reboot):

- **The load** peaked at the cgroup's 58 GiB without an out-of-memory kill: 55.6 GiB that cannot be dropped (42.0 GiB of
  pinned experts and 13.6 GiB the load needs for a while), the rest page cache.
- **Once loaded** the engine holds 45-46.5 GiB (43.4 GiB pinned, 1.5-3 GiB of its own); the page cache took about 3 GB of
  the lookup table by the end of the benchmark.
- **Speed** against the 512k profile on 96 GB (table under "Our numbers"): fresh writing −5 %, edits −16 % and −12 %, a
  cold 38k read −20 %, the reads on a running conversation −10 to −12 %, writing after a long read the same. The answers
  are the same: first-token logits identical in 27 of 27 requests. The lookup table had been dropped from the page
  cache before the run (as after a reboot), so these are figures of a cold start.
- On a real 64 GB machine Linux sees about 62 GiB. Leave little else running: the load's 55.6 GiB must fit beside the
  system, and what is left is page cache for the lookup table.

## Unsloth's UD-Q4_K_XL

[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), folder `UD-Q4_K_XL`, four
parts, 111.3 GB. Its routed experts are Q4_K for gate|up (layer 2 Q5_K) and Q5_1 for down (layers 2, 4, 30, 46 and 47
Q8_0): 71.7 GiB in all, against IQ3_S's 48.8. This release has P100 kernels for those four types, for writing (the row
kernel) and for reading prompts (the job kernels), made like the IQ3_S ones: the values become FP16 exactly, then the
block's scales apply. `setup.sh --model q4_k_xl` downloads it with ISTA-DASLab's picture projector and writes
`profile/flash-next-q4_k_xl-512k.template.json` (524,288 positions with YaRN x2, as the IQ3_S profile;
`profile/flash-next-q4_k_xl-256k.template.json` is the same at 262,144 without YaRN, through `fill_profile.py --template`);
[docs/SWITCHES.md](docs/SWITCHES.md#the-ud-q4_k_xl-profiles) lists what differs from the IQ3_S profile and why.

- **96 GB of RAM** at least. The profiles use the 64 GB mode's switches to fit at all: with every expert pinned (73 GiB),
  9 GiB of page cache were left for the 28.8 GB lookup table and a 36k read ran at 146 tok/s. With the table read
  through the page cache and only the experts the cards may give up kept in RAM (`FNX_EXPERT_DYN_GIB=18`), RAM holds
  70.5 GiB of experts and 10 GiB stayed available at the lowest in the benchmark runs. The first long read after a load
  waits for the disk (765 tok/s at 38,862 tokens); later ones find more of the table in the page cache (about 900).
- **Q4_0 draft experts** (`FNX_MTP_EXPERTS=q4_0`, since 13 Nov 2026): the MTP draft layer's experts made from Qwen's BF16
  tensors with `scripts/mtp_q4.py` instead of Strata's 2-bit ones, in the same VRAM. The model checks every drafted token,
  so the answers stay the same token for token; more drafts are accepted, and fresh writing ran 10 % faster (our round
  `mtp3` at `FNX_ROUTE_PRIOR=0`: 46.99 -> 51.84 tok/s; edits, drafted mostly by prompt lookup, equal). With IQ3_S, whose
  distribution sits further from the BF16 model the draft now follows, the gain was about 1 %, so its profile keeps the
  2-bit file. `setup.sh` makes the file; the benchmark numbers above are from before it.
- **What the RAM copies are for.** Of the 72.5 GiB of experts 52.5 GiB live only in RAM; the cards hold the rest. Those
  keep a RAM copy when the prompt path (its staging buffers) or the K/V pool (at 524,288, up to 7.6 GiB a card) may
  borrow their slots, and when they are dynamic: the cards' expert cache then follows the work. With 5.2 GiB fixed on the
  cards by a general route profile instead (`FNX_EXPERT_DYN_GIB=14`, 4 GiB less RAM, 256k only) edits ran 12-14 % slower.
- **Next-layer copies of 2 guessed experts a row** (`FNX_PREFETCH_WANT=2`, IQ3_S 6): every guessed copy competes with a
  window's own missing experts for PCIe. In our rounds at 512k (`q4k5`, `q4k7`) edits went from 67.4-70.3 tok/s at 6 a
  row to 77.4-84.3 at 2 (9 a row: 54-57; none: 72.2-78.1); fresh writing stayed at 55.8-57.5.
- **Slower than IQ3_S**, most of all in edits. In our rounds (our own test chain, with our production's
  `FNX_ROUTE_PRIOR=0.15`) UD-Q4_K_XL at 512k ran fresh writing at 56.9 tok/s against IQ3_S's 69.3 (82 %), edits at
  77.4-84.3 against 115.0-122.9 (68 %), a cold 36k read at 819 against 946 (87 %). On the cards are 27 % of its experts
  (IQ3_S 41 %), and each missing one is 1.33 times bigger: an edit's prompt-lookup windows of up to 16 rows touch about
  100 different experts a layer and wait for the PCIe copies (an edit's round 151 ms against IQ3_S's 99). Shorter lookup
  drafts (`--lookup` 5, 7, 10), NUMA placement of the RAM copies and half-size staging buffers did not help.
- **Deep contexts** (round `q4k4`, 512k): reads to 134k / 284k / 501k tokens ran at 809 / 765 / 737 tok/s and writing
  there at 49.3 / 48.0 / 38.4 tok/s, close to IQ3_S's (835 / 800 / 725, writing about 53 / 50 / 37).
- **Pictures** work with the same projector; the encoder reads the vocabulary from the first part.
- **The same bits as our production:** see "Our numbers". The kernels compute in FP16 like the IQ3_S ones, so against
  other engines the answers are close, not identical.

## Coder

[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
(Apache-2.0) removes half of the routed experts, 256 of 512 in each layer. RCO chose them for code, agentic tool use and
pictures, and the model card says abilities outside that set degrade. We do not use their file. Its allocation list
names the experts it keeps by their index in the full model ("the mask can be reapplied to the original weights").
`FNX_EXPERT_MASK` applies that list to the full IQ3_S or UD-Q4_K_XL weights: the router never picks a dropped expert,
and the ten picks are weighted among themselves as in the pruned model. So this is the Coder's choice of experts with
your file's weights, not ISTA-DASLab's Coder file.

- **Turn it on:** `bash setup.sh /data/flash-next-coder --models /data/flash-next/models --coder` writes a Coder profile
  into its own folder, sharing the model files. `setup.sh` keeps an existing `profile.json`, hence the new folder. By
  hand: `fill_profile.py --coder`, or add `"FNX_EXPERT_MASK": "<this repository>/engine/data/coder_mask.txt"` to a
  profile's `env`.
- **The list:** `engine/data/coder_mask.txt`, one line a layer. `python3 scripts/coder_mask.py ALLOCATION.txt OUT.txt`
  makes it from their
  `tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.rco-allocation.txt` (revision
  `bba88856f8c3c0c9e40e45f82d05e228b814ac43`), byte for byte.
- **Why it is faster:** each token's ten experts come from 256 a layer instead of 512, so more of them are on the cards
  already. The dropped experts go to RAM first, which leaves the cards' slots to the kept ones. RAM use does not change
  (every expert is still loaded), so it needs 96 GB like the full profile. With the 64 GB mode it has not been tried.

**Speed.** Our own test chain (the benchmark's fresh prompts and Markdown edit, other texts for the reads and the Python
edit, see "Our rounds"), not `bench/bench.py`. It ran at our production's `FNX_ROUTE_PRIOR=0.15`, full and Coder in the
same test window on 13 Nov 2026, each once on a freshly loaded engine, tok/s:

| Test | IQ3_S | IQ3_S, Coder | UD-Q4_K_XL, 512k | UD-Q4_K_XL, 512k, Coder (two runs) |
|---|---:|---:|---:|---:|
| Fresh writing: 12 new prompts, 300 tokens each | 68.8 | **75.2** (+9 %) | 55.3 | **60.9-61.6** (+10-11 %) |
| Edit: a 4.6 KB Python file with a variable renamed (1,388 tokens) | 116.7-118.8 | **130.2-131.2** (+11 %) | 77.5-77.6 | **110.2-112.6** (+43 %) |
| Edit: a 5,000-character Markdown text with a word replaced (1,228 tokens) | 121.5-122.5 | 133.0-133.9, did not edit | 79.4-83.2 | 102.6-108.5, did not edit |
| Cold read of 36,424 tokens | 943 | **979** (+4 %) | 695 | 751-886 |
| New messages on a running conversation, 1,719 to 20,787 tokens | 665-941 | 738-980 | 459-781 | 571-906 |
| Picture, asked in Slovenian: the answer, time to it | "Rdeča in modra", 3.4 s | "red, blue", 2.9 s | not run | "red, blue", 3.2 s |

With `FNX_EXPERT_RAM_GIB=26` instead of 29 (more of the kept experts on the cards) IQ3_S Coder was not faster (fresh
73.6). UD-Q4_K_XL's reads vary between runs because its RAM is tight (see "Unsloth's UD-Q4_K_XL").

**What it does to the answers.** It is lossy by design, far more than `FNX_ROUTE_PRIOR` (see "Exact and lossy"):

- **First token:** against the full IQ3_S model, both at 0.15, the first-token logits of our 29 test requests gave KL
  2.2 on average, median 0.15, and the same first token in 23 of 29. UD-Q4_K_XL Coder against full IQ3_S: the same
  first token in 22-23 of 27-29. For scale, in the same windows: full UD-Q4_K_XL (at 0.15) against full IQ3_S (at 0)
  gave KL 0.041 (26 of 27), and our IQ3_S production engine against this build without the mask, both at 0.15,
  KL 0.0034 (29 of 29; at 0 the two give KL 0).
- **The Python edit** (rename a variable): the same text as the full model's, on both files.
- **The Markdown edit,** asked in Slovenian to replace a word in a Markdown text: the full model replaced every
  occurrence, and the Coder returned the text without a single replacement on both files. So its speed in that row is
  the speed of copying the text.
- **The picture question,** asked in Slovenian: the full model answered in Slovenian, the Coder correctly but in English.
- **Code benchmarks:** we have not run any. ISTA-DASLab's card gives, for their own file, SWE-bench Verified 79.20
  (82.80 at BF16) and LiveCodeBench v6 87.43 (the same as BF16). Those are not numbers for the masked IQ3_S or UD-Q4_K_XL.

Use it for code. For general use, and for instructions in languages other than English, keep the full model. We have
not checked it on English prose.

## Requirements

- **Linux, x86-64.**
- **Two Tesla P100 16 GB (PCIe).** The engine is built for exactly two cards of compute capability 6.0 and refuses
  others. It never uses peer-to-peer access or NVLink: the cards exchange their sums through pinned host memory.
- **PCIe Gen3 x16 for each card.** Experts that are not in VRAM are copied from RAM while it runs (see "Other machines").
- **96 GB of RAM**, or **64 GB** with the 64 GB mode (IQ3_S only, see "64 GB of RAM"). The default profile keeps a
  pinned RAM copy of every expert (about 47 GiB) and the model's 26.8 GiB lookup table (the second shard) in RAM. Once
  loaded it holds 82-85 GiB of the 91.8 GiB Linux sees on our server; during the load it needs 8.4 GiB more, freed at
  the end. UD-Q4_K_XL needs 96 GB too (see "Unsloth's UD-Q4_K_XL").
- **Disk:** 83.6 GB for the IQ3_S shards (UD-Q4_K_XL: 111.3 GB), 0.9 GB for the picture projector, 0.8 GB for the MTP
  draft layer (another 6 GB while you prepare it). The IQ3_S load reads 48 GB from disk: about 100 s of the 5 minutes
  on ours (0.5 GB/s).
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

- `bash setup.sh /data/flash-next --ram 64` writes the 64 GB mode's profile; `bash setup.sh /data/flash-next-q4 --model q4_k_xl`
  downloads and sets up Unsloth's UD-Q4_K_XL instead of the IQ3_S file (112 GB). One `DATA` folder holds one profile, so
  give each model its own folder, or `--models` the same model folder to share the downloads. `--coder` (with either
  file) writes a Coder profile (see "Coder").
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
(set `HOSTCXX=g++-13`). A full build took 17.5 minutes on our two Xeon Bronze 3106 (`kernels_p2.cu` alone most of it).
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

**Or UD-Q4_K_XL** (`setup.sh --model q4_k_xl`), from [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF),
folder `UD-Q4_K_XL`, revision `084c631ac221016019d6e0a863afe92d90edd3b0`, with the same projector from ISTA-DASLab:

| File | Bytes | SHA-256 |
|---|---:|---|
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` | 10,946,618 | `acc44040f74d0e55fb89c3967056cce5dbda1bcc52e909c3ba6fc72b5e9c0858` |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf` | 49,720,539,552 | `45792ae869072d8229384c51c7c73dd959e191505252b41dd71852d9746f5ce4` |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf` | 32,803,030,176 | `8c76e79d444b16a663d9c8db54e585ed3b1da13eebd2619c37b0c355da803c82` |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf` | 28,800,138,432 | `5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36` |

The fourth part is the same lookup table as IQ3_S's second shard: the two files differ only in two header bytes (the
part's number and count). The tokenizer folder made from the first part is byte-identical to the one made from the IQ3_S
file.

`setup.sh` downloads them with `curl` (resumable; set `HF_TOKEN` if Hugging Face asks for one) and checks them against
the tables. By hand, for example with the Hugging Face CLI, one file at a time, then `sha256sum`:

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
# $PREP/mtp-rt holds dense.bin, dense.txt and experts.bin (Strata's 2-bit draft experts)
.venv/bin/python ../scripts/mtp_q4.py "$PREP"/mtp-bf16 "$PREP"/mtp-rt/experts-q4_0.bin
# the draft experts as Q4_0 from the BF16 tensors (1.4 GB), for FNX_MTP_EXPERTS=q4_0 (the UD-Q4_K_XL profiles);
# $PREP/mtp-bf16 and mtp-q2_0.gguf can go afterwards
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
# not cards 0 and 1; --exe if you built elsewhere than build/; --draft-vocab for another draft vocabulary;
# --ram 64: the 64 GB mode; --model q4_k_xl: UD-Q4_K_XL (profile/flash-next-q4_k_xl-512k.template.json; its 256k one
# with --template profile/flash-next-q4_k_xl-256k.template.json); --coder: only the Coder's experts (lossy)
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

The `env` block holds 38 `FNX_*` switches. The ones that matter most for speed: `FNX_EXPERT_RAM_GIB=29.0` (how much of
the experts stays in RAM only), `FNX_P100_DMA=1` with `FNX_DMA_SPLIT=2` and `FNX_PREFETCH=1` (copying missing experts),
`FNX_PF_JOBV=7`, `FNX_PF_STAGE_ALL=1`, `FNX_PF_SP=1` and `FNX_PF_F16ACC=1` (reading), `FNX_P100_ATTN_DEC=1` (attention
while writing). The UD-Q4_K_XL profiles differ in eight of them ([docs/SWITCHES.md](docs/SWITCHES.md#the-ud-q4_k_xl-profiles)).
To stay within the trained 262,144 positions, edit the written JSON: `--max-context 262144`, and remove `FNX_ROPE_SCALE`
and `FNX_YARN_ORIG_CTX`.

## Benchmark and compare

With the server running:

```bash
cd "$P100"
python3 bench/bench.py --strata-dir "$STRATA" --url http://127.0.0.1:8080 --wait --out results.json
python3 bench/bench.py --strata-dir "$STRATA" --tests fresh,edit,long,medium,image --out results.json   # with a picture
```

It runs our tests (fresh writing, two edits, a long read, medium reads, optionally a picture), 10-12 minutes on our
machine, and prints your numbers beside our run of the same benchmark on this release ("Our numbers"): IQ3_S's or
UD-Q4_K_XL's by the model the server reports, the 64 GB mode's with `--ours iq3_s_64gb`. Some prompts are in Slovenian,
as in our runs. The texts are read from the Strata v0.2.9 checkout, so every run reads the same ones. Before you compare:

- **Start the engine fresh and run the benchmark once.** The expert cache learns what it reads, so a second run on the same
  engine is faster.
- Ours are one run each. Our rounds differed by up to 2 % between runs of the same build (long read 903-918), more right
  after a change of settings. Our IQ3_S picture numbers are from our rounds (pictures were off in the release run), and
  UD-Q4_K_XL's from the second run, with the profiles as they are now (`--ours q4_k_xl` is the 512k one, `q4_k_xl_256k` the
  256k one; its long read was not cold).
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
- **`FNX_EXPERT_MASK`** (`--coder`) makes it a different model on purpose, with half of the experts (see "Coder").
  Unset, it changes nothing.
- **Against other engines** the answers are close, not identical: the P100 kernels compute in FP16 where Strata's use
  other arithmetic (`FNX_PF_F16ACC`, the HFMA2 expert kernels). Against our earlier engine on the same cards (Strata
  0.2.9's layer-split engine with our P100 kernels) the first token matched in 21 of 21 requests. The YaRN setting
  changes the position encoding at every length (see "Run" to turn it off).
- The models are quantizations, not the full model: ISTA-DASLab's IQ3_S (Strata's `docs/QUANTS.md` has its numbers) and
  Unsloth's UD-Q4_K_XL. We have not compared their answers' quality.

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
- **More RAM** does not make IQ3_S faster; the cards' PCIe links are the limit. With UD-Q4_K_XL, about 128 GB would keep
  its lookup table in RAM, so the first long read after a load would run like the later ones; writing would stay the same. With **64 GB** use the 64 GB mode (see
  "64 GB of RAM"); less than that does not load.
- **Power limit:** at 250 W instead of our 185 W, long reads were at most 1-1.5 % faster on our cards.
- **P100 SXM2:** the same chip and should run; NVLink is not used. Not tried.
- **Volta and newer cards** (sm_70 and up): the P100 kernels are used only below sm_70, so a build for those cards is
  Strata v0.2.10's split engine with a profile tuned for 16 GB P100s, older than current Strata. Use Strata there.
- **Other Pascal cards** (P40, GTX 10x0, sm_61): not supported. The engine refuses an architecture it was not built for,
  and its kernels are built on FP16 arithmetic, which those chips run at a small fraction of the P100's rate.
- **Three or four cards:** Strata's code has them; this port has only run on two.
- **An older processor** (no AVX2, e.g. Sandy Bridge or Ivy Bridge Xeons): the engine and Strata's server need none, but
  Strata's picture encoder is built for an AVX2 baseline by default; on Sandy Bridge-EP NVIDIA's driver also runs the
  cards at PCIe Gen2 unless told otherwise. `tune/check.py` checks both (next section).

## Tuning for your machine (tune/)

Two tools, described in [tune/README.md](tune/README.md):

```bash
python3 tune/check.py --data /data/flash-next --out report.md   # about a minute, no model loaded
python3 tune/tune.py --data /data/flash-next --screen           # 2-3 hours, one load per candidate
```

- **`check.py`** reads the cards, each card's PCIe path up to the CPU, the processor, RAM, NUMA, the IOMMU and the disk
  with the model; measures host-to-card copies from pinned memory and the disk; picks the profile for your RAM and names
  what limits your machine against ours, with advice. Each piece of advice says whether it rests on a measurement on our
  machine or is only what the design implies. The report is written to paste into a GitHub issue. Run it with the cards
  free; the copy test skips a card that is in use.
- **`tune.py`** measures the switches that give the same bits (`FNX_PREFETCH_WANT`, `FNX_PREFETCH_MIN_T`,
  `FNX_DMA_SPLIT`, `FNX_EXPERT_RAM_GIB`, `FNX_EXPERT_DYN_GIB`, `FNX_PLE_RAM`, `--conversation-cache-mib`; the staging
  buffers on request) on your machine, each candidate on a freshly loaded engine with a shortened benchmark, and keeps a
  change only when it wins by more than the noise it measured between two baseline runs and a final run confirms it. It
  checks the RAM before each load, stops when the machine keeps swapping, compares every arm's first-token logits with
  your starting profile's, and writes `DATA/profile.tuned.json`. `FNX_ROUTE_PRIOR` is tried only with `--allow-lossy`;
  `--coder`'s expert mask is never changed.
- **On our machine** (13 Nov 2026, a test window, the IQ3_S 512k profile at `FNX_ROUTE_PRIOR=0`, engine `c213636c`):
  `check.py` measured 12.3 GB/s to each card, the same from the other CPU's memory and with both cards at once, Gen3 x16
  while copying, and found no bottleneck. `tune.py --screen --only prefetch,dma` took 7 loads (about 70 minutes): the
  two baselines gave fresh 66.7-67.0, edits 111.0 / 115.9-116.1 and a 38,862-token read at 961-962 tok/s (noise 0.5 % or
  less, writing after the long read 2.1 %); 2, 3 or 4 guessed next-layer copies a row were within 1 % of the profile's
  6 (9: −5 %), prefetch off, `FNX_PREFETCH_MIN_T=2` and `FNX_DMA_SPLIT=1` were in the noise, `FNX_DMA_SPLIT=0` cost 4 %
  in edits. It kept the profile as it is, and every arm's first-token logits were the baseline's bit for bit (19 of 19).
  It has not run on another machine yet; results from other machines are welcome as issues.

## Known limitations

- **Only these two files.** The port's P100 expert kernels cover the expert types IQ3_S, IQ3_XXS, IQ2_S, IQ4_XS, IQ4_NL,
  Q4_0, IQ2_XXS and IQ1_M, and since this release Q4_K, Q5_K, Q5_1 and Q8_0 (UD-Q4_K_XL's). Files with other expert
  types (Q6_K, Q3_K, Q2_K, ...) get none of them: writing falls back to Strata's original kernels, and reading a prompt
  stops with `p100 moe: expert type N not handled` (with `FNX_PF_MOE_Q8=1` it reads through the writing kernels,
  slowly). On the first release, which had no Q4_K kernels, a tester ran UD-Q4_K_XL beside current Strata on the same
  machine and the port was slower on every test; on IQ3_S the tester got the same results as ours. Of ISTA's other
  files, IQ2_XS ran on an earlier build of this port (7-8 Nov, about 65 tok/s fresh) but not on this one; Q2_0 and
  IQ3_XXS have not been run. Unsloth's other files have not been run.
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
- **Coder:** worse than the full model outside code, and in our tests it ignored an instruction given in Slovenian and
  answered a Slovenian question in English (see "Coder"). It has not been run with the 64 GB mode or through
  `bench/bench.py`.
- This folder's `build.sh` built the engine on our server (17.5 minutes with `JOBS=12`, llama.cpp from our Strata v0.2.9
  build tree at the pinned commit), and that engine gave our production engines' bits ("Our numbers"); the profile
  writer and the benchmark ran there as they are. The Coder change came after that: its sources (`maska_patch.py`
  applied to the ones `build.sh` built) were built in the build directory `c213636c` came from, not yet with `build.sh`. `setup.sh` and `run.sh` were run against stand-ins (a fake Strata
  checkout and tools, small model files), not yet against the real downloads.

## Credits and license

- **Strata** by Niko1221 and the Strata contributors (MIT): the engine this is built from, its server and tools.
- **sFNX** by John Fink (MIT, https://github.com/johnfink8/sfnx): the split-every-layer engine Strata ported.
- **ggml / llama.cpp** by the ggml authors (MIT): the GPU block formats and dot products; `engine/src/iq3s_x.cuh` is made
  from its `vecdotq.cuh`.
- **ISTA-DASLab** for the GSQ-RCO model files and the Coder release's list of kept experts (Apache-2.0, see
  [third_party/ista-daslab/](third_party/ista-daslab/)), **Qwen** for Qwen3.8-Flash-Next.
- Ideas taken from others: the 16-bit `dp4a` form (`fnx_dp4a_xmad` in `p100_compat.h`) from KaneQuah/Kmic-68's llama.cpp
  fork (commit `99eb48176d`); the copy-engine thread for missing experts (`FNX_P100_DMA`) from DeepInfer; the register
  constraint in `fnx_dp4a_vmad` from Strata PR #4274; the expert pool floor from Strata 0.2.12 (commit `20885f7a`).

MIT, the same license as Strata: see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). The licenses of sFNX and ggml, and of
the Coder's expert list (Apache-2.0), are in [third_party/](third_party/).
