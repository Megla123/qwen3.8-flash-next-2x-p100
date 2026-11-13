# How the port came about

Strata's all-cards split engine (from sFNX) is written for Ampere and newer NVIDIA cards and for AMD RDNA. It does not
build for sm_60: the Tesla P100 has no `dp4a`, no `__nanosleep`, no tensor cores, and its FP16 is fast only through
paired `HFMA2` instructions. Before this engine our two P100 ran Strata 0.2.9's layer-split engine with our own P100
kernels (about 47-55 tok/s fresh, 620-650 tok/s long reads), where one card waits while the other works. This engine gives
each card half of every layer, so both read their weights at once.

The port started on 5 November 2026 from Strata v0.2.10 and went into our production on 6 November. Each step below
was measured in an A/B round (`splitNN`) against the build before and, unless marked lossy, checked to give the same bits
(first-token logits KL 0, identical answers at temperature 0). Dates are 2026, times CET; the hashes are the first
digits of each binary's SHA-256.

| Date | Build | Change | Effect |
|---|---|---|---|
| 5-6 Nov | `f0f40dca` | The port: `p100_compat.h` (dp4a through `vmad`, no `nanosleep`); P100 expert kernels for prompts (`p100_moe.cu`, `p100_iq_gemm.cuh`: FP16 tiles) and for writing (`p100_moe_rows.cuh`: a row kernel); P100 attention for prompts (`p100_attn.cuh`); the lookup table locked in RAM; the arena (K/V pages borrow expert slots, up to 524,544 positions); copies of missing experts beside the row kernels; reads of up to 144 tokens through windows; checkpoints every 16,384 rows; YaRN to 512k | in production from 6 Nov 05:30 |
| 6 Nov | `a7a7f0a8` | `logit_bias` (our chat app sends a list that bans foreign scripts) | |
| 6 Nov | `93b71b1f` | next-layer prefetch (`FNX_PREFETCH`), `FNX_ROUTE_PRIOR` (lossy) | edit py 91.0 → 106.9, md 98.5 → 109.8, fresh 63.4 → 65.1 (with λ 0.15) |
| 6 Nov | `a2b6f771` | GDN recurrence kernel with a warp per part (the kernel 2.0-3.4x faster); rows per warp in the dense int8 products by window size | fresh 65.3 → 70.9 (+8.5 %), edits +4.3 % (λ 0.15) |
| 7 Nov | `a381cc7c`, `333ecfe5` | control vectors (`FNX_CVEC`); IQ2_XXS and IQ1_M kernels for ISTA's IQ2_XS file | IQ2_XS: fresh 65.4, edit py 105.5, md 108.5 |
| 9 Nov | `6f47ed0f` | each copied expert's blocks wait only for its own copy (`FNX_DMA_SPLIT=2`) | edits +2-3 % |
| 10 Nov | `e35054e3` | the prompt expert kernel tuned for small jobs (`FNX_PF_JOBV=3`) | long read 714 → 839 (+17.6 %) |
| 10 Nov | `8be80815` | `hc` reads without needless conversions, attention-selection scores in tiles, small launches folded together | long read 838 → 879, fresh +1.6 % |
| 11 Nov | `c16e3b6d` | attention selection: a lower bound first, then work on candidates only (`FNX_QSA_SEL`) | reads deep in the context (279k → 505k) +14.2 % |
| 11 Nov | `1882468e` | crash fix: the conversation cache picked a conversation or checkpoint by an index that parking the current one had just moved (a `segfault` in `load_state`, 9 times since 9 Nov) | |
| 11 Nov | `b1438fe1` | the prompt path stages every missing expert of a layer instead of reading some across PCIe; next-step loads ahead in the expert kernel; 64-row tiles for `down` (`FNX_PF_JOBV=6`, `FNX_PF_STAGE_ALL`, `FNX_PF_PREP`) | long read 835 → 906 (+8.5 %), medium reads +6-8 % |
| 12 Nov | `2ef67b0e` | the byte dot product in one of three exact forms (`vmad`, `prmt`+`xmad`, the compiler's), chosen per kernel and window size | edits +3.3 %, fresh +0.7 % |
| 12 Nov | `5ef3a6ae` | the prompt expert kernel with two weight buffers (`FNX_PF_JOBV=7`); also the code of two router and `hc` ideas that gave nothing (off) | long read +0.6 %; the first release |
| 12 Nov | `bf9c7f1f` | each card reads `hc` only for its own rows of a prompt pass and the cards trade the mixed rows (`FNX_PF_SP`) | long read 920 → 961 (+4.4 %), deep +4.8 %, medium reads from 36k +3.7-4.5 % |
| 13 Nov | **`c213636c`** | P100 kernels for UD-Q4_K_XL's expert types (Q4_K, Q5_K, Q5_1, Q8_0) and `FNX_SH_IN`; the 64 GB mode (`FNX_PLE_RAM`, `FNX_EXPERT_DYN_GIB`, written in this repository); two RAM placement ideas that gave nothing (`FNX_NUMA_EXPERTS`, `FNX_PLE_WARM`, off) | UD-Q4_K_XL: a 36k read 146 → 722-905 tok/s, fresh writing 52.6; IQ3_S unchanged; **this release** |

## Tried and dropped

Kept in the code where it was cheap, off by default (see [SWITCHES.md](SWITCHES.md)):

- `hc` on 80 blocks for 56 SMs (stuck at the grid barrier) or two blocks per SM (slower); `hc` with f16 dots (fresh −7 %).
- Overlapping copies with compute at every window size (no gain at 2 rows); FP16 weights on the fly when writing (not faster).
- The router with plans per 32 slots, ballot-numbered jobs, 2 or 4 rows per warp, priority streams, another launch order:
  the same bits, no gain in the engine (the router is not on the critical path).
- Dense products in half precision for wide windows (`FNX_DENSE_H`): +0.4 % at most.
- The `xmad` dot product everywhere (fresh −1.9 %: kernels that cross a register step lose occupancy) and in the IQ3_S
  writing kernels (no gain: those are memory-bound).
- Smaller row-kernel blocks, more lanes per row in phase 2, weights past L1, a routing memory for edits: none faster in the
  engine (edits waited for copies; the memory guessed only 27 % right).
- LRU or LFU instead of CLOCK for the expert slots (simulated: 5 % fewer copies), other pool splits, an expert profile
  learned from our own use (slower at the start), refilling the lent slots after a read.
- Huge pages for the lookup table (load 309 s), NUMA placement of RAM (both cards read either CPU's memory at 11.9 GB/s),
  the `performance` governor, C-states off, a higher power limit (+1-1.5 % long reads at 250 W).
- Strata's one-hop sum (`FNX_ONEHOP=2`): +1.4 %, but different last bits.

## The patch scripts

`history/patches/` holds the scripts of the last steps (11-13 Nov) as they were run on our build machine. They edit the
development tree in place and name its paths, so they will not run elsewhere, and their changes are already in `engine/`.
`gen_iq3s_x.sh` there is the original of `scripts/gen_iq3s_x.sh`. The full change against Strata is
[upstream-v0.2.10.diff](upstream-v0.2.10.diff).

## Our prepared files

SHA-256 of the files our production uses, to compare with what Strata's tools give you (README, "Prepare the model files").
Ours were made on 21 Oct 2026 with an older Strata.

| File | SHA-256 |
|---|---|
| `mtp-rt/dense.bin` | `c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a` |
| `mtp-rt/dense.txt` | `991456cf4eb266eb8a888c349e548d11bc640f5dbd9149439aa9786af0f732fd` |
| `mtp-rt/experts.bin` | `09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f` |
| `tokenizer/chat_template.jinja` | `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce` |
| `tokenizer/merges.txt` | `4ba698529766c7a2f5052083ea10cdf2485577f5237a332ee291054eb11e3332` |
| `tokenizer/tokenizer.json` | `c0bd435efbc6c3b28964b47999afe16c4ad268c7c7c4e8cb3af410796a6f30bf` |
| `tokenizer/token_type.json` | `5088c8c298fc06af8382ddb3b76c888703ac2634e82263b97eedcc2ad202738b` |
| `tokenizer/vocab.json` | `4ba64f0332abcfb0b600b7df1537d1e836e68009d5fb7cdb77339738dc6365c4` |
| `engine/data/draft_vocab_sl.bin` (shipped) | `3e3fa1a67d8dbe0989ff682758eb9467ee827174ab3471086b6794b81e50620b` |
