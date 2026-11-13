# Engine switches (`FNX_*` environment variables)

The engine takes its settings from the environment. Put them in the run config's `"env"` block and Strata's server
passes them to the engine process. Command-line arguments (`--max-context`, `--spec`, ...) are listed in the README.

"Strata" means the switch exists in Strata v0.2.10's split engine. "Ours" means this port added it. Unless a row
says otherwise, the speed effects were measured on earlier builds of this engine, in the rounds named in brackets
(`splitNN`), at `FNX_ROUTE_PRIOR=0`. The full logs are not part of this release.

**Output effect:**
- *none*: placement, scheduling or a kernel variant that gives the same bits (checked: first-token logits KL 0 and
  identical answers at temperature 0 against the build before).
- *rounding*: other arithmetic, so the last bits differ (answers were checked to stay the same).
- *lossy*: the computation itself changes.

## The switches in the example profile

| Switch | Profile | Unset | Origin | What it does | Output effect | Measured |
|---|---|---|---|---|---|---|
| `FNX_ARENA` | 1 | 1 | Strata | K/V pages borrow expert slots as the context grows, so 524,288 positions fit | none | context up to 524,544 positions in 16 GB cards |
| `FNX_EXPERT_RAM_GIB` | 29.0 | 3.0 | Strata | the coldest N GiB of experts stay in RAM only; the rest sizes the VRAM expert slots (`off` with `FNX_ARENA=0`: all experts in VRAM) | none | about 18 of 47 GiB of experts fit the slots |
| `FNX_EXPERT_PROFILE` | `engine/data/expert-routes.bin` | the same file under the source dir | Strata | per-expert use counts that rank which experts are coldest | none | (sFNX's file; a profile learned from our own use was slower: fresh 26.2 vs 31.0 at start) |
| `FNX_PF_MOE_Q8` | 0 | 0 | Strata | 1: prompt-path experts through the decode kernels' q8_1 arithmetic | rounding | 0 takes the P100 prompt expert kernels |
| `FNX_PF_ATTN_DEC` | 0 | 0 | Strata | 1: prompt-path attention through the decode kernels | rounding | 0 takes the P100 prompt attention (`att` 6,338 → 1,520 ms a pass) |
| `FNX_PF_GDN_CHUNK` | 0 | 1 | Strata | the prompt path's delta rule in sub-chunks (1) or whole (0) | rounding | 0 is what all our rounds ran |
| `FNX_PROF` | 0 | 0 | ours | timing stamps and VRAM reports in the log (costs about 2 % of a window; some lines are in Slovenian) | none | keep 0 when measuring |
| `FNX_PF_LEND` | 1 | 0 | ours | the prompt path's buffers live in the last expert slots while reading (2,846 MiB a card), so they need no VRAM of their own | none | with the live draft cost and other changes of `split18`: fresh 48.1 → 55.2, read 539 → 595 tok/s |
| `FNX_ROUND_MS`, `FNX_DRAFT_ROW_MS`, `FNX_DRAFT_CALL_MS` | 27, 10, 1.5 | 11.2, 1.3, 0.65 | ours | starting costs of a decode round, a drafted row and a draft call (ms) for the draft-length planner; measured costs replace them as it runs (`FNX_DRAFT_COST_LIVE`, default 1) | none | |
| `FNX_DRAFT_MIN_P` | 0.7 | 0 | ours | an MTP draft chain stops below this draft probability | none at temperature 0 (every token is checked by the full model) | |
| `FNX_COPY_BLOCKS`, `FNX_COPY_THREADS` | 16, 256 | 40, 256 | ours | grid of the kernel that copies missing experts beside the row kernels | none | |
| `FNX_PF_F16ACC` | 1 | 0 | Strata | the prompt path's dense GEMMs accumulate in f16 (cuBLAS `CUBLAS_COMPUTE_16F`) | rounding | read 595 → 750 tok/s, same answers |
| `FNX_READ_WINDOWS` | 144 | 0 | ours | prompts of up to N new tokens are read through decode windows, with the expert slots left as they are | none | a 58-token follow-up 1,150 → 857 ms |
| `FNX_ROPE_SCALE`, `FNX_YARN_ORIG_CTX` | 2, 262144 | off, from the file | Strata | YaRN x2 over the trained 262,144 positions, for `--max-context 524288` | changes the position encoding at every length | needed above 262,144 |
| `FNX_PF_STAGE_ROWS` | 256 | 2048 | Strata | prompts from this many tokens on stage the experts' copies | none | 1,764 tokens to first token 5,469 → 3,266 ms |
| `FNX_P100_ATTN_DEC` | 1 | 0 | ours | decode windows' attention through the P100 prompt attention kernels | rounding | edit py 86.1 → 89.2, md 93.3 → 97.8; first token the same in 16 of 16 |
| `FNX_CKPT_EVERY`, `FNX_CKPTS` | 16384, 12 | 0, 6 | ours | a state checkpoint every N rows of a long read, at most M a conversation, so an edited conversation resumes from the nearest one | none | changed 36k conversation 29.7 → 21.6 s |
| `FNX_P100_DMA` | 1 | 0 | ours | a host thread per card copies missing experts with the copy engine (`cudaMemcpyAsync`) instead of a copy kernel | none | fresh 61.05 → 62.49 |
| `FNX_PREFETCH`, `FNX_PREFETCH_WANT`, `FNX_PREFETCH_MIN_T` | 1, 6, 4 | 0, 10, 1 | ours | guess the next layer's experts and start their copies early: 6 guesses a row, windows of 4 rows or more | none | edit py 91.0 → 97.5, md 98.5 → 102.9 (`split34`, `split35`) |
| `FNX_ROUTE_PRIOR` | 0 (our production: 0.15) | 0 | ours | lambda: past each row's top 2, experts already in VRAM are preferred | **lossy** | 0.15: fresh 66.64 → 68.27, edit py 105.3 → 113.05, md 111.45 → 117.45 (`split67`, older binary); first-token KL 0.0095 (`split35`) |
| `FNX_DMA_SPLIT` | 2 | 0 | ours | 1: the copier sends all gate\|up parts first; 2: each copied expert's blocks wait for their own part only | none | edits +2-3 % (`split49`, `split51`) |
| `FNX_PF_JOBV` | 7 | 0 | ours | the P100 prompt expert kernel's variant (0-7; 7: down in 64-row tiles, gate\|up with two weight buffers; IQ2_S as 4) | none | mode 3: long read 714 → 839 (+17.6 %); 6 with the four `STAGE`/`PREP` switches: 835 → 906; 7: +0.6 % |
| `FNX_PF_CONV` | 3 | 0 | ours | the prompt path's `hc` read without needless f32/f16 conversions (bits 1, 2) | none | with the next two: long read 838 → 879, fresh +1.6 % (`split59`) |
| `FNX_QSA_TILE` | 1 | 0 | ours | attention-selection scores computed in tiles | none | (above) |
| `FNX_FOLD` | 7 | 0 | ours | small window launches folded into their neighbours (bits 1, 2, 4) | none | (above) |
| `FNX_QSA_SEL` | 1 | 0 | ours | attention selection: a lower bound from the top 1,024 first, then work on candidates only | none | reads at 279k → 505k tokens +14.2 %; below 60k in the noise |
| `FNX_PF_PREP` | 1 | 0 | ours | `k_hcb_prep` loads a row's four streams before the first block sum | none | part of the 835 → 906 step |
| `FNX_PF_STAGE_ALL`, `FNX_PF_STAGE_MIB`, `FNX_PF_TAIL_MIB` | 1, 1024, 3000 | 0, 384, 2450 | ours (`STAGE_MIB`: Strata) | while reading, the staging buffer holds every missing expert of the largest layer (568 MiB), so no expert is read across PCIe in place; `TAIL` must cover two staging buffers and the lent buffers | none | part of the 835 → 906 step |
| `FNX_PF_SP` | 1 | 0 | ours | in a prompt pass of two chunks or more each card reads `hc` only for the rows of its own part of the ring and the cards trade the mixed rows through host memory; each layer's sums stop after their reduce-scatter (from 3,584 rows in the second chunk, `FNX_PF_SP_MIN`) | none | long read +4.4 %, deep reads +4.8 %, medium reads from 36k on +3.7-4.5 % (`split74`, `split75`) |

**Trap:** `FNX_PF_WINDOWS` is a name the engine reserves for another model (27B) and stops on. The read-through-windows
switch is `FNX_READ_WINDOWS`.

## The 64 GB mode

Off in the IQ3_S profile; `fill_profile.py --ram 64` (or `setup.sh --ram 64`) sets them, and the UD-Q4_K_XL profile uses the
first two. They come from our port of this engine to two Tesla M40 with 30 GiB of RAM. README, "64 GB of RAM", has what
they cost.

| Switch | 64 GB profile | Unset | Origin | What it does | Output effect |
|---|---|---|---|---|---|
| `FNX_PLE_RAM` | 0 | 1 | ours | 0: the 26.8 GiB lookup table (`per_layer_token_embd`, the second shard) is not copied into RAM but read from the file through the page cache, as Strata's own engine does (random access, and each decode window or prompt chunk asks for its rows' pages at once with `MADV_WILLNEED`) | none |
| `FNX_EXPERT_DYN_GIB` | 13 | unset | ours | of the experts in VRAM only the coldest N GiB keep a RAM copy and take part in the expert cache; the rest stay in VRAM for good with no RAM copy, so RAM holds `FNX_EXPERT_RAM_GIB` + N GiB of experts instead of all of them (IQ3_S: 48.8 GiB). The K/V pool (`FNX_ARENA`) and the prompt path (`FNX_PF_LEND`) borrow only those N GiB's slots: the load stops with a message when they cannot hold `--max-context` or the prompt buffers | none |
| `FNX_EXPERT_DYN_MIN` | unset | 384 | ours | with `FNX_EXPERT_DYN_GIB`: at least this many experts of each expert size stay in the cache (a window's picks) | none |

## The UD-Q4_K_XL profiles

`flash-next-q4_k_xl-512k.template.json` (`fill_profile.py --model q4_k_xl`, `setup.sh --model q4_k_xl`) and
`flash-next-q4_k_xl-256k.template.json` (`fill_profile.py --model q4_k_xl --template profile/flash-next-q4_k_xl-256k.template.json`)
are the IQ3_S template with the changes below; the two differ only in `--max-context` and YaRN. README, "Unsloth's
UD-Q4_K_XL", has what they gave.

| Switch or argument | Q4 profiles | IQ3_S profile | Why |
|---|---|---|---|
| `--max-context` | 524288 with YaRN x2 (512k), or 262144 without (256k) | 524288, YaRN x2 | at 524,288 the K/V pool may borrow up to 7.6 GiB of expert slots a card; those experts need their RAM copies (`FNX_EXPERT_DYN_GIB` below) |
| `FNX_EXPERT_RAM_GIB` | 52.5 | 29.0 | 71.7 GiB of experts (IQ3_S 48.8); the rest sizes the VRAM slots as before |
| `FNX_EXPERT_DYN_GIB`, `FNX_PLE_RAM` | 18, 0 | unset, 1 | the 64 GB mode's switches. With every expert pinned in RAM (73 GiB) 9 GiB of page cache were left for the 28.8 GB lookup table, and a 36k read ran at 146 tok/s. With the table read through the page cache and 18 GiB of the cards' experts dynamic (1.2 GiB stay in VRAM for good without a RAM copy), RAM holds 70.5 GiB of experts. 14 (5.2 GiB for good, 66.5 GiB in RAM) loads at 256k only and edits ran 12-14 % slower: the experts kept for good are chosen by a general route profile, the dynamic ones follow the work |
| `FNX_PF_TAIL_MIB` | 4096 | 3000 | the bigger staging buffers (about 1.95 GiB a card, IQ3_S about 1). Staging buffers of half the size (`FNX_PF_STAGE_MIB=512`) need 2 GiB less of RAM copies, but reads ran 19-33 % slower |
| `FNX_SH_IN` | 0 | unset (1) | ours: 0 sends the Q4_K layers through the P100 row kernel with the shared expert beside it, instead of Strata's decode kernels with the shared expert inside (writing +7 %; rounding) |
| `FNX_PREFETCH_MIN_T`, `FNX_PREFETCH_WANT` | 2, 2 | 4, 6 | next-layer copies from windows of 2 rows on, of 2 guessed experts a row: every guessed copy is 1.46 MiB a card that competes with the window's own missing experts on PCIe. Edits at 512k (our rounds `q4k5`, `q4k7`, py / md tok/s): 9 a row 54 / 57, 6 a row 67.4-67.9 / 69.3-70.3, 4: 73.5-75.3 / 78.1-79.3, 3: 76.1-77.7 / 81.7-83.3, 2: 77.4-78.0 / 81.9-84.3, prefetch off: 72.2-72.6 / 77.2-78.1; fresh writing 55.8-57.5 in all. With IQ3_S, 4 a row and from 2 rows gave nothing |

## Our other switches

Off, or at the value the engine picks, unless you set them. None is needed to reproduce our numbers.

| Switch | Unset | What it does |
|---|---|---|
| `FNX_P100_ATTN`, `FNX_P100_ROWS`, `FNX_P100_OVERLAP` | on for sm < 70 | the P100 prompt attention, the row kernel for decode experts, and copying missing experts beside it; 0 falls back to Strata's kernels |
| `FNX_OVER_MIN_T` | 3 | windows of fewer rows do not overlap copies with compute |
| `FNX_DRAFT_COST_LIVE` | 1 | 0: the draft planner keeps the starting costs |
| `FNX_PARK_RESERVE_MIB` | 3072 | RAM kept free when a conversation is parked in the conversation cache |
| `FNX_PF_EMBED_THREADS` | 12 | CPU threads that build a prompt chunk's input rows (embeddings) when it has 256 tokens or more |
| `FNX_POOL_SPIN_MS` | 100 | how long the second card's host thread spins for work before it sleeps (a sleeping one wakes 50-100 µs later) |
| `FNX_CVEC`, `FNX_CVEC_LAYERS`, `FNX_CVEC_MODE`, `FNX_CVEC_DIR` | off | a control vector, as Strata's `--control-vector-scaled` / layer range / mode / direction; changes the answers by design |
| `FNX_ROUTE_PRIOR_FILE`, `FNX_PREFETCH_FILE`, `FNX_DENSE_H_FILE`, `FNX_PF_REFILL_FILE`, `FNX_NOC_FILE` | off | read the matching setting from a file at run time (for A/B rounds in one load) |
| `FNX_DENSE_H`, `FNX_DENSE_H_MIN_T` | off | dense products in half precision for wide windows (lossy at the 1e-4 level; +0.4 % or less in the engine) |
| `FNX_HC_F16`, `FNX_HC_UP2` | off | `hc` variants (f16 dots: fresh −7 %; earlier weight loads: no gain) |
| `FNX_MEMO`, `FNX_MEMO_CAP` | off | routing memory for edits (edits −7 to −14 %) |
| `FNX_ROWS_BT`, `FNX_ROWS_BT2`, `FNX_ROWS_G2`, `FNX_ROWS_LDCG` | 256, 256, 0, 0 | row-kernel block sizes, lanes per row in phase 2, weights past L1 (none faster in the engine) |
| `FNX_RR_FAST`, `FNX_RR_ORDER`, `FNX_RR_RQ4`, `FNX_PRIO` | off | router variants and stream priority (same bits, no gain) |
| `FNX_MOE_X`, `FNX_MOE_XD` | off | IQ3_S decode expert kernels with the xmad dot product (same bits, no gain: those kernels are memory-bound) |
| `FNX_PF_REFILL`, `FNX_PF_REFILL_BATCH` | off, 4 | refill the lent expert slots after a read (no gain) |
| `FNX_PF_SP_MIN` | 3584 | with `FNX_PF_SP=1`: the rows a pass's second chunk needs for it; shorter passes run as before (their exchange would wait) |
| `FNX_NUMA_EXPERTS` | off | 1: each card's RAM copy of the experts on the memory of the CPU the card hangs from (same bits; IQ3_S and Q4 in the noise) |
| `FNX_PLE_WARM` | off | 1, with `FNX_PLE_RAM=0`: a thread reads the lookup table once after the load (Q4: writing −20 %, it pushes hot rows out of the page cache) |
| `FNX_PF_DRAFT_WINDOW` | off | run the draft layer over a prompt's last N rows only |
| `FNX_RR_TIMES`, `FNX_PA_CHECK`, `FNX_QSA_DUMP`, `FNX_QSA_DUMP_EVERY`, `FNX_QSA_DUMP_MAX`, `FNX_NVPROF_PASS`, `FNX_NVPROF_WIN`, `FNX_NVPROF_WINS` | off | diagnostics: router timings, kernel cross-checks, score dumps, `nvprof` capture of one prompt pass or a run of windows |

## Strata's switches

The engine keeps every switch of Strata v0.2.10's split engine. The ones not covered above are listed here; see
Strata's source (`src/split/`) and `docs/SPLIT_ENGINE.md` for their meaning. Those for AMD cards, peer access and RCCL do
nothing in this CUDA build.

`FNX_ADMIT_INTERLEAVE` `FNX_ARENA_BASE_PAGES` `FNX_ARENA_CHURN` `FNX_ARENA_RESERVE_MIB` `FNX_BATCH_DRAFT` `FNX_BATCH_DRAFTS`
`FNX_BATCH_FIXED` `FNX_BATCH_HORIZON` `FNX_BATCH_STATS` `FNX_BATCH_TRACE` `FNX_CAPTURE_AFTER` `FNX_DRAFT_ADAPT` `FNX_DRAFT_PEN`
`FNX_DRAFT_SAMPLE` `FNX_DUMP_FIRST_LOGITS` `FNX_HC1` `FNX_HC_BAR_WRAP` `FNX_HC_GRID` `FNX_HIP_MEMOP` `FNX_KV8` `FNX_KV_PAGES`
`FNX_KV_RESERVE_MIB` `FNX_MTP_EXPERTS` `FNX_NO_TWIN` `FNX_ONEHOP` `FNX_PEER` `FNX_PEER_PARTS` `FNX_PEER_PROBE_FAIL`
`FNX_PFX_ATTN_DMA` `FNX_PF_ATT_PIECES` `FNX_PF_DMA` `FNX_PF_DMA_BARRIER` `FNX_PF_DMA_CHECK` `FNX_PF_DRAFT8` `FNX_PF_GATHER`
`FNX_PF_GDN_CHECK` `FNX_PF_GEMM_LOG` `FNX_PF_HC_CHECK` `FNX_PF_OVERLAP` `FNX_PF_PASS` `FNX_PF_QSA_ROWS` `FNX_PF_RCCL`
`FNX_PF_RCCL_F32` `FNX_PF_RING_SM` `FNX_PF_ROWS` `FNX_PF_SCORES` `FNX_PF_SCORE_MIB` `FNX_PF_SEND` `FNX_PF_SEQ_START`
`FNX_PF_SKIP` `FNX_PF_SKIP_CONFIRM` `FNX_PF_TIMING` `FNX_PF_WINDOWS` `FNX_PRECAP` `FNX_PRECAP_GRAPHS` `FNX_PRECAP_TRACE`
`FNX_PREFETCH_PROBE` `FNX_QSA_DENSE` `FNX_RCCL_LIB` `FNX_RING_F32` `FNX_RING_GRID` `FNX_RING_TAG_START` `FNX_ROUTE_COUNT`
`FNX_ROUTE_FILE` `FNX_ROUTE_TRACE` `FNX_ROUTE_TRACE_FILE` `FNX_SELFTEST` `FNX_SELFTEST_FAIL` `FNX_SKIP_BAR1_CHECK`

`FNX_DUMP_FIRST_LOGITS=<path prefix>` is what we used to compare first-token logits between builds.
