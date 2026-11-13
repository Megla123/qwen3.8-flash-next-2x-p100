# Patch scripts (a record, not to be run)

The scripts that made the last steps of the port (11-13 November 2026), as they were run on our build machine. Each
edits the development tree in place, checks that every edit matched as often as expected, and some save the original
first. They name that machine's paths (`/srv/build/...`) and our test tools (`mikro/...`), which are not part of this
release. Their changes are already in `engine/`; [docs/HISTORY.md](../../docs/HISTORY.md) says what each step did, and
[docs/upstream-v0.2.10.diff](../../docs/upstream-v0.2.10.diff) holds the whole change against Strata.

| Script | Switch it added | Step |
|---|---|---|
| `nvp_patch.py`, `nvw_patch.py` | `FNX_NVPROF_PASS`, `FNX_NVPROF_WIN` | `nvprof` capture of one prompt pass or a run of windows |
| `rrt_patch.py`, `topk_patch.py`, `prio_patch.py` | `FNX_RR_TIMES`, `FNX_RR_FAST`, `FNX_PRIO` | router timing, a faster top-10, stream priority (no gain) |
| `stage_patch.py`, `prep_patch.py`, `jobv4_patch.py`, `jobv5_patch.py`, `jobv6_patch.py` | `FNX_PF_STAGE_ALL`, `FNX_PF_PREP`, `FNX_PF_JOBV` 4-6 | build `b1438fe1` |
| `xmad_patch.py` | none (the dot-product form per kernel and window size is fixed by constants at its top) | build `2ef67b0e` |
| `moe_x_patch.py`, `gen_iq3s_x.sh` | `FNX_MOE_X`, `FNX_MOE_XD` | the xmad dot product in the IQ3_S writing kernels (no gain, off) |
| `rr_order_patch.py`, `hc_up2_patch.py`, `rr_rq4_patch.py` | `FNX_RR_ORDER`, `FNX_HC_UP2`, `FNX_RR_RQ4` | router and `hc` ideas (no gain, off) |
| `jobv7_patch.py` | `FNX_PF_JOBV=7` | build `5ef3a6ae`, the first release |
| `sp_patch.py`, `sp2_patch.py` | `FNX_PF_SP`, `FNX_PF_SP_MIN` | build `bf9c7f1f`: each card reads `hc` only for its own rows of a prompt pass |
| `q4x_patch.py` | `FNX_SH_IN` (its `FNX_PLE_DISK` was dropped for `FNX_PLE_RAM`) | P100 kernels for UD-Q4_K_XL's expert types (Q4_K, Q5_K, Q5_1, Q8_0) |
| `numa_patch.py` | `FNX_NUMA_EXPERTS` | experts' RAM copy on the card's CPU (no gain, off) |
| `warm_patch.py` | `FNX_PLE_WARM` | read the lookup table once after the load (slower, off) |

The 64 GB mode (`FNX_PLE_RAM`, `FNX_EXPERT_DYN_GIB`, `FNX_EXPERT_DYN_MIN`) was written in this repository, not by a script: our
Q4 build took it from here. All of the above together are the sources of build `c213636c`.
