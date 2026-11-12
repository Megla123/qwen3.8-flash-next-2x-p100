#!/usr/bin/env python3
"""Writes a run config for Strata's server (serve/server.py --config) from flash-next-iq3s-512k.template.json.

    python3 profile/fill_profile.py --models /data/flash-next --mtp /data/flash-next/mtp/rt \
        --tokenizer /data/flash-next/tokenizer --vision-exe /opt/strata-v0.2.9/build-vision/bin/strata-vision \
        --logs /var/tmp --out my-profile.json

--models     the folder with both IQ3_S shards and mmproj-Qwen3.8-Flash-Next-BF16.gguf
--mtp        the MTP draft layer's runtime folder (dense.bin, dense.txt, experts.bin; README, "The MTP draft layer")
--tokenizer  the tokenizer folder (vocab.json, merges.txt, chat_template.jinja ...; README, "The tokenizer folder")
--vision-exe Strata's strata-vision; leave it out to run without pictures (the "vision" block and --vision go)
--route-prior FNX_ROUTE_PRIOR: 0 (default) is exact; 0.15 is our production setting, faster and lossy (README)
--draft-vocab another draft vocabulary than engine/data/draft_vocab_sl.bin, e.g. engine/data/draft_vocab_en.bin
--exe        the engine, if build.sh was run with another build directory (default build/split/strata-split-2)
--gpu        the two P100 as nvidia-smi numbers them (default "0,1")

Every path the config names is checked; a missing one is reported and nothing is written. Anything else (the context
length, other switches) is edited in the written JSON; docs/SWITCHES.md lists the switches.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", required=True)
    ap.add_argument("--mtp", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--vision-exe")
    ap.add_argument("--logs", default="/var/tmp")
    ap.add_argument("--route-prior", default="0")
    ap.add_argument("--draft-vocab", help="another draft vocabulary (default engine/data/draft_vocab_sl.bin)")
    ap.add_argument("--exe", default=str(RELEASE / "build" / "split" / "strata-split-2"))
    ap.add_argument("--gpu", default="0,1")
    ap.add_argument("--template", default=str(HERE / "flash-next-iq3s-512k.template.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if float(a.route_prior) < 0:
        sys.exit("--route-prior: 0 or more")
    cards = a.gpu.split(",")
    if len(cards) != 2 or not all(c.strip().isdigit() for c in cards) or cards[0].strip() == cards[1].strip():
        sys.exit("--gpu: two different card numbers, e.g. 0,1")
    values = {"{{RELEASE}}": str(RELEASE), "{{EXE}}": str(Path(a.exe).resolve()),
              "{{MODELS}}": str(Path(a.models).resolve()), "{{MTP}}": str(Path(a.mtp).resolve()),
              "{{TOKENIZER}}": str(Path(a.tokenizer).resolve()), "{{LOGS}}": str(Path(a.logs).resolve()),
              "{{VISION_EXE}}": str(Path(a.vision_exe).resolve()) if a.vision_exe else "", "{{ROUTE_PRIOR}}": a.route_prior}

    def fill(x):
        if isinstance(x, str):
            for k, v in values.items():
                x = x.replace(k, v)
            return x
        if isinstance(x, list):
            return [fill(v) for v in x]
        if isinstance(x, dict):
            return {k: fill(v) for k, v in x.items()}
        return x

    cfg = fill(json.loads(Path(a.template).read_text(encoding="utf-8")))
    cfg["gpu"] = ",".join(c.strip() for c in cards)
    if a.draft_vocab:
        cfg["args"][cfg["args"].index("--draft-vocab") + 1] = str(Path(a.draft_vocab).resolve())
    if not a.vision_exe:
        cfg.pop("vision")
        cfg["args"].remove("--vision")

    need = [cfg["args"][cfg["args"].index("--native") + 1], cfg["args"][cfg["args"].index("--draft-vocab") + 1],
            cfg["env"]["FNX_EXPERT_PROFILE"], cfg["tokenizer"] + "/vocab.json", cfg["tokenizer"] + "/chat_template.jinja",
            cfg["args"][cfg["args"].index("--mtp") + 1] + "/dense.bin", cfg["args"][cfg["args"].index("--mtp") + 1] + "/experts.bin",
            str(Path(cfg["args"][cfg["args"].index("--native") + 1]).with_name("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf")),
            cfg["exe"]]
    if "vision" in cfg:
        need += [cfg["vision"]["exe"], cfg["vision"]["mmproj"]]
    missing = [p for p in need if not Path(p).exists()]
    if missing:
        sys.exit("missing, nothing written:\n  " + "\n  ".join(missing))
    Path(a.out).write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {a.out} (FNX_ROUTE_PRIOR={a.route_prior}, pictures {'on' if 'vision' in cfg else 'off'})")


if __name__ == "__main__":
    main()
