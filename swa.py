"""Average the EMA weights of several checkpoints (uniform SWA) into one.

Usage: python swa.py <task> <src_tag> <dst_tag> <ep> [<ep> ...]
   e.g. python swa.py can _interp _interp_swa 300 400 500 600
Writes outputs/<task>/cl_sfp<dst_tag>/ep<last>.ckpt so the usual eval path
(--tag <dst_tag> --ckpt <last>) picks it up.
"""

import sys
from pathlib import Path

import torch

task, src, dst, *eps = sys.argv[1:]
eps = [int(e) for e in eps]
root = Path(__file__).resolve().parent / "outputs" / task
blobs = [torch.load(root / f"cl_sfp{src}" / f"ep{e}.ckpt", map_location="cpu", weights_only=False)
         for e in eps]
sd = {k: sum(b["state_dict"][k].float() for b in blobs) / len(blobs)
      if blobs[0]["state_dict"][k].is_floating_point() else blobs[0]["state_dict"][k]
      for k in blobs[0]["state_dict"]}
meta = {k: v for k, v in blobs[-1].items() if k != "state_dict"}
meta.update(state_dict=sd, tag=dst, swa_of=eps)
out = root / f"cl_sfp{dst}"
out.mkdir(parents=True, exist_ok=True)
torch.save(meta, out / f"ep{eps[-1]}.ckpt")
print(f"saved {out / f'ep{eps[-1]}.ckpt'}  (average of {eps})")
