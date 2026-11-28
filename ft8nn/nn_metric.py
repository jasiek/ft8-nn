"""Plug the trained soft demodulator into decoder.decode_slot() (batched over all hypotheses)."""
import os
import sys
from pathlib import Path

import numpy as np
import torch

from . import decoder, model, ft8
from . import frontend as fe

# Option 3B pre-screener threshold for pass hypotheses (tuned on validation run 1): active by default on non-macOS
# ORT platforms (macOS runs full SoftDemod natively via Accelerate/AMX in C); override with FT8NN_SCREENER=0|1.
SCREEN_EE_GATE = 69.0


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    net = model.build(ck["arch"]).to(device).eval()
    net.load_state_dict(ck["state_dict"])
    return net


def make_llr_fn(path, also_classical=False, device="cpu", batch=128, n_valid=None, erase=False, mask_median=False):
    """n_valid: audio samples (12 kHz) that carry signal, for an early pass; with erase, the LLRs of data symbols
    that end past them are set to 0, with mask_median the feature normalisation ignores those symbols."""
    net = load(path, device)
    scr_env = os.environ.get("FT8NN_SCREENER")
    use_scr = (scr_env == "1") if scr_env in ("0", "1") else (sys.platform != "darwin")
    scr_path = Path(path).parent / "screener.pt"
    scr_net = load(scr_path, device) if (use_scr and Path(path).name != "screener.pt" and scr_path.exists()) else None
    nv_bb = None if n_valid is None else int(n_valid * fe.FS_BB / ft8.FS)

    def _finish(llr, hyps):
        if erase and nv_bb is not None:
            st = np.array([h.start for h in hyps])[:, None]
            gone = st + fe.SPS_BB * (ft8.DATA_POS[None, :] + 1) > nv_bb  # (H, 58)
            llr[np.repeat(gone, 3, axis=1)] = 0
        return [[l] for l in llr]

    @torch.no_grad()
    def nn_fn(hyps):
        y = torch.as_tensor(np.stack([h.y for h in hyps]), dtype=torch.complex64, device=device)
        start = torch.tensor([h.start for h in hyps], device=device)
        out = []
        for i in range(0, len(hyps), batch):
            f = model.features(y[i : i + batch], start[i : i + batch], nv_bb if mask_median else None)
            out.append(net(f).float().cpu().numpy().astype(np.float64))
        return _finish(np.concatenate(out), hyps)

    @torch.no_grad()
    def nn_pass_fn(hyps):
        if scr_net is None:
            return nn_fn(hyps)
        y = torch.as_tensor(np.stack([h.y for h in hyps]), dtype=torch.complex64, device=device)
        start = torch.tensor([h.start for h in hyps], device=device)
        out = []
        for i in range(0, len(hyps), batch):
            f = model.features(y[i : i + batch], start[i : i + batch], nv_bb if mask_median else None)
            see = scr_net(f).float()
            keep = see < SCREEN_EE_GATE
            blk = np.zeros((len(f), 174), dtype=np.float64)
            if keep.any():
                blk[keep.cpu().numpy()] = net(f[keep]).float().cpu().numpy().astype(np.float64)
            out.append(blk)
        return _finish(np.concatenate(out), hyps)

    def llr_fn(hyps):
        res = nn_fn(hyps)
        if also_classical:
            for r, cl in zip(res, decoder.classical_metrics(hyps)):
                r.extend(cl)
        return res

    llr_fn.nn = nn_fn            # unscreened NN metric for targeted post-stages (co-channel, refine, repeat)
    llr_fn.nn_pass = nn_pass_fn  # screened NN metric for main candidate-search passes
    llr_fn.also_classical = bool(also_classical)
    return llr_fn
