"""Plug the trained soft demodulator into decoder.decode_slot() (batched over all hypotheses)."""
import numpy as np
import torch

from . import decoder, model, ft8
from . import frontend as fe


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    net = model.build(ck["arch"]).to(device).eval()
    net.load_state_dict(ck["state_dict"])
    return net


def make_llr_fn(path, also_classical=False, device="cpu", batch=128, n_valid=None, erase=False, mask_median=False):
    """n_valid: audio samples (12 kHz) that carry signal, for an early pass; with erase, the LLRs of data symbols
    that end past them are set to 0, with mask_median the feature normalisation ignores those symbols."""
    net = load(path, device)
    nv_bb = None if n_valid is None else int(n_valid * fe.FS_BB / ft8.FS)

    @torch.no_grad()
    def nn_fn(hyps):
        y = torch.as_tensor(np.stack([h.y for h in hyps]), dtype=torch.complex64, device=device)
        start = torch.tensor([h.start for h in hyps], device=device)
        out = []
        for i in range(0, len(hyps), batch):
            f = model.features(y[i : i + batch], start[i : i + batch], nv_bb if mask_median else None)
            out.append(net(f).double().cpu().numpy())
        llr = np.concatenate(out)
        if erase and nv_bb is not None:
            st = np.array([h.start for h in hyps])[:, None]
            gone = st + fe.SPS_BB * (ft8.DATA_POS[None, :] + 1) > nv_bb  # (H, 58)
            llr[np.repeat(gone, 3, axis=1)] = 0
        return [[l] for l in llr]

    def llr_fn(hyps):
        res = nn_fn(hyps)
        if also_classical:
            for r, cl in zip(res, decoder.classical_metrics(hyps)):
                r.extend(cl)
        return res

    llr_fn.nn = nn_fn  # NN metric only, for stages that use nothing else (decoder._nn_only)
    return llr_fn
