"""Plug the trained soft demodulator into decoder.decode_slot() (batched over all hypotheses)."""
import numpy as np
import torch

from . import decoder, model


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    net = model.build(ck["arch"]).to(device).eval()
    net.load_state_dict(ck["state_dict"])
    return net


def make_llr_fn(path, also_classical=False, device="cpu", batch=128):
    net = load(path, device)

    @torch.no_grad()
    def llr_fn(hyps):
        y = torch.as_tensor(np.stack([h.y for h in hyps]), dtype=torch.complex64, device=device)
        start = torch.tensor([h.start for h in hyps], device=device)
        out = []
        for i in range(0, len(hyps), batch):
            out.append(net(model.features(y[i : i + batch], start[i : i + batch])).double().cpu().numpy())
        nn = np.concatenate(out)
        res = [[l] for l in nn]
        if also_classical:
            for r, cl in zip(res, decoder.classical_metrics(hyps)):
                r.extend(cl)
        return res

    return llr_fn
