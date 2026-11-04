"""Train the neural soft demodulator on infinite synthetic baseband data.

  uv run python train.py --steps 6000 --out runs/v1

Validation compares NN LLRs vs the classical nsym=1 metric on identical samples carrying real
codewords, decoded with BP (no OSD, for speed): the decode-rate gap is the sensitivity gain.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from ft8nn import ldpc, model, synth_bb


class Stream(IterableDataset):
    def __init__(self, batch, cfg):
        self.batch, self.cfg = batch, cfg

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        seed = (info.seed if info else 0) % 2**31
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        while True:
            y, bits, snr, _ = synth_bb.make_batch(self.batch, self.cfg, "cpu")
            yield model.features(y), bits, snr


def make_val(n, snr, cfg, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    payload = np.random.randint(0, 2, (n, 77), dtype=np.uint8)
    y, bits, _, _ = synth_bb.make_batch(n, cfg, "cpu", snr=snr, payload_bits=payload)
    return model.features(y), bits.numpy().astype(np.uint8)


VAL_CHANNELS = {
    "awgn": synth_bb.ChannelCfg(fade_p=0.0, qrm_p=0.0, drift_p=0.0, df_wide_p=0.0),
    "fade1Hz": synth_bb.ChannelCfg(fade_p=1.0, fspread_max=1.0, rician_p=0.0, qrm_p=0.0, drift_p=0.0, df_wide_p=0.0),
    "qrm": synth_bb.ChannelCfg(fade_p=0.0, qrm_p=1.0, drift_p=0.0, df_wide_p=0.0),
}
VAL_SNRS = {"awgn": [-20, -21, -22, -23], "fade1Hz": [-17, -19, -21], "qrm": [-19, -21]}


@torch.no_grad()
def evaluate(net, device, val, n_bp=400):
    net.eval()
    out = {}
    for key, (x, cw) in val.items():
        llr_nn = torch.cat([net(x[i : i + 256].to(device)).float().cpu() for i in range(0, len(x), 256)]).numpy()
        llr_cl = model.classical_llr_torch(x).numpy()
        r = {}
        for name, llr in (("nn", llr_nn), ("cl", llr_cl)):
            bits, ok, _ = ldpc.bp_decode(llr[:n_bp], iters=30)
            r[name] = float((ok & (bits == cw[:n_bp]).all(1)).mean())
        r["bce"] = float(torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(llr_nn), torch.tensor(cw, dtype=torch.float32)))
        out[key] = r
    net.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ff", type=int, default=384)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from")
    a = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arch = dict(d=a.d, layers=a.layers, ff=a.ff)
    net = model.SoftDemod(**arch).to(device)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=device)["state_dict"])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))
    )

    val = {}
    for ch, snrs in VAL_SNRS.items():
        for snr in snrs:
            val[f"{ch}@{snr}"] = make_val(512, snr, VAL_CHANNELS[ch], seed=1000 + snr)

    dl = DataLoader(Stream(a.batch, synth_bb.ChannelCfg()), batch_size=None, num_workers=a.workers, prefetch_factor=4, persistent_workers=True)
    log = []
    t0 = time.time()
    run_loss = 0.0
    for step, (x, bits, snr) in enumerate(dl, 1):
        x, bits = x.to(device, non_blocking=True), bits.to(device, non_blocking=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x), bits)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss += loss.item()
        if step % 100 == 0:
            print(f"step {step:6d}  loss {run_loss / 100:.4f}  lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s", flush=True)
            run_loss = 0.0
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(net, device, val)
            log.append({"step": step, **ev})
            print("  eval " + "  ".join(f"{k}: nn {v['nn']:.2f} cl {v['cl']:.2f}" for k, v in ev.items()), flush=True)
            torch.save({"state_dict": net.state_dict(), "arch": arch, "step": step}, out / "model.pt")
            (out / "log.json").write_text(json.dumps(log, indent=1))
        if step >= a.steps:
            break


if __name__ == "__main__":
    main()
