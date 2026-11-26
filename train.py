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
    def __init__(self, batch, cfg, noise_bank=None):
        self.batch, self.cfg, self.noise_bank = batch, cfg, noise_bank

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        seed = (info.seed if info else 0) % 2**31
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        bank = np.load(self.noise_bank, mmap_mode="r") if self.noise_bank else None
        while True:
            y, bits, snr, _ = synth_bb.make_batch(self.batch, self.cfg, "cpu", noise_bank=bank)
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
    "multipath": synth_bb.ChannelCfg(fade_p=1.0, fspread_max=1.0, rician_p=0.0, multipath_p=1.0, multipath_delay_max=0.4, qrm_p=0.0, drift_p=0.0, df_wide_p=0.0),
    "cochan": synth_bb.ChannelCfg(fade_p=0.0, qrm_p=0.0, drift_p=0.0, df_wide_p=0.0, df_sigma=0.08, dt_sigma=0.4, costas_att_p=1.0, costas_att_db=(12.0, 12.0)),
    "qrm": synth_bb.ChannelCfg(fade_p=0.0, qrm_p=1.0, drift_p=0.0, df_wide_p=0.0),
    "qrmclose": synth_bb.ChannelCfg(fade_p=0.0, qrm_p=1.0, qrm_off=(2.0, 10.0), qrm_rel_db=(0.0, 10.0), drift_p=0.0, df_wide_p=0.0, df_sigma=0.08, dt_sigma=0.4),
    "strong": synth_bb.ChannelCfg(fade_p=0.3, qrm_p=0.5, drift_p=0.0, df_wide_p=0.0, df_sigma=0.08, dt_sigma=0.4),
    "gainburst": synth_bb.ChannelCfg(fade_p=0.0, qrm_p=0.0, drift_p=0.0, df_wide_p=0.0, df_sigma=0.08, dt_sigma=0.4, gainstep_p=0.5, burst_p=0.5),
}
VAL_SNRS = {
    "awgn": [-20, -21, -22, -23],
    "fade1Hz": [-17, -19, -21],
    "multipath": [-17, -19, -21],
    "cochan": [-19, -21],
    "qrm": [-19, -21],
    "qrmclose": [-12, -16],
    "strong": [0, 15],
    "gainburst": [-14, 5],
}


def make_noise_val(n=512, seed=9999):
    """Signal-absent validation batches (noise+ghosts, and wrong-Costas 8-FSK) for LLR calibration."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = synth_bb.ChannelCfg(
        snr_lo=-20.0, snr_hi=5.0, qrm_p=0.5, qrm_max=2, qrm_off=(0.0, 60.0),
        qrm_rel_db=(-5.0, 15.0), burst_p=0.15, neg_p=1.0, neg_wrong_costas_p=0.0,
    )
    y_n, _, _, _ = synth_bb.make_batch(n, cfg, "cpu")
    cfg_wc = synth_bb.ChannelCfg(
        snr_lo=-18.0, snr_hi=15.0, qrm_p=0.0, neg_p=1.0, neg_wrong_costas_p=1.0,
    )
    y_wc, _, _, _ = synth_bb.make_batch(n, cfg_wc, "cpu")
    return {"noise_ee": model.features(y_n), "wc_ee": model.features(y_wc)}


@torch.no_grad()
def evaluate(net, device, val, n_bp=400, noise_x=None):
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
    if noise_x is not None:
        for nkey, nx in noise_x.items():
            llr_n = torch.cat([net(nx[i : i + 256].to(device)).float().cpu() for i in range(0, len(nx), 256)])
            ee = torch.sigmoid(-llr_n.abs()).sum(1).numpy()
            out[nkey] = {"mean": float(ee.mean()), "p05": float(np.percentile(ee, 5)), "min": float(ee.min())}
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
    ap.add_argument("--cfg", default="v1", choices=["v1", "v2", "v4"], help="training channel distribution")
    ap.add_argument("--eval-only", action="store_true", help="evaluate --init checkpoint on validation sets and exit")
    ap.add_argument("--kind", default="tf", choices=list(model.KINDS), help="tf = SoftDemod (default), conv, gru")
    ap.add_argument("--blocks", type=int, default=5, help="conv: residual blocks")
    ap.add_argument("--hidden", type=int, default=152, help="gru: hidden size per direction")
    ap.add_argument("--gru-layers", type=int, default=2)
    ap.add_argument("--teacher", default=None, help="distil: also fit this checkpoint's P(bit=1) (soft targets)")
    ap.add_argument("--kd-weight", type=float, default=0.5, help="weight of the soft-target loss (rest: true bits)")
    ap.add_argument("--noise-bank", default=None, help="real-noise segments (.npy, tools/make_noise_bank.py)")
    ap.add_argument("--real-noise-p", type=float, default=0.5, help="share of training rows with real noise (with --noise-bank)")
    ap.add_argument("--trunc-p", type=float, default=0.0, help="share of training rows whose audio stops mid-frame "
                    "(an early pass, synth_bb trunc_sym), the rest zero")
    ap.add_argument("--seed", type=int, default=None, help="training-data seed (default: fixed by the validation sets, so reruns are identical)")
    a = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arch = {"tf": dict(d=a.d, layers=a.layers, ff=a.ff), "conv": dict(kind="conv", d=a.d, blocks=a.blocks),
            "gru": dict(kind="gru", d=a.d, hidden=a.hidden, layers=a.gru_layers)}[a.kind]
    net = model.build(arch).to(device)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=device)["state_dict"])
    teacher = None
    if a.teacher:
        ck = torch.load(a.teacher, map_location=device)
        teacher = model.build(ck["arch"]).to(device).eval()
        teacher.load_state_dict(ck["state_dict"])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    warm = 100 if a.init else 300
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))
    )

    val = {}
    for ch, snrs in VAL_SNRS.items():
        for snr in snrs:
            val[f"{ch}@{snr}"] = make_val(512, snr, VAL_CHANNELS[ch], seed=1000 + snr)
    noise_x = make_noise_val()

    def _fmt_ev(ev):
        parts = [f"{k}: nn {v['nn']:.2f} cl {v['cl']:.2f}" for k, v in ev.items() if k not in ("noise_ee", "wc_ee")]
        for nk in ("noise_ee", "wc_ee"):
            if nk in ev:
                ne = ev[nk]
                parts.append(f"{nk}: mean {ne['mean']:.1f} p05 {ne['p05']:.1f} min {ne['min']:.1f}")
        return "  ".join(parts)

    if a.eval_only:
        ev = evaluate(net, device, val, noise_x=noise_x)
        print("  eval " + _fmt_ev(ev), flush=True)
        return

    cfg = {"v1": synth_bb.ChannelCfg, "v2": synth_bb.v2_cfg, "v4": synth_bb.v4_cfg}[a.cfg]()
    if a.noise_bank:
        cfg.real_noise_p = a.real_noise_p
    cfg.trunc_p = a.trunc_p
    if a.seed is not None:
        torch.manual_seed(a.seed)  # DataLoader worker seeds derive from this
    dl = DataLoader(Stream(a.batch, cfg, a.noise_bank), batch_size=None, num_workers=a.workers, prefetch_factor=4, persistent_workers=True)
    log = []
    t0 = time.time()
    run_loss = 0.0
    for step, (x, bits, snr) in enumerate(dl, 1):
        x, bits = x.to(device, non_blocking=True), bits.to(device, non_blocking=True)
        logits = net(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, bits)
        if teacher is not None:
            with torch.no_grad():
                soft = torch.sigmoid(teacher(x))
            loss = (1 - a.kd_weight) * loss + a.kd_weight * torch.nn.functional.binary_cross_entropy_with_logits(logits, soft)
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
            ev = evaluate(net, device, val, noise_x=noise_x)
            log.append({"step": step, **ev})
            print("  eval " + _fmt_ev(ev), flush=True)
            torch.save({"state_dict": net.state_dict(), "arch": arch, "step": step}, out / "model.pt")
            (out / "log.json").write_text(json.dumps(log, indent=1))
        if step >= a.steps:
            break


if __name__ == "__main__":
    main()
