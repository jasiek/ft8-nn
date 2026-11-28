"""Train the lightweight SoftDemod pre-screener (models/v4/screener.pt, model.Screener).

  uv run python tools/train_screener.py --teacher models/v4/model.pt \
      --rec datasets/ft8-training-data/recordings --dec data/real --tag nn_v4_cpu \
      --noise-bank data/noise/train_bank_v4.npy --out models/v4/screener.pt

Extracts multi-pass candidate features and exact teacher expected-bit-error targets (v4_ee) from the
training recordings ONLY (never validation runs 1-4), augments with synthetic v4_cfg + truncated early-pass
batches, and trains model.Screener(d=48, layers=1, ff=144) with an asymmetric Huber + binary gate loss that
heavily penalises over-predicting exp_err on real signals (v4_ee < 69).
"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ft8nn import frontend as fe, ft8, model, nn_metric, synth_bb  # noqa: E402
from ft8nn.subtract import subtract  # noqa: E402


def _extract_slot(args):
    wav_path, decs_p0, do_early = args
    _, x = wavfile.read(wav_path)
    x = x.astype(float)
    x = np.pad(x, (0, max(0, 180000 - len(x))))[:180000]
    x = x / (np.std(x) + 1e-9)

    out_f = []
    cands0 = fe.find_candidates(x, 150.0, 4000.0, max_cand=900)
    bb0 = fe.Baseband(x)
    hyps0 = [h for c in cands0 for h in fe.fine_sync2(bb0, c, n_hyp=2)]
    if hyps0:
        yt = torch.as_tensor(np.stack([h.y for h in hyps0]), dtype=torch.complex64)
        st = torch.tensor([h.start for h in hyps0])
        out_f.append(model.features(yt, st).half().numpy())

    if decs_p0:
        x1 = x.copy()
        for f0, dt, hb in decs_p0:
            bits = np.array([int(c) for c in format(int(hb, 16), "077b")], np.uint8)
            subtract(x1, ft8.encode77(bits)[1], f0, dt)
        cands1 = fe.find_candidates(x1, 150.0, 4000.0, max_cand=450)
        bb1 = fe.Baseband(x1)
        hyps1 = [h for c in cands1 for h in fe.fine_sync2(bb1, c, n_hyp=2)]
        if hyps1:
            yt = torch.as_tensor(np.stack([h.y for h in hyps1]), dtype=torch.complex64)
            st = torch.tensor([h.start for h in hyps1])
            out_f.append(model.features(yt, st).half().numpy())

    if do_early:
        xe = x.copy()
        xe[141696:] = 0.0
        nv_bb = int(141696 * fe.FS_BB / ft8.FS)
        ecands = fe.find_candidates(xe, 150.0, 4000.0, max_cand=600)
        bbe = fe.Baseband(xe)
        ehyps = [h for c in ecands for h in fe.fine_sync2(bbe, c, n_hyp=2)]
        if ehyps:
            yt = torch.as_tensor(np.stack([h.y for h in ehyps]), dtype=torch.complex64)
            st = torch.tensor([h.start for h in ehyps])
            out_f.append(model.features(yt, st, nv_bb).half().numpy())

    return np.concatenate(out_f) if out_f else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="models/v4/model.pt")
    ap.add_argument("--rec", default="datasets/ft8-training-data/recordings")
    ap.add_argument("--dec", default="data/real")
    ap.add_argument("--tag", default="nn_v4_cpu")
    ap.add_argument("--noise-bank", default="data/noise/train_bank_v4.npy")
    ap.add_argument("--out", default="models/v4/screener.pt")
    ap.add_argument("--d", type=int, default=48)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--ff", type=int, default=144)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--per-station", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    teacher = nn_metric.load(a.teacher, dev)

    decs = {}
    for df in glob.glob(str(Path(a.dec) / "*" / f"decodes_{a.tag}.txt")):
        st = Path(df).parent.name
        for ln in open(df):
            p = ln.strip().split("\t")
            decs.setdefault((st, p[0]), []).append((float(p[1]), float(p[2]), p[5]))

    jobs = []
    for sd in sorted(glob.glob(str(Path(a.rec) / "*" / "slots"))):
        st = Path(sd).parent.name
        ws = sorted(glob.glob(os.path.join(sd, "*.wav")))[: a.per_station]
        for i, w in enumerate(ws):
            dl = decs.get((st, Path(w).stem), [])
            jobs.append((w, dl[: max(1, len(dl) * 2 // 3)] if i % 2 == 0 else [], i < 4))

    t0 = time.time()
    fs = []
    with ProcessPoolExecutor(a.workers) as ex:
        for f in ex.map(_extract_slot, jobs, chunksize=2):
            if f is not None:
                fs.append(f)
    F_real = torch.from_numpy(np.concatenate(fs))
    with torch.no_grad():
        V_real = torch.cat([
            torch.sigmoid(-teacher(F_real[i : i + 512].float().to(dev)).float().abs()).sum(1).cpu()
            for i in range(0, len(F_real), 512)
        ])
    print(f"Extracted {len(F_real)} training-recording hypotheses in {time.time() - t0:.1f}s", flush=True)

    cfg = synth_bb.v4_cfg()
    cfg.real_noise_p = 0.5 if a.noise_bank else 0.0
    cfg.trunc_p = 0.15
    cfg.neg_p = 0.35
    bank = np.load(a.noise_bank, mmap_mode="r") if a.noise_bank else None
    np.random.seed(42)
    torch.manual_seed(42)
    s_fs, s_vs = [], []
    with torch.no_grad():
        for _ in range(128):
            y, _, _, _ = synth_bb.make_batch(512, cfg, "cpu", noise_bank=bank)
            feat = model.features(y)
            lv = teacher(feat.to(dev)).float()
            s_fs.append(feat.half())
            s_vs.append(torch.sigmoid(-lv.abs()).sum(1).cpu())
    F_all = torch.cat([F_real, torch.cat(s_fs)], 0)
    V_all = torch.cat([V_real, torch.cat(s_vs)], 0)
    pos_m = V_all < 70.0
    F_pos, V_pos = F_all[pos_m], V_all[pos_m]
    F_neg, V_neg = F_all[~pos_m], V_all[~pos_m]

    torch.manual_seed(a.seed)
    arch = {"kind": "scr", "d": a.d, "layers": a.layers, "heads": 4, "ff": a.ff}
    net = model.build(arch).to(dev)
    with torch.no_grad():
        net.head[2].bias.fill_(65.0)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=1e-5)
    pos_w = torch.tensor(4.0, device=dev)

    t1 = time.time()
    for s in range(1, a.steps + 1):
        ip = torch.randint(0, len(F_pos), (192,))
        in_ = torch.randint(0, len(F_neg), (320,))
        xb = torch.cat([F_pos[ip], F_neg[in_]], 0).float().to(dev)
        yb = torch.cat([V_pos[ip], V_neg[in_]], 0).to(dev)

        pred = net(xb)
        err = pred - yb
        huber = F.smooth_l1_loss(pred, yb, reduction="none", beta=2.0)
        w = torch.ones_like(yb)
        w = torch.where((yb < 69.0) & (err > 0), torch.full_like(w, 10.0), w)
        w = torch.where((yb < 52.0) & (err > 0), torch.full_like(w, 25.0), w)
        loss = (w * huber).mean() * 0.1 + F.binary_cross_entropy_with_logits(
            (68.5 - pred) * 0.4, (yb < 68.5).float(), pos_weight=pos_w
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if s % 500 == 0:
            print(f"step {s:5d}/{a.steps}  loss {loss.item():.4f}  {time.time() - t1:.0f}s", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.cpu().state_dict(), "arch": arch, "step": a.steps}, a.out)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
