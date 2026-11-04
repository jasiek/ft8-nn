"""Batched, on-the-fly FT8 training data at 200 Hz complex baseband (torch, runs on MPS/CPU).

Matches frontend.Baseband.downsample(): same 16 s buffer (3200 samples), same frequency mask and
taper, same noise normalisation (in-band DFT-bin noise variance = NBB/(hi-lo) = 3.2).
SNR convention: WSJT-X 2500 Hz reference bandwidth.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

from . import frontend as fe
from . import ft8

NBB = fe.NBB
SPS = fe.SPS_BB
START0 = int(ft8.NOMINAL_START * fe.FS_BB)  # nominal symbol-0 sample (100)
_LO = -int(round(1.5 * ft8.TONE_SPACING / fe.DF_FULL))  # -150
_HI = int(round(8.5 * ft8.TONE_SPACING / fe.DF_FULL))  # 850
NOISE_VAR = NBB / (_HI - _LO)  # 3.2
# |a|^2 per unit SNR. The real front end zero-pads 15 s -> 16 s before estimating the noise PSD,
# which dilutes the estimate by NFFT_FULL/SLOT_SAMPLES (+0.28 dB); mirror that here.
SIG_POW_PER_SNR = 2500.0 * NOISE_VAR / fe.FS_BB * fe.NFFT_FULL / ft8.SLOT_SAMPLES


def band_mask(device):
    k = np.arange(_LO, _HI)
    w = np.ones(len(k))
    t = 0.5 * (1 - np.cos(np.pi * np.arange(101) / 100))
    w[:101] = t
    w[-101:] = t[::-1]
    m = np.zeros(NBB)
    m[k % NBB] = w
    return torch.tensor(m, dtype=torch.float32, device=device)


def _pulse(device):
    p = ft8._gfsk_pulse(2.0, (np.arange(3 * SPS) - 1.5 * SPS) / SPS)
    return torch.tensor(p, dtype=torch.float32, device=device).view(1, 1, -1)


def gfsk_baseband(tones, f0, drift, device):
    """tones (B,79) long, f0 (B,) Hz, drift (B,) Hz/s -> complex64 (B, 79*32), unit amplitude."""
    B = tones.shape[0]
    t = tones.float()
    t_ext = torch.cat([t[:, :1], t, t[:, -1:]], 1).unsqueeze(1)  # (B,1,81)
    dphi = torch.nn.functional.conv_transpose1d(t_ext, _pulse(device), stride=SPS)[:, 0, 2 * SPS : 2 * SPS + 79 * SPS]
    dphi = dphi * (2 * math.pi / SPS)
    n = torch.arange(79 * SPS, device=device, dtype=torch.float32)
    tsec = n / fe.FS_BB
    finst = f0[:, None] + drift[:, None] * tsec[None, :]
    dphi = dphi + 2 * math.pi * finst / fe.FS_BB
    phi = torch.cumsum(dphi, 1) - dphi  # phi[0]=0
    nr = SPS // 8
    env = torch.ones(79 * SPS, device=device)
    r = 0.5 * (1 - torch.cos(math.pi * torch.arange(nr, device=device) / nr))
    env[:nr] = r
    env[-nr:] = r.flip(0)
    return torch.polar(env.expand(B, -1).contiguous(), phi)


def doppler_fading(B, n, fspread, device, gen=None):
    """Complex Gaussian fading (B,n), unit power, Gaussian Doppler spectrum width fspread (B,) Hz.
    Rows with fspread==0 get 1."""
    nfft = 1 << int(math.ceil(math.log2(n)))
    f = torch.fft.fftfreq(nfft, 1 / fe.FS_BB, device=device)
    fs_ = fspread.clamp(min=1e-3)[:, None]
    shape = torch.exp(-((f[None, :] / (0.5 * fs_)) ** 2))
    x = torch.complex(torch.randn(B, nfft, device=device), torch.randn(B, nfft, device=device)) * shape.sqrt()
    y = torch.fft.ifft(x)[:, :n]
    y = y / y.abs().pow(2).mean(1, keepdim=True).sqrt()
    return torch.where((fspread > 0)[:, None], y, torch.ones_like(y))


@dataclass
class ChannelCfg:
    snr_lo: float = -27.0
    snr_hi: float = -14.0
    df_sigma: float = 0.25  # residual freq error after fine sync, Hz
    df_wide_p: float = 0.15  # prob. of a wide freq error (sync at very low SNR)
    df_wide: float = 1.2
    dt_sigma: float = 1.0  # residual timing error, samples at 200 Hz
    drift_p: float = 0.2
    drift_max: float = 0.15  # Hz/s
    fade_p: float = 0.5
    fspread_max: float = 2.5  # Hz
    rician_p: float = 0.3  # fraction of fading cases with a specular component
    qrm_p: float = 0.35  # prob. of an interfering FT8 signal in-band
    qrm_rel_db: tuple = (-6.0, 20.0)  # interferer power relative to wanted
    random_data: bool = True  # random data tones (labels are bits of tones), not codewords


def make_batch(B, cfg: ChannelCfg, device, snr=None, payload_bits=None):
    """Returns (y (B,3200) complex64 baseband, bits (B,174) float, snr (B,), tones (B,79)).

    If payload_bits (B,77) is given, transmits real FT8 codewords for those payloads.
    """
    if payload_bits is not None:
        cws = np.stack([ft8.encode77(p)[0] for p in payload_bits])
        tones = np.stack([ft8.codeword_to_tones(c) for c in cws])
        bits = cws
    elif cfg.random_data:
        vals = np.random.randint(0, 8, size=(B, 58))
        tones = np.zeros((B, 79), np.int64)
        for p in ft8.SYNC_POS:
            tones[:, p : p + 7] = ft8.COSTAS
        tones[:, ft8.DATA_POS] = ft8.GRAY[vals]
        bits = np.stack([(vals >> 2) & 1, (vals >> 1) & 1, vals & 1], -1).reshape(B, 174)
    else:
        payload = np.random.randint(0, 2, (B, 77), dtype=np.uint8)
        return make_batch(B, cfg, device, snr, payload)
    tones_t = torch.tensor(tones, device=device)
    bits_t = torch.tensor(bits, dtype=torch.float32, device=device)

    def U(lo, hi, n=B):
        return lo + (hi - lo) * torch.rand(n, device=device)

    if snr is None:
        snr = U(cfg.snr_lo, cfg.snr_hi)
    elif not torch.is_tensor(snr):
        snr = torch.full((B,), float(snr), device=device)

    df = torch.randn(B, device=device) * cfg.df_sigma
    wide = torch.rand(B, device=device) < cfg.df_wide_p
    df = torch.where(wide, U(-cfg.df_wide, cfg.df_wide), df)
    drift = torch.where(torch.rand(B, device=device) < cfg.drift_p, U(-cfg.drift_max, cfg.drift_max), torch.zeros(B, device=device))
    w = gfsk_baseband(tones_t, df, drift, device)

    # fading: Rayleigh or Rician with Gaussian Doppler spectrum
    fspread = torch.where(torch.rand(B, device=device) < cfg.fade_p, U(0.05, cfg.fspread_max), torch.zeros(B, device=device))
    g = doppler_fading(B, w.shape[1], fspread, device)
    K = torch.where(torch.rand(B, device=device) < cfg.rician_p, U(0.5, 5.0), torch.zeros(B, device=device))
    los = torch.sqrt(K / (K + 1))[:, None]
    g = torch.where((fspread > 0)[:, None], los + torch.sqrt(1 / (K + 1))[:, None] * g, g)
    phase = torch.polar(torch.ones(B, device=device), U(0, 2 * math.pi))
    amp = torch.sqrt(SIG_POW_PER_SNR * 10 ** (snr / 10))
    sig = (amp * phase)[:, None] * g * w

    y = torch.zeros(B, NBB, dtype=torch.complex64, device=device)
    y[:, START0 : START0 + w.shape[1]] = sig

    # interferer: another FT8 signal, offset in freq/time, possibly much stronger
    q = torch.rand(B, device=device) < cfg.qrm_p
    if q.any():
        nq = int(q.sum())
        qt = np.random.randint(0, 8, size=(nq, 79))
        for p in ft8.SYNC_POS:
            qt[:, p : p + 7] = ft8.COSTAS
        side = torch.where(torch.rand(nq, device=device) < 0.5, -1.0, 1.0)
        qf = torch.where(side < 0, U(-60.0, -8.0, nq), U(8.0, 60.0, nq))
        qw = gfsk_baseband(torch.tensor(qt, device=device), qf, torch.zeros(nq, device=device), device)
        qrel = U(*cfg.qrm_rel_db, nq)
        qamp = amp[q] * 10 ** (qrel / 20)
        qph = torch.polar(torch.ones(nq, device=device), U(0, 2 * math.pi, nq))
        qshift = torch.randint(-150, 150, (nq,), device=device)
        buf = torch.zeros(nq, NBB, dtype=torch.complex64, device=device)
        buf[:, START0 : START0 + qw.shape[1]] = (qamp * qph)[:, None] * qw
        idx = (torch.arange(NBB, device=device)[None, :] - qshift[:, None]) % NBB
        y[q] = y[q] + torch.gather(buf, 1, idx)

    noise = torch.complex(torch.randn(B, NBB, device=device), torch.randn(B, NBB, device=device)) * math.sqrt(NOISE_VAR / 2)
    y = y + noise

    # timing error (fractional) + band mask in frequency domain
    tau = torch.randn(B, device=device) * cfg.dt_sigma
    k = torch.fft.fftfreq(NBB, device=device) * NBB
    Y = torch.fft.fft(y) * band_mask(device)[None, :]
    Y = Y * torch.polar(torch.ones_like(k), -2 * math.pi * k / NBB * tau[:, None].float())
    y = torch.fft.ifft(Y)
    return y, bits_t, snr, tones_t
