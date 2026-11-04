"""Neural soft demodulator: baseband symbol spectra -> 174 bit LLRs.

Input features: per symbol, a 64-point (2x zero-padded) DFT of the 32 baseband samples,
bins -4..19 (3.125 Hz spacing; tones sit on even bins 0..14). 24 complex bins -> 48 reals.
Features are normalised by the median bin power of the candidate (robust local noise estimate).
"""
import math

import torch
from torch import nn

from . import ft8

NBIN_LO, NBIN_HI = -4, 20  # bins at 3.125 Hz, relative to tone 0
NBINS = NBIN_HI - NBIN_LO
SPS = 32
START0 = 100


def features(y, start=None):
    """y: (B, N) complex baseband (200 Hz), symbol 0 at sample `start` (int or (B,) tensor).

    Returns (B, 79, 2*NBINS) float.
    """
    B = y.shape[0]
    if start is None:
        start = torch.full((B,), START0, device=y.device, dtype=torch.long)
    elif not torch.is_tensor(start):
        start = torch.full((B,), int(start), device=y.device, dtype=torch.long)
    idx = start[:, None] + torch.arange(79 * SPS, device=y.device)[None, :]
    valid = (idx >= 0) & (idx < y.shape[1])
    seg = torch.gather(y, 1, idx.clamp(0, y.shape[1] - 1)) * valid
    seg = seg.view(B, 79, SPS)
    Z = torch.fft.fft(seg, n=2 * SPS, dim=-1) / math.sqrt(SPS)
    bins = torch.arange(NBIN_LO, NBIN_HI, device=y.device) % (2 * SPS)
    Z = Z[..., bins]  # (B,79,24)
    p = Z.abs().pow(2)
    scale = p.flatten(1).median(1).values.clamp(min=1e-6).sqrt()[:, None, None]
    Z = Z / scale
    return torch.cat([Z.real, Z.imag], -1)


def _conv_bl(h, conv):
    """'same' Conv1d on (B, L, C) via unfold + matmul. Equivalent to conv(h.transpose(1,2)).transpose(1,2)
    but ~30x faster on ARM CPUs, where torch's conv1d falls back to a slow kernel."""
    k = conv.kernel_size[0]
    hp = nn.functional.pad(h.transpose(1, 2), (k // 2, k // 2))  # (B, C, L+k-1)
    cols = hp.unfold(2, k, 1).permute(0, 2, 1, 3).reshape(h.shape[0], h.shape[1], -1)  # (B, L, C*k)
    return cols @ conv.weight.reshape(conv.out_channels, -1).T + conv.bias


class SoftDemod(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, ff=384, conv_k=5, dropout=0.0):
        super().__init__()
        f_in = 2 * NBINS
        self.inp = nn.Linear(f_in, d)
        self.pos = nn.Parameter(torch.randn(1, 79, d) * 0.02)
        self.conv = nn.Sequential(
            nn.Conv1d(d, d, conv_k, padding=conv_k // 2), nn.GELU(),
            nn.Conv1d(d, d, conv_k, padding=conv_k // 2), nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 3)
        self.register_buffer("data_pos", torch.tensor(ft8.DATA_POS), persistent=False)

    def forward(self, x):
        """x: (B,79,F) features -> (B,174) logits of P(bit=1)."""
        h = self.inp(x) + self.pos
        c1, c2 = self.conv[0], self.conv[2]
        gelu = nn.functional.gelu  # same as self.conv (Conv, GELU, Conv, GELU), fast path
        h = h + gelu(_conv_bl(gelu(_conv_bl(h, c1)), c2))
        h = self.tf(h)
        h = self.out(self.norm(h[:, self.data_pos]))  # (B,58,3)
        return h.reshape(h.shape[0], 174)


def classical_llr_torch(x):
    """nsym=1 max-log metric from the same features (tone bins = even bins 0..14), for comparison."""
    B = x.shape[0]
    re, im = x[..., :NBINS], x[..., NBINS:]
    mag = torch.sqrt(re**2 + im**2)[:, :, [b - NBIN_LO for b in range(0, 16, 2)]]  # (B,79,8)
    s = mag[:, torch.tensor(ft8.DATA_POS, device=x.device)]  # (B,58,8)
    vals = torch.tensor(ft8.GRAY_INV, device=x.device)
    out = []
    for b in range(3):
        bit = ((vals >> (2 - b)) & 1).bool()
        out.append(s[..., bit].max(-1).values - s[..., ~bit].max(-1).values)
    llr = torch.stack(out, -1).reshape(B, 174)
    return 2.83 * llr / llr.std(1, keepdim=True).clamp(min=1e-9)
