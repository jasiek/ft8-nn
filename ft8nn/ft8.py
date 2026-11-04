"""FT8 protocol primitives: CRC-14, LDPC(174,91) encode, tone mapping, GFSK waveform.

Bit-exact with WSJT-X (verified against `ft8code` in tests/test_encode.py).
Message *packing* (text <-> 77 bits) is not implemented here; payloads are 77-bit arrays.
"""
from pathlib import Path

import numpy as np
from scipy.special import erf

_T = np.load(Path(__file__).parent / "data" / "ldpc.npz")
LDPC_GEN = _T["gen"]  # (83, 91)
LDPC_NM = _T["Nm"]  # (83, 7), -1 padded
LDPC_MN = _T["Mn"]  # (174, 3)

H = np.zeros((83, 174), np.uint8)
for _m, _row in enumerate(LDPC_NM):
    H[_m, _row[_row >= 0]] = 1

COSTAS = np.array([3, 1, 4, 0, 6, 5, 2])
GRAY = np.array([0, 1, 3, 2, 5, 6, 4, 7])  # 3-bit value -> tone
GRAY_INV = np.argsort(GRAY)  # tone -> 3-bit value

NN = 79  # channel symbols
ND = 58  # data symbols
SYNC_POS = np.array([0, 36, 72])
DATA_POS = np.r_[7:36, 43:72]  # 58 data symbol positions
SYNC_SYMS = np.r_[0:7, 36:43, 72:79]

FS = 12000
NSPS = 1920  # samples per symbol at 12 kHz (0.16 s)
TONE_SPACING = FS / NSPS  # 6.25 Hz
SLOT_SAMPLES = 15 * FS
NOMINAL_START = 0.5  # seconds; DT=0 means signal starts 0.5 s into the slot

CRC_POLY = 0x2757


def crc14(bits):
    """CRC-14 over a bit array (MSB first), as in WSJT-X: computed over 77 payload bits + 5 zeros."""
    rem = 0
    for b in bits:
        rem ^= int(b) << 13
        rem = ((rem << 1) ^ CRC_POLY) if rem & 0x2000 else (rem << 1)
        rem &= 0x3FFF
    return rem


def add_crc(payload77):
    payload77 = np.asarray(payload77, np.uint8)
    c = crc14(np.r_[payload77, np.zeros(5, np.uint8)])
    crc_bits = np.array([(c >> (13 - i)) & 1 for i in range(14)], np.uint8)
    return np.r_[payload77, crc_bits]


def crc_ok(a91):
    return crc14(np.r_[a91[:77], np.zeros(5, np.uint8)]) == int("".join(map(str, a91[77:91])), 2)


def ldpc_encode(a91):
    """91 info bits -> 174-bit systematic codeword [a91 | parity83]."""
    a91 = np.asarray(a91, np.uint8)
    parity = (LDPC_GEN.astype(np.int32) @ a91) & 1
    return np.r_[a91, parity.astype(np.uint8)]


def codeword_to_tones(cw):
    cw = np.asarray(cw).reshape(ND, 3)
    vals = cw[:, 0] * 4 + cw[:, 1] * 2 + cw[:, 2]
    tones = np.zeros(NN, np.int64)
    for p in SYNC_POS:
        tones[p : p + 7] = COSTAS
    tones[DATA_POS] = GRAY[vals]
    return tones


def encode77(payload77):
    """77-bit payload -> (codeword174, tones79)."""
    cw = ldpc_encode(add_crc(payload77))
    return cw, codeword_to_tones(cw)


def random_payloads(n, rng):
    """Random 77-bit payloads. Fine for NN training (the decoder never sees text)."""
    return rng.integers(0, 2, size=(n, 77), dtype=np.uint8)


# ---------------------------------------------------------------- waveform

def _gfsk_pulse(bt, t):
    c = np.pi * np.sqrt(2 / np.log(2))
    return 0.5 * (erf(c * bt * (t + 0.5)) - erf(c * bt * (t - 0.5)))


_PULSE = _gfsk_pulse(2.0, (np.arange(3 * NSPS) - 1.5 * NSPS) / NSPS)


def gfsk_phase(tones, f0, fs=FS, nsps=NSPS):
    """Instantaneous phase (radians) of the FT8 GFSK waveform, as in WSJT-X gen_ft8wave.

    Returns len(tones)*nsps samples. f0 is the frequency of tone 0 in Hz.
    """
    tones = np.asarray(tones, np.float64)
    nsym = len(tones)
    pulse = _PULSE if nsps == NSPS else _gfsk_pulse(2.0, (np.arange(3 * nsps) - 1.5 * nsps) / nsps)
    dphi_peak = 2 * np.pi / nsps
    dphi = np.zeros((nsym + 2) * nsps)
    for j in range(nsym):
        dphi[j * nsps : j * nsps + 3 * nsps] += dphi_peak * pulse * tones[j]
    dphi[: 2 * nsps] += dphi_peak * tones[0] * pulse[nsps:]
    dphi[nsym * nsps :] += dphi_peak * tones[-1] * pulse[: 2 * nsps]
    dphi += 2 * np.pi * f0 / fs
    d = dphi[nsps : nsps + nsym * nsps]
    return np.concatenate([[0.0], np.cumsum(d[:-1])])


def ramp(n, nsps=NSPS):
    """Raised-cosine ramp envelope over first/last nsps/8 samples."""
    nr = nsps // 8
    env = np.ones(n)
    r = 0.5 * (1 - np.cos(np.pi * np.arange(nr) / nr))
    env[:nr] = r
    env[-nr:] = r[::-1]
    return env


def complex_wave(tones, f0, fs=FS, nsps=NSPS):
    """Unit-amplitude analytic FT8 signal (complex), with ramps."""
    ph = gfsk_phase(tones, f0, fs, nsps)
    return ramp(len(ph), nsps) * np.exp(1j * ph)
