import numpy as np

from ft8nn import ft8, ldpc


def _noisy_llrs(n, ebn0_db, rng):
    """BPSK over AWGN at given Eb/N0 (rate 91/174), exact LLRs."""
    p = ft8.random_payloads(n, rng)
    cws = np.stack([ft8.encode77(x)[0] for x in p])
    rate = 91 / 174
    sigma = np.sqrt(1 / (2 * rate * 10 ** (ebn0_db / 10)))
    y = (2.0 * cws - 1) + sigma * rng.standard_normal(cws.shape)
    return cws, 2 * y / sigma**2


def test_bp_noiseless():
    rng = np.random.default_rng(0)
    cws, llr = _noisy_llrs(20, 20, rng)
    bits, ok, nhard = ldpc.bp_decode(llr)
    assert ok.all() and (bits == cws).all() and (nhard == 0).all()


def test_bp_and_osd_correct_errors():
    rng = np.random.default_rng(1)
    cws, llr = _noisy_llrs(200, 2.0, rng)
    bits, ok, _ = ldpc.bp_decode(llr)
    bp_rate = (ok & (bits == cws).all(1)).mean()
    res = ldpc.decode(llr)
    full_rate = np.mean([r is not None and (r == c[:91]).all() for (r, _), c in zip(res, cws)])
    print(f"BP {bp_rate:.2f}  BP+OSD {full_rate:.2f}")
    assert bp_rate > 0.5
    assert full_rate >= bp_rate


def test_osd_order0_recovers_clean():
    rng = np.random.default_rng(2)
    cws, llr = _noisy_llrs(3, 10, rng)
    for c, l in zip(cws, llr):
        cw, crc, dist, nh = ldpc.osd_decode(l, order=0)
        assert crc and (cw == c).all()
