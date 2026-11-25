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


def test_osd_order3_recovers_weight3_and_matches_unpruned():
    rng = np.random.default_rng(3)
    p = ft8.random_payloads(1, rng)[0]
    cw_true = ft8.encode77(p)[0]
    # Construct LLRs where the 3 least-reliable basis bits have wrong sign (weight-3 error in basis -> needs OSD-3)
    rel = np.linspace(3.0, 0.2, 174)
    # Use a fixed permutation order: assign high reliability to first 91, lowest 3 of those flipped
    llr = np.where(cw_true == 1, rel, -rel)
    perm = np.argsort(-rel, kind="stable")
    _, piv = ldpc._gf2_reduce(ldpc._G[:, perm])
    for idx in perm[piv[-3:]]:
        llr[idx] = -llr[idx]
    cw2, crc2, _, _ = ldpc.osd_decode(llr, order=2, k2=60)
    cw3, crc3, d3, _ = ldpc.osd_decode(llr, order=3, k2=60, k3=28)
    assert not (crc2 and (cw2 == cw_true).all())
    assert crc3 and (cw3 == cw_true).all()

    # Verify pruned OSD-3 matches exhaustive unpruned argmin on noisy LLRs
    cws, llrs = _noisy_llrs(10, 1.0, rng)
    for l in llrs:
        cw_p, crc_p, d_p, nh_p = ldpc.osd_decode(l, order=3, k2=20, k3=10)
        rel = np.abs(l)
        hard = (l > 0).astype(np.uint8)
        perm = np.argsort(-rel, kind="stable")
        Gr, piv = ldpc._gf2_reduce(ldpc._G[:, perm])
        hp, rp = hard[perm], rel[perm]
        c0 = (hp[piv].astype(np.int32) @ Gr.astype(np.int32)) & 1
        gi, gj = ldpc._osd_pairs(20)
        ga, gb, gc = ldpc._osd_trips(10)
        C = np.concatenate([
            c0[None, :],
            c0[None, :] ^ Gr,
            c0[None, :] ^ Gr[gi] ^ Gr[gj],
            c0[None, :] ^ Gr[ga] ^ Gr[gb] ^ Gr[gc],
        ]).astype(np.uint8)
        dists = ((C != hp[None, :]) * rp[None, :]).sum(-1)
        best = int(np.argmin(dists))
        cw_e = np.empty(174, np.uint8)
        cw_e[perm] = C[best]
        assert (cw_p == cw_e).all() and crc_p == ft8.crc_ok(cw_e[:91]) and abs(d_p - float(dists[best])) < 1e-12 and nh_p == int((cw_e != hard).sum())
