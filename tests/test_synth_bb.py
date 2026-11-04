"""Training-data generator must match the real 12 kHz -> 200 Hz pipeline (power calibration + waveform)."""
import numpy as np
import torch

from ft8nn import frontend as fe
from ft8nn import ft8, synth_bb
from ft8nn.channel import Signal, synth_slot


def _tone_bin_power(y, tones, start=100):
    Z = fe.symbol_spectra(y, start)
    return np.mean(np.abs(Z[np.arange(79), tones]) ** 2)


def test_gfsk_matches_reference():
    tones = ft8.codeword_to_tones(ft8.ldpc_encode(ft8.add_crc(np.random.default_rng(0).integers(0, 2, 77))))
    ref = ft8.complex_wave(tones, 1.3, fs=200, nsps=32)
    got = synth_bb.gfsk_baseband(torch.tensor(tones)[None], torch.tensor([1.3]), torch.tensor([0.0]), "cpu")[0].numpy()
    assert np.allclose(got, ref, atol=2e-3)


def test_power_calibration_matches_real_pipeline():
    rng = np.random.default_rng(1)
    tones = ft8.codeword_to_tones(ft8.ldpc_encode(ft8.add_crc(rng.integers(0, 2, 77))))
    # real pipeline: signal only, and noise only
    sig_real = fe.Baseband(synth_slot([Signal(tones, 1000.0, 0.0, -10.0)], rng, noise=False) + 1e-9 * rng.standard_normal(ft8.SLOT_SAMPLES))
    bb_noise = fe.Baseband(synth_slot([], rng))
    bb_sig = fe.Baseband(synth_slot([Signal(tones, 1000.0, 0.0, -10.0)], rng, noise=False))
    bb_sig.noise_psd = bb_noise.noise_psd  # normalise by the real noise level
    p_sig_real = _tone_bin_power(bb_sig.downsample(1000.0), tones)
    p_noise_real = np.mean(np.abs(fe.symbol_spectra(bb_noise.downsample(1000.0), 100)[:, :8]) ** 2)

    cfg = synth_bb.ChannelCfg(df_sigma=0, df_wide_p=0, dt_sigma=0, drift_p=0, fade_p=0, qrm_p=0)
    torch.manual_seed(0)
    y, _, _, t = synth_bb.make_batch(64, cfg, "cpu", snr=-10.0)
    p_tot = np.mean([_tone_bin_power(y[i].numpy(), t[i].numpy()) for i in range(64)])
    y0, _, _, _ = synth_bb.make_batch(64, cfg, "cpu", snr=-200.0)
    p_noise = np.mean(np.abs(np.stack([fe.symbol_spectra(y0[i].numpy(), 100)[:, :8] for i in range(64)])) ** 2)
    p_sig = p_tot - p_noise
    print(f"signal real {p_sig_real:.2f} synth {p_sig:.2f} | noise real {p_noise_real:.3f} synth {p_noise:.3f}")
    snr_err_db = 10 * np.log10((p_sig / p_noise) / (p_sig_real / p_noise_real))
    print(f"SNR calibration error {snr_err_db:+.3f} dB")
    assert abs(snr_err_db) < 0.15
