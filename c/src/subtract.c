/* Subtract a decoded FT8 signal from the slot audio (port of ft8nn/subtract.py).
 *
 * The complex envelope a(t) = LPF{x(t) conj(ref(t))} / LPF{|ref|^2} is tracked with a ~0.33 s
 * sin^2 window, and 2 Re{a(t) ref(t)} is removed. The two 'same'-mode convolutions run as FFT
 * convolutions of size NCONV; LPF{|ref|^2} depends only on the ramp envelope, so it is computed once.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

#define NFILT 4001
#define NREF (NN * NSPS)  /* 151680 */
#define NCONV 160000      /* >= NREF + NFILT - 1, factors 2^8 5^4 */
#define BLK 960

static fft_plan *conv_plan;
static cplx *win_f;    /* FFT of the window, zero padded to NCONV */
static double *refpow_lp; /* max(LPF{|ref|^2}, 1e-3), 'same' alignment */

/* 'same' convolution with the window: out[i] = full[i + (NFILT-1)/2] */
static void conv_same(const cplx *in, int n, cplx *out, cplx *buf1, cplx *buf2) {
    memset(buf1, 0, sizeof(cplx) * NCONV);
    memcpy(buf1, in, sizeof(cplx) * n);
    fft_forward(conv_plan, buf1, buf2);
    for (int k = 0; k < NCONV; k++) buf2[k] *= win_f[k];
    fft_inverse(conv_plan, buf2, buf1);
    for (int i = 0; i < n; i++) out[i] = buf1[i + (NFILT - 1) / 2] / NCONV;
}

void subtract_init(void) {
    conv_plan = fft_plan_new(NCONV);
    cplx *w = calloc(NCONV, sizeof(cplx)), *b1 = malloc(sizeof(cplx) * NCONV), *b2 = malloc(sizeof(cplx) * NCONV);
    double sum = 0;
    for (int i = 0; i < NFILT; i++) {
        double s = sin(M_PI * i / (NFILT - 1));
        sum += s * s;
    }
    for (int i = 0; i < NFILT; i++) {
        double s = sin(M_PI * i / (NFILT - 1));
        w[i] = s * s / sum;
    }
    win_f = malloc(sizeof(cplx) * NCONV);
    fft_forward(conv_plan, w, win_f);
    /* |ref|^2 is the squared ramp envelope, independent of tones and f0 */
    int nr = NSPS / 8;
    cplx *rp = malloc(sizeof(cplx) * NREF), *lp = malloc(sizeof(cplx) * NREF);
    for (int i = 0; i < NREF; i++) {
        double env = 1.0;
        if (i < nr) env = 0.5 * (1 - cos(M_PI * i / nr));
        else if (i >= NREF - nr) env = 0.5 * (1 - cos(M_PI * (NREF - 1 - i) / nr));
        rp[i] = env * env;
    }
    conv_same(rp, NREF, lp, b1, b2);
    refpow_lp = malloc(sizeof(double) * NREF);
    for (int i = 0; i < NREF; i++) refpow_lp[i] = fmax(creal(lp[i]), 1e-3);
    free(w);
    free(b1);
    free(b2);
    free(rp);
    free(lp);
}

static double block_energy(const double *x, int n, const cplx *ref, int k0) {
    int nb = NREF / BLK * BLK;
    int lo = k0 > 0 ? k0 : 0, hi = k0 + nb < n ? k0 + nb : n;
    if (hi - lo < BLK) return 0.0;
    double e = 0;
    for (int b = 0; b < nb / BLK; b++) {
        int i0 = b * BLK, a = lo - k0 > i0 ? lo - k0 : i0, z = hi - k0 < i0 + BLK ? hi - k0 : i0 + BLK;
        cplx s = 0;
        for (int i = a; i < z; i++) s += x[k0 + i] * conj(ref[i]);
        double m = cabs(s);
        e += m * m;
    }
    return e;
}

void subtract_signal(double *x, int n, const int *tones, double f0, double dt) {
    cplx *ref = malloc(sizeof(cplx) * NREF);
    complex_wave(tones, f0, ref);
    int k0 = (int)nearbyint((NOMINAL_START + dt) * FS);
    /* refine alignment: coarse +-48 in steps of 4, then +-3 around the best (first max wins) */
    int coarse = -48;
    double best = -1;
    for (int d = -48; d <= 48; d += 4) {
        double e = block_energy(x, n, ref, k0 + d);
        if (e > best) {
            best = e;
            coarse = d;
        }
    }
    int fine = coarse - 3;
    best = -1;
    for (int d = coarse - 3; d <= coarse + 3; d++) {
        double e = block_energy(x, n, ref, k0 + d);
        if (e > best) {
            best = e;
            fine = d;
        }
    }
    k0 += fine;
    cplx *seg = malloc(sizeof(cplx) * NREF), *a = malloc(sizeof(cplx) * NREF);
    cplx *b1 = malloc(sizeof(cplx) * NCONV), *b2 = malloc(sizeof(cplx) * NCONV);
    for (int i = 0; i < NREF; i++) {
        int j = k0 + i;
        seg[i] = (j >= 0 && j < n) ? x[j] * conj(ref[i]) : 0;
    }
    conv_same(seg, NREF, a, b1, b2);
    int lo = k0 > 0 ? k0 : 0, hi = k0 + NREF < n ? k0 + NREF : n;
    for (int j = lo; j < hi; j++) {
        int i = j - k0;
        x[j] -= 2 * creal(a[i] / refpow_lp[i] * ref[i]);
    }
    free(ref);
    free(seg);
    free(a);
    free(b1);
    free(b2);
}
