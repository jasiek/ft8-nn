/* Subtract a decoded FT8 signal from the slot audio (port of ft8nn/subtract.py).
 *
 * The complex envelope a(t) = LPF{x(t) conj(ref(t))} / LPF{|ref|^2} is tracked with a ~0.33 s
 * sin^2 window, and 2 Re{a(t) ref(t)} is removed. Because sin^2(pi k / 4000) = (1 - cos(2 pi k / 4000)) / 2
 * (or centred at u = k - 2000: (1 + cos(2 pi u / 4000)) / 2), the exact 'same'-mode FIR convolution is
 * three O(N) sliding boxcar sums (1, cos(w j), sin(w j)) with periodic table lookup and zero FFTs.
 * LPF{|ref|^2} depends only on the ramp envelope, so it is computed once.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

#define NFILT 4001
#define HFILT ((NFILT - 1) / 2) /* 2000 */
#define MFILT (NFILT - 1)       /* 4000 */
#define NREF (NN * NSPS)        /* 151680 */
#define BLK 960

static double ctab[MFILT], stab[MFILT];
static double *refpow_lp; /* max(LPF{|ref|^2}, 1e-3), 'same' alignment */

/* Exact 'same'-mode convolution of real array sr[0..n) with w[k] = sin^2(pi k / MFILT) / (MFILT / 2) */
static void conv_same_real(const double *restrict sr, int n, double *restrict ar) {
    const double scale = 1.0 / (double)MFILT;
    double *uc = malloc(sizeof(double) * (size_t)n);
    double *us = malloc(sizeof(double) * (size_t)n);
    for (int j = 0; j < n; j++) {
        double c = ctab[j % MFILT], s = stab[j % MFILT];
        uc[j] = sr[j] * c;
        us[j] = sr[j] * s;
    }
    double s0 = 0.0, ac = 0.0, as_ = 0.0;
    for (int j = 0; j < HFILT && j < n; j++) {
        s0 += sr[j];
        ac += uc[j];
        as_ += us[j];
    }
    for (int i = 0; i < n; i++) {
        ar[i] = (s0 + ctab[i % MFILT] * ac + stab[i % MFILT] * as_) * scale;
        int ji = i + HFILT, jo = i - HFILT;
        if (ji < n) {
            s0 += sr[ji];
            ac += uc[ji];
            as_ += us[ji];
        }
        if (jo >= 0) {
            s0 -= sr[jo];
            ac -= uc[jo];
            as_ -= us[jo];
        }
    }
    free(uc);
    free(us);
}

/* Exact 'same'-mode convolution of split complex (sr, si)[0..n) */
static void conv_same_cplx(const double *restrict sr, const double *restrict si, int n,
                           double *restrict ar, double *restrict ai) {
    const double scale = 1.0 / (double)MFILT;
    double *ucr = malloc(sizeof(double) * (size_t)n), *uci = malloc(sizeof(double) * (size_t)n);
    double *usr = malloc(sizeof(double) * (size_t)n), *usi = malloc(sizeof(double) * (size_t)n);
    for (int j = 0; j < n; j++) {
        double c = ctab[j % MFILT], s = stab[j % MFILT];
        ucr[j] = sr[j] * c;
        uci[j] = si[j] * c;
        usr[j] = sr[j] * s;
        usi[j] = si[j] * s;
    }
    double s0r = 0.0, s0i = 0.0, acr = 0.0, aci = 0.0, asr = 0.0, asi = 0.0;
    for (int j = 0; j < HFILT && j < n; j++) {
        s0r += sr[j];
        s0i += si[j];
        acr += ucr[j];
        aci += uci[j];
        asr += usr[j];
        asi += usi[j];
    }
    for (int i = 0; i < n; i++) {
        double c = ctab[i % MFILT], s = stab[i % MFILT];
        ar[i] = (s0r + c * acr + s * asr) * scale;
        ai[i] = (s0i + c * aci + s * asi) * scale;
        int ji = i + HFILT, jo = i - HFILT;
        if (ji < n) {
            s0r += sr[ji];
            s0i += si[ji];
            acr += ucr[ji];
            aci += uci[ji];
            asr += usr[ji];
            asi += usi[ji];
        }
        if (jo >= 0) {
            s0r -= sr[jo];
            s0i -= si[jo];
            acr -= ucr[jo];
            aci -= uci[jo];
            asr -= usr[jo];
            asi -= usi[jo];
        }
    }
    free(ucr);
    free(uci);
    free(usr);
    free(usi);
}

void subtract_init(void) {
    for (int m = 0; m < MFILT; m++) {
        double a = 2.0 * M_PI * m / MFILT;
        ctab[m] = cos(a);
        stab[m] = sin(a);
    }
    /* |ref|^2 is the squared ramp envelope, independent of tones and f0 */
    int nr = NSPS / 8;
    double *rp = malloc(sizeof(double) * NREF), *lp = malloc(sizeof(double) * NREF);
    for (int i = 0; i < NREF; i++) {
        double env = 1.0;
        if (i < nr) env = 0.5 * (1.0 - cos(M_PI * i / nr));
        else if (i >= NREF - nr) env = 0.5 * (1.0 - cos(M_PI * (NREF - 1 - i) / nr));
        rp[i] = env * env;
    }
    conv_same_real(rp, NREF, lp);
    refpow_lp = malloc(sizeof(double) * NREF);
    for (int i = 0; i < NREF; i++) refpow_lp[i] = fmax(lp[i], 1e-3);
    free(rp);
    free(lp);
}

static double block_energy(const double *restrict x, int n, const double *restrict refr, const double *restrict refi, int k0) {
    int nb = NREF / BLK * BLK;
    int lo = k0 > 0 ? k0 : 0, hi = k0 + nb < n ? k0 + nb : n;
    if (hi - lo < BLK) return 0.0;
    double e = 0.0;
    const double *xp = x + k0;
    for (int b = 0; b < nb / BLK; b++) {
        int i0 = b * BLK;
        int a = lo - k0 > i0 ? lo - k0 : i0;
        int z = hi - k0 < i0 + BLK ? hi - k0 : i0 + BLK;
        double sr = 0.0, si = 0.0;
        for (int i = a; i < z; i++) {
            double xv = xp[i];
            sr += xv * refr[i];
            si -= xv * refi[i];
        }
        e += sr * sr + si * si;
    }
    return e;
}

/* block energies at offsets d0, d0 + step, ... (independent sums: worked out in parallel) */
typedef struct {
    const double *x;
    int n, k0, d0, step, skip_idx;
    const double *refr, *refi;
    double *e;
} align_job;
static void align_item(void *a, int i) {
    align_job *j = a;
    if (i == j->skip_idx) return;
    j->e[i] = block_energy(j->x, j->n, j->refr, j->refi, j->k0 + j->d0 + i * j->step);
}

void subtract_signal(double *x, int n, const int *tones, double f0, double dt, par *pool) {
    double *refr = malloc(sizeof(double) * NREF), *refi = malloc(sizeof(double) * NREF);
    complex_wave_split(tones, f0, refr, refi);
    int k0 = (int)nearbyint((NOMINAL_START + dt) * FS);
    /* refine alignment: coarse +-48 in steps of 4, then +-3 around the best (first max wins) */
    double e[25];
    align_job aj = {x, n, k0, -48, 4, -1, refr, refi, e};
    par_for(pool, 25, align_item, &aj);
    int coarse = -48;
    double best = -1.0;
    for (int i = 0; i < 25; i++)
        if (e[i] > best) {
            best = e[i];
            coarse = -48 + 4 * i;
        }
    double e_coarse = best;
    aj.d0 = coarse - 3;
    aj.step = 1;
    aj.skip_idx = 3; /* i = 3 is d = coarse, already in e_coarse */
    e[3] = e_coarse;
    par_for(pool, 7, align_item, &aj);
    int fine = coarse - 3;
    best = -1.0;
    for (int i = 0; i < 7; i++)
        if (e[i] > best) {
            best = e[i];
            fine = coarse - 3 + i;
        }
    k0 += fine;
    double *sr = malloc(sizeof(double) * NREF), *si = malloc(sizeof(double) * NREF);
    double *ar = malloc(sizeof(double) * NREF), *ai = malloc(sizeof(double) * NREF);
    for (int i = 0; i < NREF; i++) {
        int j = k0 + i;
        double xv = (j >= 0 && j < n) ? x[j] : 0.0;
        sr[i] = xv * refr[i];
        si[i] = -xv * refi[i];
    }
    conv_same_cplx(sr, si, NREF, ar, ai);
    int lo = k0 > 0 ? k0 : 0, hi = k0 + NREF < n ? k0 + NREF : n;
    for (int j = lo; j < hi; j++) {
        int i = j - k0;
        x[j] -= 2.0 * (ar[i] * refr[i] - ai[i] * refi[i]) / refpow_lp[i];
    }
    free(refr);
    free(refi);
    free(sr);
    free(si);
    free(ar);
    free(ai);
}
