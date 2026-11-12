/* FT8 receive front end (port of ft8nn/frontend.py and model.features). */
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

#define HOP 480
#define NFFT_SPEC 3840
#define FPS 25.0
#define SYM_FRAMES 4

static inline double fmax_d(double a, double b) { return a > b ? a : b; }

static int cmp_f(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}
static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* median of 7 floats (odd count: exact middle element) */
static float median7(float *v) {
    for (int i = 1; i < 7; i++) { /* insertion sort */
        float t = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > t) {
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = t;
    }
    return v[3];
}

/* round half to even, as numpy / Python 3 round() */
static inline double rnd(double x) { return nearbyint(x); }

typedef struct {
    double s;
    int f, j;
} cand_ent;

static int cmp_cand(const void *a, const void *b) { /* descending by (score, f, j) */
    const cand_ent *x = a, *y = b;
    if (x->s != y->s) return x->s > y->s ? -1 : 1;
    if (x->f != y->f) return y->f - x->f;
    return y->j - x->j;
}

/* spectrogram frame t -> S[t*nb ..] */
typedef struct {
    const double *x;
    const fft_plan *fp;
    float *S;
    int nb;
} spec_job;
static void spec_item(void *a, int t) {
    spec_job *j = a;
    cplx *in = calloc(NFFT_SPEC, sizeof(cplx)), *X = malloc(sizeof(cplx) * NFFT_SPEC);
    for (int i = 0; i < NSPS; i++) in[i] = j->x[t * HOP + i];
    fft_forward(j->fp, in, X);
    for (int b = 0; b < j->nb; b++) {
        double re = creal(X[b]), im = cimag(X[b]);
        j->S[t * j->nb + b] = (float)(re * re + im * im);
    }
    free(in);
    free(X);
}

/* Costas score sums of lag row li: per cell, the same additions in the same order (array o, then symbol k) */
typedef struct {
    const float *S;
    int T, nb, b0, nf, lag0, lmin;
    double *num, *den;
} score_job;
static void score_item(void *a, int li) {
    static const int offs[3] = {0, 36, 72};
    score_job *j = a;
    double *nu = j->num + (size_t)li * j->nf, *de = j->den + (size_t)li * j->nf;
    for (int o = 0; o < 3; o++)
        for (int k = 0; k < 7; k++) {
            int c = COSTAS[k];
            int fr = j->lag0 + j->lmin + li + SYM_FRAMES * (offs[o] + k);
            if (fr < 0 || fr >= j->T) continue;
            const float *row = j->S + (size_t)fr * j->nb + j->b0;
            for (int f = 0; f < j->nf; f++) {
                float oth[7];
                int q = 0;
                for (int tn = 0; tn < 8; tn++)
                    if (tn != c) oth[q++] = row[f + 2 * tn];
                nu[f] += row[f + 2 * c];
                de[f] += median7(oth);
            }
        }
}

int find_candidates(const double *x, int n, double fmin, double fmax, int max_cand, double min_score, int base_median,
                    double early_dt, double early_min, double dt_shift, candidate *out, par *pool) {
    const double dtmin = -1.5, dtmax = 2.5, typ_lo = -0.6, typ_hi = 1.2;
    const int lag_excl = 10;
    /* dt_shift (frontend.find_candidates dt_shift) moves the whole window - main range, typical-DT track and
     * early band - by whole frames: a receiver whose clock or audio stream is off puts every signal at the
     * same wrong DT, and the window is only 4 s wide (decode() estimates the shift from its own decodes) */
    const int sh = (int)rnd(dt_shift * FPS);
    int T = (n - NSPS) / HOP + 1;
    if (T <= 0) return 0;
    int b0 = (int)(fmin / 3.125), b1 = (int)(fmax / 3.125), nf = b1 - b0;
    int nb = b0 + 14 + nf; /* highest bin needed + 1 */
    if (nb > NFFT_SPEC / 2 + 1) return 0;
    float *S = malloc(sizeof(float) * T * nb);
    fft_plan *fp = fft_plan_new(NFFT_SPEC);
    spec_job sj = {x, fp, S, nb};
    par_for(pool, T, spec_item, &sj);
    fft_plan_free(fp);

    int lag0 = (int)rnd(NOMINAL_START * FPS);
    /* main lags from dtmin, plus n_early "early" lags [early_dt, dtmin) that only feed a separate track
     * (as frontend.find_candidates: kept out of the baseline and the main tracks) */
    int n_early = early_dt < dtmin ? (int)floor(dtmin * FPS) - (int)floor(early_dt * FPS) : 0;
    if (n_early < 0) n_early = 0;
    int lmin = (int)floor(dtmin * FPS) + sh, lmax = (int)ceil(dtmax * FPS) + sh, nl = lmax - lmin + 1;
    lmin -= n_early;
    nl += n_early;
    double *num = calloc((size_t)nl * nf, sizeof(double)), *den = calloc((size_t)nl * nf, sizeof(double));
    score_job scj = {S, T, nb, b0, nf, lag0, lmin, num, den};
    par_for(pool, nl, score_item, &scj);
    free(S);
    double *score_all = num; /* in place */
    for (size_t i = 0; i < (size_t)nl * nf; i++) score_all[i] = num[i] / fmax_d(den[i], 1e-12);
    free(den);
    double *early = score_all, *score = score_all + (size_t)n_early * nf; /* split: early rows | main rows */
    lmin += n_early;
    nl -= n_early;
    double base;
    if (base_median) {
        /* 1.77 * median over all (lag, freq) cells: stays a noise statistic on crowded bands */
        size_t nall = (size_t)nl * nf;
        double *cp = malloc(sizeof(double) * nall);
        memcpy(cp, score, sizeof(double) * nall);
        qsort(cp, nall, sizeof(double), cmp_d);
        double med = nall % 2 ? cp[nall / 2] : (cp[nall / 2 - 1] + cp[nall / 2]) / 2;
        free(cp);
        base = 1.77 * med;
    } else {
        /* 40th percentile (linear interpolation) of the per-frequency max */
        double *mx = malloc(sizeof(double) * nf);
        for (int f = 0; f < nf; f++) {
            double m = -INFINITY;
            for (int li = 0; li < nl; li++) m = fmax_d(m, score[(size_t)li * nf + f]);
            mx[f] = m;
        }
        qsort(mx, nf, sizeof(double), cmp_d);
        double vi = 0.4 * (nf - 1);
        int lo = (int)floor(vi);
        double tt = vi - lo, a = mx[lo], bb = mx[lo + 1 < nf ? lo + 1 : lo];
        base = tt < 0.5 ? a + (bb - a) * tt : bb - (bb - a) * (1 - tt);
        free(mx);
    }
    double inv = fmax_d(base, 1e-9);
    for (size_t i = 0; i < (size_t)nl * nf; i++) score[i] /= inv;
    for (size_t i = 0; i < (size_t)n_early * nf; i++) early[i] /= inv;

    /* three lag tracks per frequency: best, second best (>lag_excl away), best within typical DT */
    int *jt[3];
    double *st[3];
    for (int k = 0; k < 3; k++) {
        jt[k] = malloc(sizeof(int) * nf);
        st[k] = malloc(sizeof(double) * nf);
    }
    for (int f = 0; f < nf; f++) {
        int j1 = 0;
        for (int li = 1; li < nl; li++)
            if (score[(size_t)li * nf + f] > score[(size_t)j1 * nf + f]) j1 = li;
        int j2 = -1, j3 = -1;
        double s2 = -INFINITY, s3 = -INFINITY;
        for (int li = 0; li < nl; li++) {
            double s = score[(size_t)li * nf + f];
            if (abs(li - j1) > lag_excl && (j2 < 0 || s > s2)) {
                j2 = li;
                s2 = s;
            }
            double dtl = (double)(lmin + li - sh) / FPS;
            if (dtl >= typ_lo && dtl <= typ_hi && (j3 < 0 || s > s3)) {
                j3 = li;
                s3 = s;
            }
        }
        jt[0][f] = j1;
        st[0][f] = score[(size_t)j1 * nf + f];
        jt[1][f] = j2 < 0 ? 0 : j2;
        st[1][f] = s2;
        jt[2][f] = j3 < 0 ? 0 : j3;
        st[2][f] = s3;
    }
    cand_ent *ents = malloc(sizeof(cand_ent) * 3 * nf);
    int ne = 0;
    uint8_t *seen = calloc((size_t)nf * nl, 1);
    for (int k = 0; k < 3; k++) {
        double *s = st[k];
        for (int f = 0; f < nf; f++) {
            double l = f > 0 ? s[f - 1] : -INFINITY, r = f + 1 < nf ? s[f + 1] : -INFINITY;
            if (!(s[f] >= l && s[f] >= r && s[f] >= min_score)) continue;
            size_t key = (size_t)f * nl + jt[k][f];
            if (seen[key]) continue;
            seen[key] = 1;
            ents[ne++] = (cand_ent){s[f], f, jt[k][f]};
        }
    }
    free(seen);
    qsort(ents, ne, sizeof(cand_ent), cmp_cand);
    int nc = ne < max_cand ? ne : max_cand;
    for (int i = 0; i < nc; i++) {
        out[i].f0 = (b0 + ents[i].f) * 3.125;
        out[i].dt = (double)(lmin + ents[i].j) / FPS;
        out[i].score = ents[i].s;
        out[i].early = 0;
    }
    if (n_early) {
        /* early track, appended after the main budget; a peak must beat every main lag at its frequency
         * (a strong on-time signal also scores high at early lags, but higher at its true lag) */
        double *s4 = malloc(sizeof(double) * nf);
        int *j4 = malloc(sizeof(int) * nf);
        for (int f = 0; f < nf; f++) {
            int jb = 0;
            for (int li = 1; li < n_early; li++)
                if (early[(size_t)li * nf + f] > early[(size_t)jb * nf + f]) jb = li;
            j4[f] = jb;
            s4[f] = early[(size_t)jb * nf + f];
        }
        int ne4 = 0;
        for (int f = 0; f < nf; f++) {
            double l = f > 0 ? s4[f - 1] : -INFINITY, r = f + 1 < nf ? s4[f + 1] : -INFINITY;
            if (s4[f] >= l && s4[f] >= r && s4[f] >= early_min && s4[f] > st[0][f]) ents[ne4++] = (cand_ent){s4[f], f, j4[f]};
        }
        qsort(ents, ne4, sizeof(cand_ent), cmp_cand);
        for (int i = 0; i < ne4; i++, nc++) {
            out[nc].f0 = (b0 + ents[i].f) * 3.125;
            out[nc].dt = (double)(lmin - n_early + ents[i].j) / FPS;
            out[nc].score = ents[i].s;
            out[nc].early = 1;
        }
        free(s4);
        free(j4);
    }
    for (int k = 0; k < 3; k++) {
        free(jt[k]);
        free(st[k]);
    }
    free(ents);
    free(score_all);
    return nc; /* at most max_cand + nf */
}

/* ------------------------------------------------------------------ baseband */

#define BB_LO (-150) /* -round(1.5 * 6.25 / DF_FULL) */
#define BB_HI 850    /*  round(8.5 * 6.25 / DF_FULL) */
#define TAPER_N 101

void baseband_init(baseband *b, const double *x, int n, par *pool) {
    b->full = fft_plan_new(NFFT_FULL);
    b->bb = fft_plan_new(NBB);
    cplx *in = calloc(NFFT_FULL, sizeof(cplx));
    b->tmp = malloc(sizeof(cplx) * NFFT_FULL);
    int m = n < NFFT_FULL ? n : NFFT_FULL;
    for (int i = 0; i < m; i++) in[i] = x[i];
    fft_forward_par(b->full, in, b->tmp, pool);
    free(in);
    int nx = NFFT_FULL / 2 + 1;
    b->X = malloc(sizeof(cplx) * nx);
    memcpy(b->X, b->tmp, sizeof(cplx) * nx);
    int lo = (int)(200 / DF_FULL), hi = (int)(3000 / DF_FULL), nb = hi - lo;
    double *band = malloc(sizeof(double) * nb);
    for (int i = 0; i < nb; i++) {
        double a = cabs(b->X[lo + i]);
        band[i] = a * a;
    }
    qsort(band, nb, sizeof(double), cmp_d);
    double med = nb % 2 ? band[nb / 2] : (band[nb / 2 - 1] + band[nb / 2]) / 2;
    free(band);
    b->noise_psd = med / log(2.0);
}

void baseband_free(baseband *b) {
    fft_plan_free(b->full);
    fft_plan_free(b->bb);
    free(b->X);
    free(b->tmp);
}

static double taper[TAPER_N];

void baseband_downsample(const baseband *b, double f0, cplx *y) {
    cplx Y[NBB];
    memset(Y, 0, sizeof Y);
    int i0 = (int)rnd(f0 / DF_FULL), nk = BB_HI - BB_LO, nx = NFFT_FULL / 2 + 1;
    for (int q = 0; q < nk; q++) {
        int k = BB_LO + q, src = i0 + k;
        if (src < 0 || src >= nx) continue;
        double w = q < TAPER_N ? taper[q] : (q >= nk - TAPER_N ? taper[nk - 1 - q] : 1.0);
        Y[((k % NBB) + NBB) % NBB] = b->X[src] * w;
    }
    fft_inverse(b->bb, Y, y);
    double sc = 1.0 / sqrt(b->noise_psd * nk);
    for (int i = 0; i < NBB; i++) y[i] *= sc;
}

/* ------------------------------------------------------------------ fine sync */

static void spectra_init(void);

#define NDF 51  /* np.arange(-2.5, 2.5 + 1e-9, 0.1) */
#define NDT 25  /* -12..12 */
#define DT_SPAN 12

static double dfs[NDF];
static double gridKr[21][SPS_BB][NDF], gridKi[21][SPS_BB][NDF]; /* split re/im so the fi loop vectorises */

void frontend_init(void) {
    /* numpy arange fills start + i * delta with delta = (start + step) - start */
    double start = -2.5, delta = (start + 0.1) - start;
    for (int i = 0; i < NDF; i++) dfs[i] = start + i * delta;
    for (int s = 0; s < 21; s++)
        for (int n = 0; n < SPS_BB; n++)
            for (int fi = 0; fi < NDF; fi++) {
                double f = COSTAS[s % 7] * TONE_SPACING + dfs[fi];
                double ph = -2 * M_PI * (f * n / FS_BB + dfs[fi] * SPS_BB * SYNC_SYMS[s] / FS_BB);
                gridKr[s][n][fi] = cos(ph);
                gridKi[s][n][fi] = sin(ph);
            }
    for (int i = 0; i < TAPER_N; i++) taper[i] = 0.5 * (1 - cos(M_PI * i / (TAPER_N - 1)));
    spectra_init();
}

static double parabolic(const double *m, int len, int stride, int i) {
    if (i <= 0 || i >= len - 1) return 0.0;
    double a = log(fmax_d(m[(i - 1) * stride], 1e-30)), b = log(fmax_d(m[i * stride], 1e-30)),
           c = log(fmax_d(m[(i + 1) * stride], 1e-30));
    double den = a - 2 * b + c;
    if (!(den < 0)) return 0.0;
    double o = 0.5 * (a - c) / den;
    return o < -0.5 ? -0.5 : (o > 0.5 ? 0.5 : o);
}

static void rotate(cplx *y, double df) {
    for (int n = 0; n < NBB; n++) {
        double a = -2 * M_PI * df * n / FS_BB;
        y[n] *= cos(a) + I * sin(a);
    }
}

void bb_rotate(cplx *y, double df) { rotate(y, df); }

int fine_sync2(const baseband *b, const candidate *c, int n_hyp, sync_hyp *out) {
    cplx y0[NBB];
    baseband_downsample(b, c->f0, y0);
    int start0 = (int)rnd((NOMINAL_START + c->dt) * FS_BB);
    double Mp[NDF][NDT];
    for (int ti = 0; ti < NDT; ti++) {
        int dt = ti - DT_SPAN;
        double acc[NDF];
        memset(acc, 0, sizeof acc);
        for (int a = 0; a < 3; a++) {
            double zr[NDF], zi[NDF];
            memset(zr, 0, sizeof zr);
            memset(zi, 0, sizeof zi);
            for (int k = 0; k < 7; k++) {
                int s = 7 * a + k, base = start0 + dt + SPS_BB * SYNC_SYMS[s];
                for (int n = 0; n < SPS_BB; n++) {
                    int idx = base + n;
                    if (idx < 0 || idx >= NBB) continue;
                    double vr = creal(y0[idx]), vi = cimag(y0[idx]);
                    const double *Kr = gridKr[s][n], *Ki = gridKi[s][n];
                    for (int fi = 0; fi < NDF; fi++) {
                        zr[fi] += vr * Kr[fi] - vi * Ki[fi];
                        zi[fi] += vr * Ki[fi] + vi * Kr[fi];
                    }
                }
            }
            for (int fi = 0; fi < NDF; fi++) acc[fi] += zr[fi] * zr[fi] + zi[fi] * zi[fi];
        }
        for (int fi = 0; fi < NDF; fi++) Mp[fi][ti] = acc[fi];
    }
    /* local maxima over a (7, 5) window (scipy maximum_filter, reflect mode == clipped window) */
    int pf[NDF * NDT], pt[NDF * NDT], np_ = 0;
    for (int i = 0; i < NDF; i++)
        for (int j = 0; j < NDT; j++) {
            double v = Mp[i][j];
            int ok = 1;
            for (int di = -3; di <= 3 && ok; di++)
                for (int dj = -2; dj <= 2; dj++) {
                    int ii = i + di, jj = j + dj;
                    if (ii < 0 || ii >= NDF || jj < 0 || jj >= NDT) continue;
                    if (Mp[ii][jj] > v) {
                        ok = 0;
                        break;
                    }
                }
            if (ok) {
                pf[np_] = i;
                pt[np_] = j;
                np_++;
            }
        }
    /* top n_hyp by metric (ties: earlier in row-major order first) */
    int nh = 0;
    uint8_t used[NDF * NDT];
    memset(used, 0, np_);
    for (int h = 0; h < n_hyp && h < np_; h++) {
        int bi = -1;
        for (int q = 0; q < np_; q++)
            if (!used[q] && (bi < 0 || Mp[pf[q]][pt[q]] > Mp[pf[bi]][pt[bi]])) bi = q;
        used[bi] = 1;
        int i = pf[bi], j = pt[bi];
        double df = dfs[i] + parabolic(&Mp[0][j], NDF, NDT, i) * 0.1;
        double dtf = (double)(j - DT_SPAN) + parabolic(&Mp[i][0], NDT, 1, j);
        sync_hyp *o = &out[nh++];
        memcpy(o->y, y0, sizeof y0);
        rotate(o->y, df);
        double r = rnd(dtf), tau = dtf - r;
        if (fabs(tau) >= 1e-6) {
            cplx Y[NBB];
            fft_forward(b->bb, o->y, Y);
            for (int k = 0; k < NBB; k++) {
                int kk = k < NBB / 2 ? k : k - NBB;
                double a = 2 * M_PI * kk * tau / NBB;
                Y[k] *= cos(a) + I * sin(a);
            }
            fft_inverse(b->bb, Y, o->y);
            for (int k = 0; k < NBB; k++) o->y[k] /= NBB;
        }
        o->start = start0 + (int)r;
        o->f0 = c->f0 + df;
        o->dt = (start0 + dtf) / FS_BB - NOMINAL_START;
        o->metric = Mp[i][j];
    }
    return nh;
}

/* ------------------------------------------------------------------ data-symbol sync (frontend.data_sync) */

/* Per (df, dt): sum over the 58 data symbols of (strongest of tones 0..7 - stronger of tones -1 and 8) power, over the
 * summed mean power of tones 0..7; M[fi * nt + ti]. Tone powers come from 32-sample DFTs at 6.25 k + df Hz of y0 from
 * start0 + dts[ti] + 32 * DATA_POS[s] (samples outside y0 count as 0). dts must be ascending and of one parity. The
 * DFTs are slid along y0 one sample at a time (X_{p+1} = e^{iw} (X_p - y_p + y_{p+32} e^{-32iw})) and restarted
 * from the direct sum every 64 samples, so rounding cannot build up; numpy's matrix product gives the same values to
 * ~1e-12 relative. */
void data_sync(const cplx *y0, int start0, const double *dfs, int nf, const int *dts, int nt, double *M) {
    int p0 = start0 + dts[0] + SPS_BB * DATA_POS[0], p1 = start0 + dts[nt - 1] + SPS_BB * DATA_POS[ND - 1];
    int np = (p1 - p0) / 2 + 1; /* window starts p0, p0 + 2, ..., p1 */
    double *P = malloc(sizeof(double) * 10 * (size_t)np);
    for (int fi = 0; fi < nf; fi++) {
        for (int k = 0; k < 10; k++) {
            double w = 2 * M_PI * ((k - 1) * TONE_SPACING + dfs[fi]) / FS_BB;
            cplx step = cos(w) + I * sin(w), back = cos(-SPS_BB * w) + I * sin(-SPS_BB * w), X = 0;
            for (int p = p0; p <= p1; p++) {
                if ((p - p0) % 64 == 0) { /* restart from the direct sum */
                    X = 0;
                    for (int n = 0; n < SPS_BB; n++) {
                        int i = p + n;
                        if (i >= 0 && i < NBB) X += y0[i] * (cos(-w * n) + I * sin(-w * n));
                    }
                }
                if (!((p - p0) & 1)) P[(size_t)k * np + (p - p0) / 2] = creal(X) * creal(X) + cimag(X) * cimag(X);
                cplx out = p >= 0 && p < NBB ? y0[p] : 0, in = p + SPS_BB >= 0 && p + SPS_BB < NBB ? y0[p + SPS_BB] : 0;
                X = step * (X - out + in * back);
            }
        }
        for (int ti = 0; ti < nt; ti++) {
            double num = 0, den = 0;
            for (int s = 0; s < ND; s++) {
                int q = (start0 + dts[ti] + SPS_BB * DATA_POS[s] - p0) / 2;
                double pin = 0, sum = 0;
                for (int k = 1; k < 9; k++) {
                    double v = P[(size_t)k * np + q];
                    pin = v > pin ? v : pin;
                    sum += v;
                }
                double a = P[q], b = P[(size_t)9 * np + q];
                num += pin - (a > b ? a : b);
                den += sum / 8;
            }
            M[fi * nt + ti] = num / (den > 1e-12 ? den : 1e-12);
        }
    }
    free(P);
}

/* ------------------------------------------------------------------ symbol spectra / metrics */

static cplx dft64[NBINS][SPS_BB]; /* exp(-2 pi i b n / 64) for b = -4..19 */
static cplx dft32[8][SPS_BB];

static void spectra_init(void) {
    for (int q = 0; q < NBINS; q++)
        for (int n = 0; n < SPS_BB; n++) {
            int b = ((NBIN_LO + q) % 64 + 64) % 64;
            double a = -2 * M_PI * b * n / 64.0;
            dft64[q][n] = cos(a) + I * sin(a);
        }
    for (int b = 0; b < 8; b++)
        for (int n = 0; n < SPS_BB; n++) {
            double a = -2 * M_PI * b * n / 32.0;
            dft32[b][n] = cos(a) + I * sin(a);
        }
}

void nn_features(const cplx *y, int start, float *out) {
    static const double isq = 0.17677669529663687; /* 1/sqrt(32) */
    double re[NN][NBINS], im[NN][NBINS];
    float p[NN * NBINS];
    for (int s = 0; s < NN; s++) {
        cplx seg[SPS_BB];
        for (int n = 0; n < SPS_BB; n++) {
            int idx = start + s * SPS_BB + n;
            /* torch path works on complex64 */
            seg[n] = (idx >= 0 && idx < NBB) ? (double)(float)creal(y[idx]) + I * (double)(float)cimag(y[idx]) : 0;
        }
        for (int q = 0; q < NBINS; q++) {
            cplx z = 0;
            for (int n = 0; n < SPS_BB; n++) z += seg[n] * dft64[q][n];
            z *= isq;
            re[s][q] = creal(z);
            im[s][q] = cimag(z);
            float a = (float)cabs(z);
            p[s * NBINS + q] = a * a;
        }
    }
    /* torch.median: lower of the two middle values */
    qsort(p, NN * NBINS, sizeof(float), cmp_f);
    double med = p[(NN * NBINS - 1) / 2];
    double scale = sqrt(fmax_d(med, 1e-6));
    for (int s = 0; s < NN; s++)
        for (int q = 0; q < NBINS; q++) {
            out[s * NFEAT + q] = (float)(re[s][q] / scale);
            out[s * NFEAT + NBINS + q] = (float)(im[s][q] / scale);
        }
}

void symbol_spectra8(const cplx *y, int start, cplx cs[NN][8]) {
    static const double isq = 0.17677669529663687;
    for (int s = 0; s < NN; s++) {
        cplx seg[SPS_BB];
        for (int n = 0; n < SPS_BB; n++) {
            int idx = start + s * SPS_BB + n;
            seg[n] = (idx >= 0 && idx < NBB) ? y[idx] : 0;
        }
        for (int b = 0; b < 8; b++) {
            cplx z = 0;
            for (int n = 0; n < SPS_BB; n++) z += seg[n] * dft32[b][n];
            cs[s][b] = z * isq;
        }
    }
}

static void normalise_llr(double *llr) {
    double m = 0, v = 0;
    for (int i = 0; i < 174; i++) m += llr[i];
    m /= 174;
    for (int i = 0; i < 174; i++) v += (llr[i] - m) * (llr[i] - m);
    double sd = fmax_d(sqrt(v / 174), 1e-9);
    for (int i = 0; i < 174; i++) llr[i] = 2.83 * llr[i] / sd;
}

static void group1(const cplx *d, double *out3) {
    double s[8];
    for (int t = 0; t < 8; t++) s[t] = cabs(d[t]);
    for (int b = 0; b < 3; b++) {
        double m1 = -INFINITY, m0 = -INFINITY;
        for (int t = 0; t < 8; t++) {
            if ((GRAY_INV[t] >> (2 - b)) & 1) m1 = fmax_d(m1, s[t]);
            else m0 = fmax_d(m0, s[t]);
        }
        out3[b] = m1 - m0;
    }
}

void coherent_llr(cplx cs[NN][8], int nsym, double *llr) {
    if (nsym == 1) {
        for (int i = 0; i < ND; i++) group1(cs[DATA_POS[i]], llr + 3 * i);
        normalise_llr(llr);
        return;
    }
    int nc = 1;
    for (int j = 0; j < nsym; j++) nc *= 8;
    int nbits = 3 * nsym;
    double *s = malloc(sizeof(double) * nc);
    for (int g0 = 0; g0 < ND; g0 += nsym) {
        int g = ND - g0 < nsym ? ND - g0 : nsym;
        if (g < nsym) {
            for (int j = 0; j < g; j++) group1(cs[DATA_POS[g0 + j]], llr + 3 * (g0 + j));
            continue;
        }
        for (int c = 0; c < nc; c++) {
            cplx acc = 0;
            for (int j = 0; j < nsym; j++) {
                int t = (c >> (3 * (nsym - 1 - j))) & 7; /* first symbol's tone most significant */
                acc += cs[DATA_POS[g0 + j]][t];
            }
            s[c] = cabs(acc);
        }
        /* per-symbol max over combos with each tone, then per bit max over tones: same maxima as a
           direct scan over all combos, with nsym*nc instead of 3*nsym*nc comparisons */
        double mt[3][8];
        for (int j = 0; j < nsym; j++)
            for (int t = 0; t < 8; t++) mt[j][t] = -INFINITY;
        for (int c = 0; c < nc; c++)
            for (int j = 0; j < nsym; j++) {
                int t = (c >> (3 * (nsym - 1 - j))) & 7;
                mt[j][t] = fmax_d(mt[j][t], s[c]);
            }
        for (int bi = 0; bi < nbits; bi++) {
            int j = bi / 3, b = bi % 3;
            double m1 = -INFINITY, m0 = -INFINITY;
            for (int t = 0; t < 8; t++) {
                if ((GRAY_INV[t] >> (2 - b)) & 1) m1 = fmax_d(m1, mt[j][t]);
                else m0 = fmax_d(m0, mt[j][t]);
            }
            llr[3 * g0 + bi] = m1 - m0;
        }
    }
    free(s);
    normalise_llr(llr);
}
