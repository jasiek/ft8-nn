/* Internal interfaces of libft8nn. Each function mirrors the Python function of the same name
 * in ft8nn/ (frontend.py, ldpc.py, ft8.py, subtract.py, message.py, decoder.py). */
#pragma once
#include <complex.h>
#include <stddef.h>
#include <stdint.h>

typedef double complex cplx;

/* ---- protocol constants (ft8.py, frontend.py) */
#define FS 12000
#define NSPS 1920
#define NN 79
#define ND 58
#define SLOT_N 180000
#define NFFT_FULL 192000
#define NBB 3200
#define SPS_BB 32
#define FS_BB 200.0
#define DF_FULL (12000.0 / NFFT_FULL)
#define TONE_SPACING 6.25
#define NOMINAL_START 0.5
#define NBIN_LO (-4)
#define NBINS 24
#define NFEAT (2 * NBINS)

extern const int COSTAS[7];
extern const int GRAY[8];
extern const int GRAY_INV[8];
extern const int DATA_POS[ND];
extern const int SYNC_SYMS[21];

/* ---- par.c: a pool of worker threads (one per context; a context is used by one thread at a time) */
#define PAR_MAX_THREADS 64
typedef struct par par;
par *par_new(int nthreads); /* nthreads <= 0: one per (performance) core */
void par_free(par *p);
int par_threads(const par *p); /* threads including the caller; 1 for p == NULL */
/* fn(arg, w) on every thread of the pool at once, w = 0 .. par_threads - 1 (0 is the caller); returns when all are
 * done. Not reentrant: fn must not call par_run / par_for on the same pool. p == NULL runs fn(arg, 0). */
void par_run(par *p, void (*fn)(void *, int), void *arg);
/* body(arg, i) for every i in [0, n), on the pool's threads in any order */
void par_for(par *p, int n, void (*body)(void *, int), void *arg);

/* ---- fft.c: complex and real-input FFTs for sizes with factors 2, 3, 5 */
typedef struct fft_plan fft_plan;
typedef struct rfft_plan rfft_plan;
fft_plan *fft_plan_new(int n);
void fft_plan_free(fft_plan *p);
int fft_size(const fft_plan *p);
/* out = DFT(in), forward sign exp(-2 pi i k n / N), unnormalised. in != out. */
void fft_forward(const fft_plan *p, const cplx *in, cplx *out);
/* out = unnormalised inverse DFT (sign +). in != out. */
void fft_inverse(const fft_plan *p, const cplx *in, cplx *out);
/* the same on a thread pool (bit-identical results; pool may be NULL) */
void fft_forward_par(const fft_plan *p, const cplx *in, cplx *out, par *pool);
void fft_inverse_par(const fft_plan *p, const cplx *in, cplx *out, par *pool);
/* real-to-complex forward DFT of even size n: reads in[0..m_valid) zero-padded to n, writes out[0..n/2] */
rfft_plan *rfft_plan_new(int n);
void rfft_plan_free(rfft_plan *p);
void fft_rfft(const rfft_plan *p, const double *in, int m_valid, cplx *out);
void fft_rfft_par(const rfft_plan *p, const double *in, int m_valid, cplx *out, par *pool);

/* ---- ft8.c */
void ft8_init(void);
int crc14(const uint8_t *bits, int n);
int crc_ok(const uint8_t *a91);
void encode77(const uint8_t *payload77, uint8_t *cw174, int *tones79);
/* unit-amplitude analytic FT8 waveform with ramps, NN*NSPS samples */
void complex_wave(const int *tones, double f0, cplx *out);
void complex_wave_split(const int *tones, double f0, double *refr, double *refi);

/* ---- ldpc.c (LLR = log P(1)/P(0)) */
void ldpc_init(void);
/* returns 1 if all checks pass; bits, nhard written */
int bp_decode(const double *llr, int iters, uint8_t *bits, int *nhard);
/* returns crc_ok; writes cw, dist, nhard */
int osd_decode(const double *llr, int order, uint8_t *cw, double *dist, int *nhard);
/* running sums of BP posteriors after iterations 1..n -> zs[n*174] */
void bp_posterior_sums(const double *llr, int n, double *zs);

/* ---- frontend.c */
void frontend_init(void); /* once, before threads */
typedef struct {
    double f0, dt, score;
    int early; /* from the early-DT track (2 of 3 Costas arrays in the slot): plain decoders only */
} candidate;
/* out must hold max_cand + (fmax - fmin) / 3.125 entries: up to max_cand main candidates, then the early-DT track */
int find_candidates(const double *x, int n, double fmin, double fmax, int max_cand, double min_score, int base_median,
                    double early_dt, double early_min, double dt_shift, candidate *out, par *pool);

typedef struct {
    const fft_plan *bb;
    cplx *X;  /* NFFT_FULL/2+1 bins of the slot */
    double noise_psd;
    double *sbase;  /* jt9's SNR noise floor (spectrum_baseline), dB per 3.125 Hz bin, BASE_NB bins */
    double base_ref; /* 10^(sbase/10) / base_ref: noise at that bin / the noise downsample() normalises to */
} baseband;
#define BASE_NFA 200.0  /* frontend.BASE_NFA, BASE_NFB: the noise floor's fit range without a WSJT-X request */
#define BASE_NFB 4000.0
#define BASE_NB 1921    /* bins of a 3840-point real FFT */
/* nfa, nfb: the range the SNR noise floor is fitted over (frontend.Baseband rng) */
void baseband_init(baseband *b, const double *x, int n, double nfa, double nfb, par *pool);
void baseband_free(baseband *b);
void baseband_downsample(const baseband *b, double f0, cplx *y);
double baseband_noise_ratio(const baseband *b, double f0); /* frontend.Baseband.noise_ratio */

typedef struct {
    double f0, dt, metric;
    double nratio; /* baseband_noise_ratio at f0, for estimate_snr */
    int start;
    cplx y[NBB];
} sync_hyp;
/* returns number of hypotheses written (<= n_hyp) */
int fine_sync2(const baseband *b, const candidate *c, int n_hyp, sync_hyp *out);
/* frontend.data_sync: data-symbol sync metric M[fi * nt + ti] at offsets dfs (Hz) x dts (baseband samples, ascending,
 * one parity) from start0 */
void data_sync(const cplx *y0, int start0, const double *dfs, int nf, const int *dts, int nt, double *M);
/* frontend._rotate: y *= exp(-2 pi i df n / FS_BB) */
void bb_rotate(cplx *y, double df);

/* NN input features (79 x NFEAT floats) from y rounded to float32, as model.features() */
/* nv: baseband samples that carry audio (an early pass; the rest is zero), 0 = all. The normalising median is then
 * taken over the symbols that end within them (model.features n_valid). */
void nn_features(const cplx *y, int start, int nv, float *out);
/* symbol_spectra(y, start)[:, :8] */
void symbol_spectra8(const cplx *y, int start, cplx cs[NN][8]);
void coherent_llr(cplx cs[NN][8], int nsym, double *llr174);

/* ---- subtract.c */
void subtract_init(void);
void subtract_signal(double *x, int n, const int *tones, double f0, double dt, par *pool);

/* ---- unpack.c */
typedef struct hash_table hash_table;
hash_table *ht_new(void);
void ht_free(hash_table *h);
void ht_clear(hash_table *h);
void unpack77(const uint8_t *bits77, hash_table *ht, char *out, int outlen);
int plausible77(const uint8_t *bits77, int deep); /* message.plausible */

/* ---- nn_accel.c: native SoftDemod inference via Apple Accelerate BLAS/AMX on macOS */
#define NN_ACCEL_MAXB 32
typedef struct nn_accel nn_accel;
nn_accel *nn_accel_open(const char *onnx_path);
void nn_accel_close(nn_accel *m);
size_t nn_accel_scratch_floats(const nn_accel *m);
void nn_accel_run(const nn_accel *m, const float *feat, int b, double *llr, float *scratch);
