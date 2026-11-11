/* Internal interfaces of libft8nn. Each function mirrors the Python function of the same name
 * in ft8nn/ (frontend.py, ldpc.py, ft8.py, subtract.py, message.py, decoder.py). */
#pragma once
#include <complex.h>
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

/* ---- fft.c: complex FFT for sizes with factors 2, 3, 5 */
typedef struct fft_plan fft_plan;
fft_plan *fft_plan_new(int n);
void fft_plan_free(fft_plan *p);
int fft_size(const fft_plan *p);
/* out = DFT(in), forward sign exp(-2 pi i k n / N), unnormalised. in != out. */
void fft_forward(const fft_plan *p, const cplx *in, cplx *out);
/* out = unnormalised inverse DFT (sign +). in != out. */
void fft_inverse(const fft_plan *p, const cplx *in, cplx *out);

/* ---- ft8.c */
void ft8_init(void);
int crc14(const uint8_t *bits, int n);
int crc_ok(const uint8_t *a91);
void encode77(const uint8_t *payload77, uint8_t *cw174, int *tones79);
/* unit-amplitude analytic FT8 waveform with ramps, NN*NSPS samples */
void complex_wave(const int *tones, double f0, cplx *out);

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
                    double early_dt, double early_min, double dt_shift, candidate *out);

typedef struct {
    fft_plan *full, *bb;
    cplx *X;  /* NFFT_FULL/2+1 bins of the slot */
    double noise_psd;
    cplx *tmp; /* scratch NFFT_FULL */
} baseband;
void baseband_init(baseband *b, const double *x, int n);
void baseband_free(baseband *b);
void baseband_downsample(const baseband *b, double f0, cplx *y);

typedef struct {
    double f0, dt, metric;
    int start;
    cplx y[NBB];
} sync_hyp;
/* returns number of hypotheses written (<= n_hyp) */
int fine_sync2(const baseband *b, const candidate *c, int n_hyp, sync_hyp *out);

/* NN input features (79 x NFEAT floats) from y rounded to float32, as model.features() */
void nn_features(const cplx *y, int start, float *out);
/* symbol_spectra(y, start)[:, :8] */
void symbol_spectra8(const cplx *y, int start, cplx cs[NN][8]);
void coherent_llr(cplx cs[NN][8], int nsym, double *llr174);

/* ---- subtract.c */
void subtract_init(void);
void subtract_signal(double *x, int n, const int *tones, double f0, double dt);

/* ---- unpack.c */
typedef struct hash_table hash_table;
hash_table *ht_new(void);
void ht_free(hash_table *h);
void ht_clear(hash_table *h);
void unpack77(const uint8_t *bits77, hash_table *ht, char *out, int outlen);
int plausible77(const uint8_t *bits77, int deep); /* message.plausible */
