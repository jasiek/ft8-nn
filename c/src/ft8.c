/* FT8 protocol primitives (port of ft8nn/ft8.py): CRC-14, LDPC encode, tones, GFSK waveform. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"
#include "ldpc_tables.h"

const int COSTAS[7] = {3, 1, 4, 0, 6, 5, 2};
const int GRAY[8] = {0, 1, 3, 2, 5, 6, 4, 7};
const int GRAY_INV[8] = {0, 1, 3, 2, 6, 4, 5, 7};
const int DATA_POS[ND] = {7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                          27, 28, 29, 30, 31, 32, 33, 34, 35, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                          54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71};
const int SYNC_SYMS[21] = {0, 1, 2, 3, 4, 5, 6, 36, 37, 38, 39, 40, 41, 42, 72, 73, 74, 75, 76, 77, 78};

int crc14(const uint8_t *bits, int n) {
    int rem = 0;
    for (int i = 0; i < n; i++) {
        rem ^= (bits[i] & 1) << 13;
        rem = (rem & 0x2000) ? ((rem << 1) ^ 0x2757) : (rem << 1);
        rem &= 0x3FFF;
    }
    return rem;
}

int crc_ok(const uint8_t *a91) {
    uint8_t b[82];
    memcpy(b, a91, 77);
    memset(b + 77, 0, 5);
    int c = crc14(b, 82), got = 0;
    for (int i = 77; i < 91; i++) got = (got << 1) | a91[i];
    return c == got;
}

void encode77(const uint8_t *payload77, uint8_t *cw, int *tones) {
    uint8_t b[82];
    memcpy(b, payload77, 77);
    memset(b + 77, 0, 5);
    int c = crc14(b, 82);
    memcpy(cw, payload77, 77);
    for (int i = 0; i < 14; i++) cw[77 + i] = (c >> (13 - i)) & 1;
    for (int p = 0; p < 83; p++) {
        int s = 0;
        for (int j = 0; j < 91; j++) s += LDPC_GEN[p][j] & cw[j];
        cw[91 + p] = s & 1;
    }
    if (!tones) return;
    for (int k = 0; k < 7; k++) tones[k] = tones[36 + k] = tones[72 + k] = COSTAS[k];
    for (int i = 0; i < ND; i++) tones[DATA_POS[i]] = GRAY[cw[3 * i] * 4 + cw[3 * i + 1] * 2 + cw[3 * i + 2]];
}

static double dphi_tab[3 * NSPS];
static double ramp_tab[NSPS / 8];

void ft8_init(void) { /* call once before threads start (ft8nn_open does) */
    double c = M_PI * sqrt(2.0 / log(2.0)), bt = 2.0, pk = 2.0 * M_PI / NSPS;
    for (int i = 0; i < 3 * NSPS; i++) {
        double t = ((double)i - 1.5 * NSPS) / NSPS;
        dphi_tab[i] = pk * 0.5 * (erf(c * bt * (t + 0.5)) - erf(c * bt * (t - 0.5)));
    }
    int nr = NSPS / 8;
    for (int i = 0; i < nr; i++) ramp_tab[i] = 0.5 * (1.0 - cos(M_PI * i / nr));
}

void complex_wave_split(const int *tones, double f0, double *restrict refr, double *restrict refi) {
    enum { N = NN * NSPS, ND_ = (NN + 2) * NSPS };
    double *d = calloc(ND_, sizeof(double));
    for (int j = 0; j < NN; j++) {
        double tj = tones[j];
        double *dj = d + j * NSPS;
        for (int i = 0; i < 3 * NSPS; i++) dj[i] += dphi_tab[i] * tj;
    }
    double t0 = tones[0], tn = tones[NN - 1];
    for (int i = 0; i < 2 * NSPS; i++) d[i] += t0 * dphi_tab[NSPS + i];
    for (int i = 0; i < 2 * NSPS; i++) d[NN * NSPS + i] += tn * dphi_tab[i];
    double w0 = 2.0 * M_PI * f0 / FS;
    /* Prefix-sum phase first so the trig loop has no loop-carried dependency and vectorises */
    double *ph = malloc(sizeof(double) * N);
    double acc = 0.0;
    for (int i = 0; i < N; i++) {
        ph[i] = acc;
        acc += d[NSPS + i] + w0;
    }
    free(d);
    const double inv_pio2 = 0.63661977236758134308;
    const double pio2_1 = 1.57079632679489655800;
    const double pio2_2 = 6.12323399573676603587e-17;
    for (int i = 0; i < N; i++) {
        double x = ph[i];
        double kf = nearbyint(x * inv_pio2);
        int q = ((int)(int64_t)kf) & 3;
        double r = (x - kf * pio2_1) - kf * pio2_2;
        double r2 = r * r;
        double s = r * (1.0 + r2 * (-1.0 / 6.0 + r2 * (1.0 / 120.0 + r2 * (-1.0 / 5040.0 +
                        r2 * (1.0 / 362880.0 + r2 * (-1.0 / 39916800.0 + r2 * (1.0 / 6227020800.0)))))));
        double c = 1.0 + r2 * (-0.5 + r2 * (1.0 / 24.0 + r2 * (-1.0 / 720.0 +
                   r2 * (1.0 / 40320.0 + r2 * (-1.0 / 3628800.0 + r2 * (1.0 / 479001600.0))))));
        double co = (q & 1) ? -s : c;
        double si = (q & 1) ? c : s;
        if (q & 2) {
            co = -co;
            si = -si;
        }
        refr[i] = co;
        refi[i] = si;
    }
    int nr = NSPS / 8;
    for (int i = 0; i < nr; i++) {
        refr[i] *= ramp_tab[i];
        refi[i] *= ramp_tab[i];
        refr[N - 1 - i] *= ramp_tab[i];
        refi[N - 1 - i] *= ramp_tab[i];
    }
    free(ph);
}

void complex_wave(const int *tones, double f0, cplx *out) {
    enum { N = NN * NSPS };
    double *refr = malloc(sizeof(double) * N), *refi = malloc(sizeof(double) * N);
    complex_wave_split(tones, f0, refr, refi);
    for (int i = 0; i < N; i++) out[i] = refr[i] + I * refi[i];
    free(refr);
    free(refi);
}
