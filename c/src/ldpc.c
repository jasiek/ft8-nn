/* LDPC(174,91) sum-product BP and order-2 OSD (port of ft8nn/ldpc.py). LLR = log P(1)/P(0). */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"
#include "ldpc_tables.h"

static int edge_of_bit[174][3]; /* flat index m*7+j of each bit's edges, in LDPC_MN order */
static uint64_t G_rows[91][3];  /* systematic generator [I | GEN^T], bit c of row i */

static inline int getb(const uint64_t *r, int c) { return (r[c >> 6] >> (c & 63)) & 1; }
static inline void setb(uint64_t *r, int c) { r[c >> 6] |= 1ULL << (c & 63); }

void ldpc_init(void) {
    for (int n = 0; n < 174; n++)
        for (int k = 0; k < 3; k++) {
            int m = LDPC_MN[n][k];
            for (int j = 0; j < 7; j++)
                if (LDPC_NM[m][j] == n) edge_of_bit[n][k] = m * 7 + j;
        }
    memset(G_rows, 0, sizeof G_rows);
    for (int i = 0; i < 91; i++) {
        setb(G_rows[i], i);
        for (int p = 0; p < 83; p++)
            if (LDPC_GEN[p][i]) setb(G_rows[i], 91 + p);
    }
}

static int syndrome_ok(const uint8_t *b) {
    for (int m = 0; m < 83; m++) {
        int s = 0;
        for (int j = 0; j < 7; j++)
            if (LDPC_NM[m][j] >= 0) s ^= b[LDPC_NM[m][j]];
        if (s) return 0;
    }
    return 1;
}

static int n_unsatisfied(const uint8_t *b) {
    int c = 0;
    for (int m = 0; m < 83; m++) {
        int s = 0;
        for (int j = 0; j < 7; j++)
            if (LDPC_NM[m][j] >= 0) s ^= b[LDPC_NM[m][j]];
        c += s;
    }
    return c;
}

/* ldpc.BP_STOP: stop a run after BP_PATIENCE iterations in a row without fewer unsatisfied checks, once at least
 * BP_MIN_ITER iterations are done and more than BP_FLOOR checks are unsatisfied */
#define BP_PATIENCE 8
#define BP_MIN_ITER 5
#define BP_FLOOR 15

int bp_decode(const double *llr, int iters, uint8_t *out, int *nhard) {
    double Lp[174], R[83 * 7], tot[174], T[7];
    uint8_t hard[174];
    for (int n = 0; n < 174; n++) {
        Lp[n] = -llr[n];
        hard[n] = llr[n] > 0;
    }
    memcpy(out, hard, 174);
    int done = syndrome_ok(hard);
    memset(R, 0, sizeof R);
    int last = 0, cnt = 0;
    for (int it = 0; it < iters && !done; it++) {
        for (int n = 0; n < 174; n++) {
            const int *e = edge_of_bit[n];
            tot[n] = Lp[n] + (R[e[0]] + R[e[1]] + R[e[2]]);
        }
        for (int m = 0; m < 83; m++) {
            double *Rm = R + 7 * m;
            for (int j = 0; j < 7; j++) {
                int nb = LDPC_NM[m][j];
                if (nb < 0) {
                    T[j] = 1.0;
                    continue;
                }
                double q = tot[nb] - Rm[j];
                q = q < -40 ? -40 : (q > 40 ? 40 : q);
                double e = expm1(q);
                double t = e / (e + 2); /* tanh(q/2) */
                T[j] = fabs(t) < 1e-12 ? 1e-12 : t;
            }
            double prod = T[0];
            for (int j = 1; j < 7; j++) prod *= T[j];
            for (int j = 0; j < 7; j++) {
                if (LDPC_NM[m][j] < 0) {
                    Rm[j] = 0.0;
                    continue;
                }
                double ext = prod / T[j];
                ext = ext < -0.9999999 ? -0.9999999 : (ext > 0.9999999 ? 0.9999999 : ext);
                Rm[j] = log((1 + ext) / (1 - ext)); /* 2 atanh(ext) */
            }
        }
        for (int n = 0; n < 174; n++) {
            const int *e = edge_of_bit[n];
            out[n] = (Lp[n] + (R[e[0]] + R[e[1]] + R[e[2]])) < 0;
        }
        int nc = n_unsatisfied(out);
        done = nc == 0;
        if (!done) {
            if (it > 0) cnt = nc < last ? 0 : cnt + 1;
            last = nc;
            if (cnt >= BP_PATIENCE && it + 1 >= BP_MIN_ITER && nc > BP_FLOOR) break;
        }
    }
    int nh = 0;
    for (int n = 0; n < 174; n++) nh += hard[n] != out[n];
    *nhard = nh;
    return done;
}

/* Running sums of the BP posteriors (LLR sign) after iterations 1..n into zs[n][174] (decoder ldpc.bp_posterior_sums,
 * WSJT-X zsave). Same arithmetic as bp_decode, no early stop. */
void bp_posterior_sums(const double *llr, int n, double *zs) {
    double Lp[174], R[83 * 7], tot[174], T[7], zsum[174];
    for (int i = 0; i < 174; i++) {
        Lp[i] = -llr[i];
        zsum[i] = 0.0;
    }
    memset(R, 0, sizeof R);
    for (int it = 0; it < n; it++) {
        for (int i = 0; i < 174; i++) {
            const int *e = edge_of_bit[i];
            tot[i] = Lp[i] + (R[e[0]] + R[e[1]] + R[e[2]]);
        }
        for (int m = 0; m < 83; m++) {
            double *Rm = R + 7 * m;
            for (int j = 0; j < 7; j++) {
                int nb = LDPC_NM[m][j];
                if (nb < 0) {
                    T[j] = 1.0;
                    continue;
                }
                double q = tot[nb] - Rm[j];
                q = q < -40 ? -40 : (q > 40 ? 40 : q);
                double e = expm1(q);
                double t = e / (e + 2); /* tanh(q/2) */
                T[j] = fabs(t) < 1e-12 ? 1e-12 : t;
            }
            double prod = T[0];
            for (int j = 1; j < 7; j++) prod *= T[j];
            for (int j = 0; j < 7; j++) {
                if (LDPC_NM[m][j] < 0) {
                    Rm[j] = 0.0;
                    continue;
                }
                double ext = prod / T[j];
                ext = ext < -0.9999999 ? -0.9999999 : (ext > 0.9999999 ? 0.9999999 : ext);
                Rm[j] = log((1 + ext) / (1 - ext)); /* 2 atanh(ext) */
            }
        }
        for (int i = 0; i < 174; i++) {
            const int *e = edge_of_bit[i];
            zsum[i] -= Lp[i] + (R[e[0]] + R[e[1]] + R[e[2]]);
            zs[it * 174 + i] = zsum[i];
        }
    }
}

typedef struct {
    double rel;
    int idx;
} relidx;

static int cmp_rel(const void *a, const void *b) {
    const relidx *x = a, *y = b;
    if (x->rel > y->rel) return -1;
    if (x->rel < y->rel) return 1;
    return x->idx - y->idx; /* stable: ties keep index order */
}

static double wdist(const uint64_t *c, const uint64_t *hp, const double *rp) {
    double d = 0;
    for (int w = 0; w < 3; w++) {
        uint64_t x = c[w] ^ hp[w];
        while (x) {
            int b = __builtin_ctzll(x);
            d += rp[64 * w + b];
            x &= x - 1;
        }
    }
    return d;
}

int osd_decode(const double *llr, int order, uint8_t *cw, double *dist, int *nhard) {
    relidx ri[174];
    uint8_t hard[174];
    for (int n = 0; n < 174; n++) {
        ri[n].rel = fabs(llr[n]);
        ri[n].idx = n;
        hard[n] = llr[n] > 0;
    }
    qsort(ri, 174, sizeof(relidx), cmp_rel);
    int perm[174];
    double rp[174];
    uint64_t hp[3] = {0, 0, 0};
    for (int i = 0; i < 174; i++) {
        perm[i] = ri[i].idx;
        rp[i] = ri[i].rel;
        if (hard[perm[i]]) setb(hp, i);
    }
    /* Gp = G[:, perm], row-reduced over GF(2) with pivots chosen left to right */
    uint64_t Gr[91][3];
    memset(Gr, 0, sizeof Gr);
    for (int i = 0; i < 91; i++)
        for (int c = 0; c < 174; c++)
            if (getb(G_rows[i], perm[c])) setb(Gr[i], c);
    int piv[91], r = 0;
    for (int c = 0; c < 174 && r < 91; c++) {
        int p = -1;
        for (int i = r; i < 91; i++)
            if (getb(Gr[i], c)) {
                p = i;
                break;
            }
        if (p < 0) continue;
        if (p != r)
            for (int w = 0; w < 3; w++) {
                uint64_t t = Gr[r][w];
                Gr[r][w] = Gr[p][w];
                Gr[p][w] = t;
            }
        for (int i = 0; i < 91; i++)
            if (i != r && getb(Gr[i], c))
                for (int w = 0; w < 3; w++) Gr[i][w] ^= Gr[r][w];
        piv[r++] = c;
    }
    /* r == 91 always: G has full rank */
    uint64_t c0[3] = {0, 0, 0};
    for (int i = 0; i < 91; i++)
        if (getb(hp, piv[i]))
            for (int w = 0; w < 3; w++) c0[w] ^= Gr[i][w];

    uint64_t best[3], c[3];
    memcpy(best, c0, sizeof c0);
    double bd = wdist(c0, hp, rp);
#define TRY()                                   \
    do {                                        \
        double d_ = wdist(c, hp, rp);           \
        if (d_ < bd) {                          \
            bd = d_;                            \
            memcpy(best, c, sizeof c);          \
        }                                       \
    } while (0)
    if (order >= 1)
        for (int i = 0; i < 91; i++) {
            for (int w = 0; w < 3; w++) c[w] = c0[w] ^ Gr[i][w];
            TRY();
        }
    double p_rel[91];
    for (int i = 0; i < 91; i++) p_rel[i] = rp[piv[i]];
    if (order >= 2) {
        const int k2 = order >= 3 ? 60 : 40, lo = 91 - k2;
        for (int i = 0; i < k2; i++) {
            double ri_p = p_rel[lo + i];
            for (int j = i + 1; j < k2; j++) {
                if (ri_p + p_rel[lo + j] >= bd + 1e-12) continue;
                for (int w = 0; w < 3; w++) c[w] = c0[w] ^ Gr[lo + i][w] ^ Gr[lo + j][w];
                TRY();
            }
        }
    }
    if (order >= 3) {
        const int k3 = 28, lo = 91 - k3;
        for (int a = 0; a < k3; a++) {
            double ra = p_rel[lo + a];
            if (ra >= bd + 1e-12) continue;
            for (int b = a + 1; b < k3; b++) {
                double rab = ra + p_rel[lo + b];
                if (rab >= bd + 1e-12) continue;
                uint64_t cab[3];
                for (int w = 0; w < 3; w++) cab[w] = c0[w] ^ Gr[lo + a][w] ^ Gr[lo + b][w];
                for (int k = b + 1; k < k3; k++) {
                    if (rab + p_rel[lo + k] >= bd + 1e-12) continue;
                    for (int w = 0; w < 3; w++) c[w] = cab[w] ^ Gr[lo + k][w];
                    TRY();
                }
            }
        }
    }
#undef TRY
    int nh = 0;
    for (int i = 0; i < 174; i++) {
        cw[perm[i]] = getb(best, i);
    }
    for (int n = 0; n < 174; n++) nh += cw[n] != hard[n];
    *dist = bd;
    *nhard = nh;
    return crc_ok(cw);
}
