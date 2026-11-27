/* LDPC(174,91) sum-product BP and order-2 OSD (port of ft8nn/ldpc.py). LLR = log P(1)/P(0). */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"
#include "ldpc_tables.h"

static int edge_of_bit[174][3]; /* compact edge index 0..n_edges-1 of each bit's 3 edges, in LDPC_MN order */
static int edge_bit[83 * 7];    /* bit index 0..173 of each compact check edge */
static int row_off[84];         /* compact edge range [row_off[m], row_off[m+1]) of check row m */
static int n_edges;
static uint64_t G_rows[91][3];  /* systematic generator [I | GEN^T], bit c of row i */

static inline int getb(const uint64_t *r, int c) { return (r[c >> 6] >> (c & 63)) & 1; }
static inline void setb(uint64_t *r, int c) { r[c >> 6] |= 1ULL << (c & 63); }

void ldpc_init(void) {
    int e = 0;
    int nm_edge[83][7];
    for (int m = 0; m < 83; m++) {
        row_off[m] = e;
        for (int j = 0; j < 7; j++) {
            if (LDPC_NM[m][j] >= 0) {
                nm_edge[m][j] = e;
                edge_bit[e++] = LDPC_NM[m][j];
            } else {
                nm_edge[m][j] = -1;
            }
        }
    }
    row_off[83] = n_edges = e;
    for (int n = 0; n < 174; n++)
        for (int k = 0; k < 3; k++) {
            int m = LDPC_MN[n][k];
            for (int j = 0; j < 7; j++)
                if (LDPC_NM[m][j] == n) edge_of_bit[n][k] = nm_edge[m][j];
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
        for (int e = row_off[m]; e < row_off[m + 1]; e++) s ^= b[edge_bit[e]];
        if (s) return 0;
    }
    return 1;
}

static int n_unsatisfied(const uint8_t *b) {
    int c = 0;
    for (int m = 0; m < 83; m++) {
        int s = 0;
        for (int e = row_off[m]; e < row_off[m + 1]; e++) s ^= b[edge_bit[e]];
        c += s;
    }
    return c;
}

/* Branchless, SIMD-vectorizable tanh(q/2) and 2*atanh(ext) in IEEE-754 double (< 5e-12 max error vs libm). */
static inline double fast_tanh_half(double q) {
    q = q < -40.0 ? -40.0 : (q > 40.0 ? 40.0 : q);
    const double inv_ln2 = 1.44269504088896340736;
    const double ln2_hi  = 0.69314718036912381649;
    const double ln2_lo  = 1.90821492927058770002e-10;
    double kf = nearbyint(q * inv_ln2);
    int64_t k = (int64_t)kf;
    double r = (q - kf * ln2_hi) - kf * ln2_lo;
    double em1 = r * (1.0 + r * (0.5 + r * (1.6666666666666666e-1 +
                 r * (4.1666666666666664e-2 + r * (8.3333333333333332e-3 +
                 r * (1.3888888888888889e-3 + r * (1.9841269841269841e-4 +
                 r * (2.4801587301587302e-5 + r * 2.7557319223985891e-6))))))));
    uint64_t sb = (uint64_t)(k + 1023) << 52;
    double scale;
    memcpy(&scale, &sb, sizeof scale);
    double se = scale * em1;
    double t = ((scale - 1.0) + se) / ((scale + 1.0) + se);
    return fabs(t) < 1e-12 ? 1e-12 : t;
}

static inline double fast_2atanh(double ext) {
    ext = ext < -0.9999999 ? -0.9999999 : (ext > 0.9999999 ? 0.9999999 : ext);
    double u = (1.0 + ext) / (1.0 - ext);
    uint64_t ub;
    memcpy(&ub, &u, sizeof ub);
    int64_t adj = (int64_t)(ub - 0x3fe6a09e667f3bcdULL);
    int64_t e = adj >> 52;
    uint64_t mb = ((uint64_t)adj & 0x000fffffffffffffULL) + 0x3fe6a09e667f3bcdULL;
    double m;
    memcpy(&m, &mb, sizeof m);
    double s = (m - 1.0) / (m + 1.0), s2 = s * s;
    double lnm = 2.0 * s * (1.0 + s2 * (0.33333333333333333 + s2 * (0.2 + s2 * (0.14285714285714285 +
                 s2 * (0.1111111111111111 + s2 * (0.09090909090909091 + s2 * 0.07692307692307693))))));
    return (double)e * 0.69314718055994530942 + lnm;
}

/* ldpc.BP_STOP: stop a run after BP_PATIENCE iterations in a row without fewer unsatisfied checks, once at least
 * BP_MIN_ITER iterations are done and more than BP_FLOOR checks are unsatisfied */
#define BP_PATIENCE 8
#define BP_MIN_ITER 5
#define BP_FLOOR 15

int bp_decode(const double *llr, int iters, uint8_t *out, int *nhard) {
    double Lp[174], R[83 * 7], tot[174], T[83 * 7], ext[83 * 7];
    uint8_t hard[174];
    const int ne = n_edges;
    for (int n = 0; n < 174; n++) {
        Lp[n] = -llr[n];
        hard[n] = llr[n] > 0;
    }
    memcpy(out, hard, 174);
    int done = syndrome_ok(hard);
    memset(R, 0, sizeof(double) * (size_t)ne);
    int last = 0, cnt = 0;
    for (int it = 0; it < iters && !done; it++) {
        for (int n = 0; n < 174; n++) {
            const int *e = edge_of_bit[n];
            tot[n] = Lp[n] + (R[e[0]] + R[e[1]] + R[e[2]]);
        }
        for (int e = 0; e < ne; e++) T[e] = fast_tanh_half(tot[edge_bit[e]] - R[e]);
        for (int m = 0; m < 83; m++) {
            int e0 = row_off[m], e1 = row_off[m + 1];
            double prod = T[e0];
            for (int e = e0 + 1; e < e1; e++) prod *= T[e];
            for (int e = e0; e < e1; e++) ext[e] = prod / T[e];
        }
        for (int e = 0; e < ne; e++) R[e] = fast_2atanh(ext[e]);
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
    double Lp[174], R[83 * 7], tot[174], T[83 * 7], ext[83 * 7], zsum[174];
    const int ne = n_edges;
    for (int i = 0; i < 174; i++) {
        Lp[i] = -llr[i];
        zsum[i] = 0.0;
    }
    memset(R, 0, sizeof(double) * (size_t)ne);
    for (int it = 0; it < n; it++) {
        for (int i = 0; i < 174; i++) {
            const int *e = edge_of_bit[i];
            tot[i] = Lp[i] + (R[e[0]] + R[e[1]] + R[e[2]]);
        }
        for (int e = 0; e < ne; e++) T[e] = fast_tanh_half(tot[edge_bit[e]] - R[e]);
        for (int m = 0; m < 83; m++) {
            int e0 = row_off[m], e1 = row_off[m + 1];
            double prod = T[e0];
            for (int e = e0 + 1; e < e1; e++) prod *= T[e];
            for (int e = e0; e < e1; e++) ext[e] = prod / T[e];
        }
        for (int e = 0; e < ne; e++) R[e] = fast_2atanh(ext[e]);
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
