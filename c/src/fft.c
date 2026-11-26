/* Mixed-radix (4, 2, 3, 5) recursive decimation-in-time complex FFT.
 *
 * Out-of-place. At each level the input is split into r interleaved subsequences, each is
 * transformed recursively into a contiguous block of the output, and the blocks are combined
 * with twiddled radix-r butterflies. Twiddles exp(-2 pi i k / N) are tabulated once per plan.
 */
#include <math.h>
#include <stdlib.h>

#include "internal.h"

struct fft_plan {
    int n, nf;
    int fac[40];
    cplx *tw; /* tw[k] = exp(-2 pi i k / n) */
};

fft_plan *fft_plan_new(int n) {
    fft_plan *p = calloc(1, sizeof *p);
    p->n = n;
    int m = n;
    static const int radices[] = {4, 2, 3, 5};
    for (int i = 0; i < 4 && m > 1;) {
        if (m % radices[i] == 0) {
            p->fac[p->nf++] = radices[i];
            m /= radices[i];
        } else {
            i++;
        }
    }
    if (m != 1) {
        free(p);
        return NULL;
    }
    p->tw = malloc(sizeof(cplx) * n);
    for (int k = 0; k < n; k++) {
        double a = -2.0 * M_PI * (double)k / n;
        p->tw[k] = cos(a) + I * sin(a);
    }
    return p;
}

void fft_plan_free(fft_plan *p) {
    if (!p) return;
    free(p->tw);
    free(p);
}

int fft_size(const fft_plan *p) { return p->n; }

/* Radix-r butterflies k0 <= k < k1 of one level: out[0..r*m) holds r transforms of length m, combined in place into
 * one of length r*m; ts = twiddle stride for this level. Each k touches only out[k + q*m], q < r. */
static void butterflies(const fft_plan *p, cplx *out, int r, int m, int ts, int k0, int k1) {
    const cplx *tw = p->tw;
    if (r == 2) {
        for (int k = k0; k < k1; k++) {
            cplx a = out[k], b = out[k + m] * tw[k * ts];
            out[k] = a + b;
            out[k + m] = a - b;
        }
    } else if (r == 4) {
        for (int k = k0; k < k1; k++) {
            cplx a0 = out[k], a1 = out[k + m] * tw[k * ts], a2 = out[k + 2 * m] * tw[2 * k * ts],
                 a3 = out[k + 3 * m] * tw[3 * k * ts];
            cplx s02 = a0 + a2, d02 = a0 - a2, s13 = a1 + a3, d13 = (a1 - a3) * (-I);
            out[k] = s02 + s13;
            out[k + m] = d02 + d13;
            out[k + 2 * m] = s02 - s13;
            out[k + 3 * m] = d02 - d13;
        }
    } else {
        /* generic radix-r butterfly (r = 3, 5) */
        int rs = p->n / r;
        cplx t[5];
        for (int k = k0; k < k1; k++) {
            for (int q = 0; q < r; q++) t[q] = out[k + q * m] * tw[q * k * ts];
            for (int j = 0; j < r; j++) {
                cplx s = t[0];
                for (int q = 1; q < r; q++) s += t[q] * tw[((j * q) % r) * rs];
                out[k + j * m] = s;
            }
        }
    }
}

/* out[0..len) = DFT_len of in[0], in[stride], ...; fi = index of this level's radix in fac. */
static void work(const fft_plan *p, cplx *out, const cplx *in, int stride, int fi, int len) {
    if (len == 1) {
        out[0] = in[0];
        return;
    }
    int r = p->fac[fi], m = len / r;
    for (int q = 0; q < r; q++) work(p, out + q * m, in + q * stride, stride * r, fi + 1, m);
    butterflies(p, out, r, m, p->n / len, 0, m);
}

/* The same transform on a thread pool, with the same arithmetic for every output: the top `depth` levels of the
 * recursion are unrolled, their sub-transforms run as independent work items, then each level's butterflies are
 * combined in chunks of k. */
#define BF_CHUNK 2048
typedef struct {
    const fft_plan *p;
    cplx *out;
    const cplx *in;
    int depth, nleaf, lenleaf, stride; /* leaves: nleaf transforms of length lenleaf, input stride `stride` */
    int lvl, blen, nkc;                /* combine step: level lvl, block length blen, k-chunks per block */
} par_fft;

static void leaf_item(void *a, int b) {
    par_fft *f = a;
    /* block b = (q0, ..., q_{depth-1}) row-major; its input starts at sum q_i * (product of the radices above i) */
    int off = 0, mul = 1, rest = b;
    int q[40];
    for (int i = f->depth - 1; i >= 0; i--) {
        q[i] = rest % f->p->fac[i];
        rest /= f->p->fac[i];
    }
    for (int i = 0; i < f->depth; i++) {
        off += q[i] * mul;
        mul *= f->p->fac[i];
    }
    work(f->p, f->out + (size_t)b * f->lenleaf, f->in + off, f->stride, f->depth, f->lenleaf);
}

static void combine_item(void *a, int it) {
    par_fft *f = a;
    int b = it / f->nkc, kc = it % f->nkc, r = f->p->fac[f->lvl], m = f->blen / r;
    int k0 = kc * BF_CHUNK, k1 = k0 + BF_CHUNK < m ? k0 + BF_CHUNK : m;
    butterflies(f->p, f->out + (size_t)b * f->blen, r, m, f->p->n / f->blen, k0, k1);
}

static void work_par(const fft_plan *p, cplx *out, const cplx *in, par *pool) {
    int nt = par_threads(pool), depth = 0, nleaf = 1;
    while (depth < p->nf && nleaf < 2 * nt) nleaf *= p->fac[depth++];
    if (nt == 1 || depth == 0) {
        work(p, out, in, 1, 0, p->n);
        return;
    }
    par_fft f = {.p = p, .out = out, .in = in, .depth = depth, .nleaf = nleaf, .lenleaf = p->n / nleaf, .stride = nleaf};
    par_for(pool, nleaf, leaf_item, &f);
    for (int lvl = depth - 1; lvl >= 0; lvl--) {
        int nb = 1;
        for (int i = 0; i < lvl; i++) nb *= p->fac[i];
        f.lvl = lvl;
        f.blen = p->n / nb;
        int m = f.blen / p->fac[lvl];
        f.nkc = (m + BF_CHUNK - 1) / BF_CHUNK;
        par_for(pool, nb * f.nkc, combine_item, &f);
    }
}

void fft_forward(const fft_plan *p, const cplx *in, cplx *out) { work(p, out, in, 1, 0, p->n); }

void fft_inverse(const fft_plan *p, const cplx *in, cplx *out) { fft_inverse_par(p, in, out, NULL); }

void fft_forward_par(const fft_plan *p, const cplx *in, cplx *out, par *pool) { work_par(p, out, in, pool); }

void fft_inverse_par(const fft_plan *p, const cplx *in, cplx *out, par *pool) {
    /* ifft(x) = conj(fft(conj(x))) */
    int n = p->n;
    cplx stk[4096];
    cplx *c = n <= 4096 ? stk : malloc(sizeof(cplx) * (size_t)n);
    for (int i = 0; i < n; i++) c[i] = conj(in[i]);
    work_par(p, out, c, pool);
    for (int i = 0; i < n; i++) out[i] = conj(out[i]);
    if (c != stk) free(c);
}

struct rfft_plan {
    int n, m;
    fft_plan *half;
    cplx *rtw; /* exp(-pi i k / m) for k = 0..m/2 */
};

rfft_plan *rfft_plan_new(int n) {
    if (n < 2 || (n & 1)) return NULL;
    int m = n / 2;
    fft_plan *half = fft_plan_new(m);
    if (!half) return NULL;
    rfft_plan *rp = calloc(1, sizeof *rp);
    rp->n = n;
    rp->m = m;
    rp->half = half;
    rp->rtw = malloc(sizeof(cplx) * (size_t)(m / 2 + 1));
    for (int k = 0; k <= m / 2; k++) {
        double a = -M_PI * (double)k / m;
        rp->rtw[k] = cos(a) + I * sin(a);
    }
    return rp;
}

void rfft_plan_free(rfft_plan *p) {
    if (!p) return;
    fft_plan_free(p->half);
    free(p->rtw);
    free(p);
}

static void rfft_unpack(const rfft_plan *p, const cplx *Z, cplx *out) {
    int m = p->m;
    const cplx *rtw = p->rtw;
    out[0] = creal(Z[0]) + cimag(Z[0]);
    out[m] = creal(Z[0]) - cimag(Z[0]);
    for (int k = 1; k <= m / 2; k++) {
        int mk = m - k;
        cplx zk = Z[k], zmk = conj(Z[mk]);
        cplx ek = 0.5 * (zk + zmk);
        cplx ok = -0.5 * I * (zk - zmk);
        cplx wk = rtw[k] * ok;
        out[k] = ek + wk;
        if (k != mk) out[mk] = conj(ek - wk);
    }
}

static void rfft_pack(const double *in, int n, int m_valid, cplx *z) {
    int m = n / 2;
    if (m_valid > n) m_valid = n;
    if (m_valid < 0) m_valid = 0;
    int pairs = m_valid / 2;
    for (int i = 0; i < pairs; i++) z[i] = in[2 * i] + I * in[2 * i + 1];
    if (m_valid & 1) {
        z[pairs] = in[2 * pairs];
        for (int i = pairs + 1; i < m; i++) z[i] = 0;
    } else {
        for (int i = pairs; i < m; i++) z[i] = 0;
    }
}

void fft_rfft(const rfft_plan *p, const double *in, int m_valid, cplx *out) {
    int m = p->m;
    cplx z_stk[2048], Z_stk[2048];
    cplx *z = m <= 2048 ? z_stk : malloc(sizeof(cplx) * (size_t)m);
    cplx *Z = m <= 2048 ? Z_stk : malloc(sizeof(cplx) * (size_t)m);
    rfft_pack(in, p->n, m_valid, z);
    fft_forward(p->half, z, Z);
    rfft_unpack(p, Z, out);
    if (z != z_stk) free(z);
    if (Z != Z_stk) free(Z);
}

void fft_rfft_par(const rfft_plan *p, const double *in, int m_valid, cplx *out, par *pool) {
    int m = p->m;
    cplx *z = malloc(sizeof(cplx) * (size_t)m);
    cplx *Z = malloc(sizeof(cplx) * (size_t)m);
    rfft_pack(in, p->n, m_valid, z);
    fft_forward_par(p->half, z, Z, pool);
    rfft_unpack(p, Z, out);
    free(z);
    free(Z);
}
