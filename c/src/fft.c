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

/* out[0..len) = DFT_len of in[0], in[stride], ...; fi = index of this level's radix in fac. */
static void work(const fft_plan *p, cplx *out, const cplx *in, int stride, int fi, int len) {
    if (len == 1) {
        out[0] = in[0];
        return;
    }
    int r = p->fac[fi], m = len / r;
    for (int q = 0; q < r; q++) work(p, out + q * m, in + q * stride, stride * r, fi + 1, m);
    int ts = p->n / len; /* twiddle stride for this level */
    const cplx *tw = p->tw;
    if (r == 2) {
        for (int k = 0; k < m; k++) {
            cplx a = out[k], b = out[k + m] * tw[k * ts];
            out[k] = a + b;
            out[k + m] = a - b;
        }
    } else if (r == 4) {
        for (int k = 0; k < m; k++) {
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
        for (int k = 0; k < m; k++) {
            for (int q = 0; q < r; q++) t[q] = out[k + q * m] * tw[q * k * ts];
            for (int j = 0; j < r; j++) {
                cplx s = t[0];
                for (int q = 1; q < r; q++) s += t[q] * tw[((j * q) % r) * rs];
                out[k + j * m] = s;
            }
        }
    }
}

void fft_forward(const fft_plan *p, const cplx *in, cplx *out) { work(p, out, in, 1, 0, p->n); }

void fft_inverse(const fft_plan *p, const cplx *in, cplx *out) {
    /* ifft(x) = conj(fft(conj(x))) */
    int n = p->n;
    cplx *c = malloc(sizeof(cplx) * n);
    for (int i = 0; i < n; i++) c[i] = conj(in[i]);
    work(p, out, c, 1, 0, n);
    for (int i = 0; i < n; i++) out[i] = conj(out[i]);
    free(c);
}
