/* Native SoftDemod FP32 forward pass on macOS using Apple Accelerate (cblas_sgemm -> AMX).
 *
 * Parses initializers directly from the ONNX graph written by tools/export_onnx.py (_export_soft_demod_ort).
 * If the ONNX file uses any unsupported op or layout (e.g. ConvDemod, GRUDemod, or --fused-layernorm), or on
 * non-Apple platforms, nn_accel_open() returns NULL so ft8nn_open() falls back to ONNX Runtime.
 */
#include "internal.h"

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LAYERS 8
#define MAX_CONVK 9

typedef struct {
    float ln1_eps, *ln1_w, *ln1_b;
    float *wqkv, *bqkv;     /* layers 0 .. n_layers - 2: wqkv is (d, 3*d), bqkv is (3*d) */
    float *wq, *wkv;        /* final layer: wq is (d, d), wkv is [wk | wv] (d, 2*d), bqkv is (3*d) */
    float *wo, *bo;         /* (d, d), (d) */
    float ln2_eps, *ln2_w, *ln2_b;
    float *w1, *b1;         /* (d, ff), (ff) */
    float *w2, *b2;         /* (ff, d), (d) */
} tf_layer;

struct nn_accel {
    int d, n_layers, heads, ff, conv_k;
    float *inp_w;           /* (NFEAT, d) */
    float *inp_bp;          /* (NN, d) */
    float *c1_w[MAX_CONVK], *c1_b;
    float *c2_w[MAX_CONVK], *c2_b;
    tf_layer lyr[MAX_LAYERS];
    float final_ln_eps, *final_ln_w, *final_ln_b;
    float *out_w, *out_b;   /* (d, 3), (3) */
};

/* ---- Minimal ONNX protobuf reader (ModelProto -> GraphProto -> NodeProto / TensorProto) */

typedef struct {
    char name[64];
    int dt, ndims;
    int64_t dims[8];
    const uint8_t *raw;
    size_t raw_len;
    const float *fdata;
    size_t n_fdata;
} pb_tensor;

static int pb_varint(const uint8_t **p, const uint8_t *end, uint64_t *v) {
    uint64_t r = 0;
    int sh = 0;
    const uint8_t *q = *p;
    while (q < end && sh < 64) {
        uint8_t b = *q++;
        r |= (uint64_t)(b & 0x7fu) << sh;
        if (!(b & 0x80u)) {
            *p = q;
            *v = r;
            return 0;
        }
        sh += 7;
    }
    return -1;
}

static int pb_skip(const uint8_t **p, const uint8_t *end, int wt) {
    uint64_t v;
    switch (wt) {
    case 0: return pb_varint(p, end, &v);
    case 1:
        if ((size_t)(end - *p) < 8) return -1;
        *p += 8;
        return 0;
    case 2:
        if (pb_varint(p, end, &v) || v > (size_t)(end - *p)) return -1;
        *p += v;
        return 0;
    case 5:
        if ((size_t)(end - *p) < 4) return -1;
        *p += 4;
        return 0;
    default: return -1;
    }
}

static int parse_tensor(const uint8_t *p, const uint8_t *end, pb_tensor *t) {
    memset(t, 0, sizeof *t);
    while (p < end) {
        uint64_t tag, v;
        if (pb_varint(&p, end, &tag)) return -1;
        int fn = (int)(tag >> 3), wt = (int)(tag & 7);
        if (fn == 1 && wt == 0) {
            if (pb_varint(&p, end, &v)) return -1;
            if (t->ndims < 8) t->dims[t->ndims++] = (int64_t)v;
        } else if (fn == 1 && wt == 2) {
            if (pb_varint(&p, end, &v) || v > (size_t)(end - p)) return -1;
            const uint8_t *pe = p + v;
            while (p < pe) {
                uint64_t dv;
                if (pb_varint(&p, pe, &dv)) return -1;
                if (t->ndims < 8) t->dims[t->ndims++] = (int64_t)dv;
            }
        } else if (fn == 2 && wt == 0) {
            if (pb_varint(&p, end, &v)) return -1;
            t->dt = (int)v;
        } else if (fn == 4 && wt == 2) {
            if (pb_varint(&p, end, &v) || v > (size_t)(end - p) || (v & 3)) return -1;
            t->fdata = (const float *)p;
            t->n_fdata = (size_t)(v / 4);
            p += v;
        } else if (fn == 8 && wt == 2) {
            if (pb_varint(&p, end, &v) || v > (size_t)(end - p)) return -1;
            size_t n = v < sizeof(t->name) - 1 ? (size_t)v : sizeof(t->name) - 1;
            memcpy(t->name, p, n);
            t->name[n] = '\0';
            p += v;
        } else if (fn == 9 && wt == 2) {
            if (pb_varint(&p, end, &v) || v > (size_t)(end - p)) return -1;
            t->raw = p;
            t->raw_len = (size_t)v;
            p += v;
        } else {
            if (pb_skip(&p, end, wt)) return -1;
        }
    }
    return 0;
}

static int parse_attr_heads(const uint8_t *p, const uint8_t *end, int *heads_out) {
    char aname[32] = {0};
    int64_t ival = -1;
    while (p < end) {
        uint64_t tag, v;
        if (pb_varint(&p, end, &tag)) return -1;
        int fn = (int)(tag >> 3), wt = (int)(tag & 7);
        if (fn == 1 && wt == 2) {
            if (pb_varint(&p, end, &v) || v > (size_t)(end - p)) return -1;
            size_t n = v < sizeof(aname) - 1 ? (size_t)v : sizeof(aname) - 1;
            memcpy(aname, p, n);
            p += v;
        } else if (fn == 3 && wt == 0) {
            if (pb_varint(&p, end, &v)) return -1;
            ival = (int64_t)v;
        } else {
            if (pb_skip(&p, end, wt)) return -1;
        }
    }
    if (!strcmp(aname, "num_heads") && ival > 0) *heads_out = (int)ival;
    return 0;
}

static int supported_op(const char *op) {
    static const char *const ops[] = {
        "MatMul", "Add", "Slice", "Pad", "Concat", "Reshape",
        "ReduceMean", "Sub", "Mul", "Sqrt", "Div",
        "BiasGelu", "Attention", "MultiHeadAttention", NULL,
    };
    for (int i = 0; ops[i]; i++)
        if (!strcmp(op, ops[i])) return 1;
    return 0;
}

static int parse_node(const uint8_t *p, const uint8_t *end, int *heads) {
    char op[40] = {0};
    int nh = -1;
    while (p < end) {
        uint64_t tag, v;
        if (pb_varint(&p, end, &tag)) return -1;
        int fn = (int)(tag >> 3), wt = (int)(tag & 7);
        if (fn == 4 && wt == 2) {
            if (pb_varint(&p, end, &v) || v > (size_t)(end - p)) return -1;
            size_t n = v < sizeof(op) - 1 ? (size_t)v : sizeof(op) - 1;
            memcpy(op, p, n);
            p += v;
        } else if (fn == 5 && wt == 2) {
            if (pb_varint(&p, end, &v) || v > (size_t)(end - p)) return -1;
            if (parse_attr_heads(p, p + v, &nh)) return -1;
            p += v;
        } else {
            if (pb_skip(&p, end, wt)) return -1;
        }
    }
    if (!supported_op(op)) return -1;
    if ((!strcmp(op, "Attention") || !strcmp(op, "MultiHeadAttention")) && nh > 0) {
        if (*heads > 0 && *heads != nh) return -1;
        *heads = nh;
    }
    return 0;
}

static const pb_tensor *find_t(const pb_tensor *ts, int nt, const char *name) {
    for (int i = 0; i < nt; i++)
        if (!strcmp(ts[i].name, name)) return &ts[i];
    return NULL;
}

static float *copy_f32(const pb_tensor *t, int nd_exp, int64_t d0, int64_t d1) {
    if (!t || t->dt != 1) return NULL;
    size_t nel = 1;
    for (int i = 0; i < t->ndims; i++) nel *= (size_t)t->dims[i];
    if (nd_exp == 0 && t->ndims != 0) return NULL;
    if (nd_exp == 1 && (t->ndims != 1 || t->dims[0] != d0)) return NULL;
    if (nd_exp == 2 && (t->ndims != 2 || t->dims[0] != d0 || t->dims[1] != d1)) return NULL;
    if (nd_exp == -2 && nel != (size_t)(d0 * d1)) return NULL;
    float *dst = malloc(sizeof(float) * (nel ? nel : 1));
    if (!dst) return NULL;
    if (t->raw && t->raw_len == nel * sizeof(float)) {
        memcpy(dst, t->raw, t->raw_len);
    } else if (t->fdata && t->n_fdata == nel) {
        memcpy(dst, t->fdata, nel * sizeof(float));
    } else {
        free(dst);
        return NULL;
    }
    return dst;
}

static int read_eps(const pb_tensor *ts, int nt, const char *name, float *out) {
    float *v = copy_f32(find_t(ts, nt, name), 0, 0, 0);
    if (!v) return -1;
    *out = v[0];
    free(v);
    return 0;
}

static int check_i64(const pb_tensor *ts, int nt, const char *name, int64_t expected) {
    const pb_tensor *t = find_t(ts, nt, name);
    if (!t || t->dt != 7 || !t->raw || t->raw_len != 8) return -1;
    int64_t v;
    memcpy(&v, t->raw, 8);
    return v == expected ? 0 : -1;
}

void nn_accel_close(nn_accel *m) {
    if (!m) return;
    free(m->inp_w);
    free(m->inp_bp);
    for (int k = 0; k < MAX_CONVK; k++) {
        free(m->c1_w[k]);
        free(m->c2_w[k]);
    }
    free(m->c1_b);
    free(m->c2_b);
    for (int i = 0; i < MAX_LAYERS; i++) {
        tf_layer *l = &m->lyr[i];
        free(l->ln1_w);
        free(l->ln1_b);
        free(l->wqkv);
        free(l->bqkv);
        free(l->wq);
        free(l->wkv);
        free(l->wo);
        free(l->bo);
        free(l->ln2_w);
        free(l->ln2_b);
        free(l->w1);
        free(l->b1);
        free(l->w2);
        free(l->b2);
    }
    free(m->final_ln_w);
    free(m->final_ln_b);
    free(m->out_w);
    free(m->out_b);
    free(m);
}

nn_accel *nn_accel_open(const char *onnx_path) {
    FILE *f = fopen(onnx_path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END)) {
        fclose(f);
        return NULL;
    }
    long sz = ftell(f);
    if (sz <= 0 || sz > 64 * 1024 * 1024 || fseek(f, 0, SEEK_SET)) {
        fclose(f);
        return NULL;
    }
    uint8_t *buf = malloc((size_t)sz);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);

    const uint8_t *gp = NULL, *gend = NULL;
    const uint8_t *p = buf, *end = buf + sz;
    while (p < end) {
        uint64_t tag, v;
        if (pb_varint(&p, end, &tag)) goto fail_buf;
        int fn = (int)(tag >> 3), wt = (int)(tag & 7);
        if (fn == 7 && wt == 2) {
            if (pb_varint(&p, end, &v) || v > (size_t)(end - p)) goto fail_buf;
            gp = p;
            gend = p + v;
            p += v;
        } else {
            if (pb_skip(&p, end, wt)) goto fail_buf;
        }
    }
    if (!gp) goto fail_buf;

    int cap_t = 160, nt = 0, heads = -1;
    pb_tensor *ts = malloc(sizeof(pb_tensor) * (size_t)cap_t);
    if (!ts) goto fail_buf;
    p = gp;
    while (p < gend) {
        uint64_t tag, v;
        if (pb_varint(&p, gend, &tag)) goto fail_ts;
        int fn = (int)(tag >> 3), wt = (int)(tag & 7);
        if (fn == 1 && wt == 2) {
            if (pb_varint(&p, gend, &v) || v > (size_t)(gend - p)) goto fail_ts;
            if (parse_node(p, p + v, &heads)) goto fail_ts;
            p += v;
        } else if (fn == 5 && wt == 2) {
            if (pb_varint(&p, gend, &v) || v > (size_t)(gend - p)) goto fail_ts;
            if (nt >= cap_t) {
                cap_t *= 2;
                pb_tensor *nts = realloc(ts, sizeof(pb_tensor) * (size_t)cap_t);
                if (!nts) goto fail_ts;
                ts = nts;
            }
            if (parse_tensor(p, p + v, &ts[nt++])) goto fail_ts;
            p += v;
        } else {
            if (pb_skip(&p, gend, wt)) goto fail_ts;
        }
    }

    const pb_tensor *tinp = find_t(ts, nt, "inp_w");
    if (!tinp || tinp->ndims != 2 || tinp->dims[0] != NFEAT) goto fail_ts;
    int d = (int)tinp->dims[1];
    if (d < 8 || d > 512) goto fail_ts;

    nn_accel *m = calloc(1, sizeof *m);
    if (!m) goto fail_ts;
    m->d = d;
    m->inp_w = copy_f32(tinp, 2, NFEAT, d);
    m->inp_bp = copy_f32(find_t(ts, nt, "inp_bp"), -2, NN, d);
    if (!m->inp_w || !m->inp_bp) goto fail_m;

    int conv_k = 0;
    for (int k = 0; k < MAX_CONVK; k++) {
        char nm[32];
        snprintf(nm, sizeof nm, "c1_t%d_w", k);
        if (!find_t(ts, nt, nm)) break;
        conv_k = k + 1;
    }
    if (conv_k < 1 || (conv_k % 2) != 1) goto fail_m;
    m->conv_k = conv_k;
    for (int k = 0; k < conv_k; k++) {
        char n1[32], n2[32];
        snprintf(n1, sizeof n1, "c1_t%d_w", k);
        snprintf(n2, sizeof n2, "c2_t%d_w", k);
        m->c1_w[k] = copy_f32(find_t(ts, nt, n1), 2, d, d);
        m->c2_w[k] = copy_f32(find_t(ts, nt, n2), 2, d, d);
        if (!m->c1_w[k] || !m->c2_w[k]) goto fail_m;
        if (k != conv_k / 2) {
            /* verify untied dilation == 1: slice span is NN - |k - kc| */
            char ns0[32], ns1[32];
            snprintf(ns0, sizeof ns0, "c1_t%d_s0", k);
            snprintf(ns1, sizeof ns1, "c1_t%d_s1", k);
            int sh = k - conv_k / 2;
            int64_t s0 = sh < 0 ? 0 : sh, s1 = sh < 0 ? NN + sh : NN;
            if (check_i64(ts, nt, ns0, s0) || check_i64(ts, nt, ns1, s1)) goto fail_m;
        }
    }
    m->c1_b = copy_f32(find_t(ts, nt, "c1_b"), 1, d, 0);
    m->c2_b = copy_f32(find_t(ts, nt, "c2_b"), 1, d, 0);
    if (!m->c1_b || !m->c2_b) goto fail_m;

    int nl = 0;
    for (int i = 0; i < MAX_LAYERS; i++) {
        char nm[32];
        snprintf(nm, sizeof nm, "l%d_ln1_w", i);
        if (!find_t(ts, nt, nm)) break;
        nl = i + 1;
    }
    m->n_layers = nl;
    if (nl > 0) {
        if (heads <= 0 || (d % heads) != 0) goto fail_m;
        m->heads = heads;
        /* verify final-layer DATA_POS slice matches FT8 [7..36) + [43..72) */
        if (check_i64(ts, nt, "l_last_uq_r0_s0", 7) || check_i64(ts, nt, "l_last_uq_r0_s1", 36) ||
            check_i64(ts, nt, "l_last_uq_r1_s0", 43) || check_i64(ts, nt, "l_last_uq_r1_s1", 72))
            goto fail_m;
    }
    for (int i = 0; i < nl; i++) {
        tf_layer *l = &m->lyr[i];
        char nm[32];
        snprintf(nm, sizeof nm, "l%d_ln1_eps", i);
        if (read_eps(ts, nt, nm, &l->ln1_eps)) goto fail_m;
        snprintf(nm, sizeof nm, "l%d_ln1_w", i);
        l->ln1_w = copy_f32(find_t(ts, nt, nm), 1, d, 0);
        snprintf(nm, sizeof nm, "l%d_ln1_b", i);
        l->ln1_b = copy_f32(find_t(ts, nt, nm), 1, d, 0);
        snprintf(nm, sizeof nm, "l%d_bqkv", i);
        l->bqkv = copy_f32(find_t(ts, nt, nm), 1, 3 * d, 0);
        if (!l->ln1_w || !l->ln1_b || !l->bqkv) goto fail_m;
        if (i < nl - 1) {
            snprintf(nm, sizeof nm, "l%d_wqkv", i);
            l->wqkv = copy_f32(find_t(ts, nt, nm), 2, d, 3 * d);
            if (!l->wqkv) goto fail_m;
        } else {
            snprintf(nm, sizeof nm, "l%d_wq", i);
            l->wq = copy_f32(find_t(ts, nt, nm), 2, d, d);
            snprintf(nm, sizeof nm, "l%d_wk", i);
            float *wk = copy_f32(find_t(ts, nt, nm), 2, d, d);
            snprintf(nm, sizeof nm, "l%d_wv", i);
            float *wv = copy_f32(find_t(ts, nt, nm), 2, d, d);
            l->wkv = malloc(sizeof(float) * (size_t)(d * 2 * d));
            if (!l->wq || !wk || !wv || !l->wkv) {
                free(wk);
                free(wv);
                goto fail_m;
            }
            for (int r = 0; r < d; r++) {
                memcpy(l->wkv + (size_t)r * 2 * d, wk + (size_t)r * d, sizeof(float) * (size_t)d);
                memcpy(l->wkv + (size_t)r * 2 * d + d, wv + (size_t)r * d, sizeof(float) * (size_t)d);
            }
            free(wk);
            free(wv);
        }
        snprintf(nm, sizeof nm, "l%d_wo", i);
        l->wo = copy_f32(find_t(ts, nt, nm), 2, d, d);
        snprintf(nm, sizeof nm, "l%d_bo", i);
        l->bo = copy_f32(find_t(ts, nt, nm), 1, d, 0);
        snprintf(nm, sizeof nm, "l%d_ln2_eps", i);
        if (read_eps(ts, nt, nm, &l->ln2_eps)) goto fail_m;
        snprintf(nm, sizeof nm, "l%d_ln2_w", i);
        l->ln2_w = copy_f32(find_t(ts, nt, nm), 1, d, 0);
        snprintf(nm, sizeof nm, "l%d_ln2_b", i);
        l->ln2_b = copy_f32(find_t(ts, nt, nm), 1, d, 0);
        snprintf(nm, sizeof nm, "l%d_w1", i);
        const pb_tensor *tw1 = find_t(ts, nt, nm);
        if (!tw1 || tw1->ndims != 2 || tw1->dims[0] != d) goto fail_m;
        int ff = (int)tw1->dims[1];
        if (ff < 8 || ff > 2048 || (m->ff > 0 && m->ff != ff)) goto fail_m;
        m->ff = ff;
        l->w1 = copy_f32(tw1, 2, d, ff);
        snprintf(nm, sizeof nm, "l%d_b1", i);
        l->b1 = copy_f32(find_t(ts, nt, nm), 1, ff, 0);
        snprintf(nm, sizeof nm, "l%d_w2", i);
        l->w2 = copy_f32(find_t(ts, nt, nm), 2, ff, d);
        snprintf(nm, sizeof nm, "l%d_b2", i);
        l->b2 = copy_f32(find_t(ts, nt, nm), 1, d, 0);
        if (!l->wo || !l->bo || !l->ln2_w || !l->ln2_b || !l->w1 || !l->b1 || !l->w2 || !l->b2) goto fail_m;
    }

    if (read_eps(ts, nt, "final_ln_eps", &m->final_ln_eps)) goto fail_m;
    m->final_ln_w = copy_f32(find_t(ts, nt, "final_ln_w"), 1, d, 0);
    m->final_ln_b = copy_f32(find_t(ts, nt, "final_ln_b"), 1, d, 0);
    m->out_w = copy_f32(find_t(ts, nt, "out_w"), 2, d, 3);
    m->out_b = copy_f32(find_t(ts, nt, "out_b"), 1, 3, 0);
    if (!m->final_ln_w || !m->final_ln_b || !m->out_w || !m->out_b) goto fail_m;

    free(ts);
    free(buf);
    return m;

fail_m:
    nn_accel_close(m);
fail_ts:
    free(ts);
fail_buf:
    free(buf);
    return NULL;
}

/* ---- SIMD helpers (explicit 4-wide ARM NEON on Apple Silicon, scalar fallback on x86_64) */

static inline float fast_expf_neg(float x) {
    if (x < -87.0f) x = -87.0f;
    float kf = nearbyintf(x * 1.4426950408889634f);
    float r = x - kf * 0.693145751953125f - kf * 1.428606765330187e-06f;
    float p = 1.98756915e-4f;
    p = p * r + 1.39819995e-3f;
    p = p * r + 8.33345190e-3f;
    p = p * r + 4.16657958e-2f;
    p = p * r + 1.66666657e-1f;
    p = p * r + 5.00000000e-1f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    int32_t ki = (int32_t)kf;
    uint32_t bits = (uint32_t)((ki + 127) << 23);
    float sc;
    memcpy(&sc, &bits, 4);
    return p * sc;
}

static inline float fast_geluf(float x) {
    float z = fabsf(x) * 0.7071067811865475f;
    if (z > 4.0f) z = 4.0f;
    float t = 1.0f / (1.0f + 0.3275911f * z);
    float poly = (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t);
    float erfc_z = poly * fast_expf_neg(-z * z);
    float erf_x = copysignf(1.0f - erfc_z, x);
    return 0.5f * x * (1.0f + erf_x);
}

#ifdef __ARM_NEON
static inline float32x4_t neon_expf_neg(float32x4_t x) {
    x = vmaxq_f32(x, vdupq_n_f32(-87.0f));
    float32x4_t kf = vrndnq_f32(vmulq_n_f32(x, 1.4426950408889634f));
    float32x4_t r = vfmsq_f32(x, kf, vdupq_n_f32(0.693145751953125f));
    r = vfmsq_f32(r, kf, vdupq_n_f32(1.428606765330187e-06f));
    float32x4_t p = vdupq_n_f32(1.98756915e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.39819995e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(8.33345190e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.16657958e-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.66666657e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(5.00000000e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    int32x4_t ki = vaddq_s32(vcvtq_s32_f32(kf), vdupq_n_s32(127));
    float32x4_t sc = vreinterpretq_f32_s32(vshlq_n_s32(ki, 23));
    return vmulq_f32(p, sc);
}

static inline float32x4_t neon_geluf(float32x4_t vx) {
    float32x4_t z = vminq_f32(vmulq_n_f32(vabsq_f32(vx), 0.7071067811865475f), vdupq_n_f32(4.0f));
    float32x4_t den = vfmaq_f32(vdupq_n_f32(1.0f), z, vdupq_n_f32(0.3275911f));
    float32x4_t t = vrecpeq_f32(den);
    t = vmulq_f32(vrecpsq_f32(den, t), t);
    t = vmulq_f32(vrecpsq_f32(den, t), t);
    float32x4_t p = vdupq_n_f32(1.061405429f);
    p = vfmaq_f32(vdupq_n_f32(-1.453152027f), p, t);
    p = vfmaq_f32(vdupq_n_f32(1.421413741f), p, t);
    p = vfmaq_f32(vdupq_n_f32(-0.284496736f), p, t);
    p = vfmaq_f32(vdupq_n_f32(0.254829592f), p, t);
    p = vmulq_f32(p, t);
    float32x4_t erfc_z = vmulq_f32(p, neon_expf_neg(vnegq_f32(vmulq_f32(z, z))));
    float32x4_t a_erf = vsubq_f32(vdupq_n_f32(1.0f), erfc_z);
    uint32x4_t sgn = vandq_u32(vreinterpretq_u32_f32(vx), vdupq_n_u32(0x80000000u));
    float32x4_t erf_x = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(a_erf), sgn));
    return vmulq_n_f32(vmulq_f32(vx, vaddq_f32(vdupq_n_f32(1.0f), erf_x)), 0.5f);
}
#endif

static void bias_gelu_copy(const float * restrict yc, const float * restrict b, int d, float * restrict dst) {
    int j = 0;
#ifdef __ARM_NEON
    for (; j + 4 <= d; j += 4)
        vst1q_f32(dst + j, neon_geluf(vaddq_f32(vld1q_f32(yc + j), vld1q_f32(b + j))));
#endif
    for (; j < d; j++) dst[j] = fast_geluf(yc[j] + b[j]);
}

static void bias_gelu_inplace(float * restrict x, const float * restrict b, int d) {
    int j = 0;
#ifdef __ARM_NEON
    for (; j + 4 <= d; j += 4)
        vst1q_f32(x + j, neon_geluf(vaddq_f32(vld1q_f32(x + j), vld1q_f32(b + j))));
#endif
    for (; j < d; j++) x[j] = fast_geluf(x[j] + b[j]);
}

static void bias_gelu_accum(const float * restrict yc, const float * restrict b, int d, float * restrict dst) {
    int j = 0;
#ifdef __ARM_NEON
    for (; j + 4 <= d; j += 4)
        vst1q_f32(dst + j, vaddq_f32(vld1q_f32(dst + j), neon_geluf(vaddq_f32(vld1q_f32(yc + j), vld1q_f32(b + j)))));
#endif
    for (; j < d; j++) dst[j] += fast_geluf(yc[j] + b[j]);
}

static void layernorm(const float * restrict x, int rows, int d, float eps, const float * restrict w,
                      const float * restrict b, float * restrict out) {
    float inv_d = 1.0f / (float)d;
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * d;
        float *yr = out + (size_t)r * d;
        int j = 0;
        float sum = 0.0f;
#ifdef __ARM_NEON
        float32x4_t vs = vdupq_n_f32(0.0f);
        for (; j + 4 <= d; j += 4) vs = vaddq_f32(vs, vld1q_f32(xr + j));
        sum = vaddvq_f32(vs);
#endif
        for (; j < d; j++) sum += xr[j];
        float mu = sum * inv_d;
        j = 0;
        float vsum = 0.0f;
#ifdef __ARM_NEON
        float32x4_t vmu = vdupq_n_f32(mu), vv = vdupq_n_f32(0.0f);
        for (; j + 4 <= d; j += 4) {
            float32x4_t df = vsubq_f32(vld1q_f32(xr + j), vmu);
            vv = vfmaq_f32(vv, df, df);
        }
        vsum = vaddvq_f32(vv);
#endif
        for (; j < d; j++) {
            float df = xr[j] - mu;
            vsum += df * df;
        }
        float inv = 1.0f / sqrtf(vsum * inv_d + eps);
        j = 0;
#ifdef __ARM_NEON
        float32x4_t vinv = vdupq_n_f32(inv);
        for (; j + 4 <= d; j += 4) {
            float32x4_t nrm = vmulq_f32(vsubq_f32(vld1q_f32(xr + j), vmu), vinv);
            vst1q_f32(yr + j, vfmaq_f32(vld1q_f32(b + j), nrm, vld1q_f32(w + j)));
        }
#endif
        for (; j < d; j++) yr[j] = (xr[j] - mu) * inv * w[j] + b[j];
    }
}

static void softmax_rows(float * restrict s, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *sr = s + (size_t)r * cols;
        int c = 0;
        float mx = sr[0];
#ifdef __ARM_NEON
        float32x4_t vmaxv = vdupq_n_f32(mx);
        for (; c + 4 <= cols; c += 4) vmaxv = vmaxq_f32(vmaxv, vld1q_f32(sr + c));
        mx = vmaxvq_f32(vmaxv);
#endif
        for (; c < cols; c++)
            if (sr[c] > mx) mx = sr[c];
        c = 0;
        float sum = 0.0f;
#ifdef __ARM_NEON
        float32x4_t vmx = vdupq_n_f32(mx), vs = vdupq_n_f32(0.0f);
        for (; c + 4 <= cols; c += 4) {
            float32x4_t v = neon_expf_neg(vsubq_f32(vld1q_f32(sr + c), vmx));
            vst1q_f32(sr + c, v);
            vs = vaddq_f32(vs, v);
        }
        sum = vaddvq_f32(vs);
#endif
        for (; c < cols; c++) {
            float v = fast_expf_neg(sr[c] - mx);
            sr[c] = v;
            sum += v;
        }
        float inv = 1.0f / sum;
        c = 0;
#ifdef __ARM_NEON
        float32x4_t vinv = vdupq_n_f32(inv);
        for (; c + 4 <= cols; c += 4) vst1q_f32(sr + c, vmulq_f32(vld1q_f32(sr + c), vinv));
#endif
        for (; c < cols; c++) sr[c] *= inv;
    }
}

static void add_bias_rows(float * restrict x, int rows, int cols, const float * restrict b) {
    for (int r = 0; r < rows; r++) {
        float *xr = x + (size_t)r * cols;
        int c = 0;
#ifdef __ARM_NEON
        for (; c + 4 <= cols; c += 4) vst1q_f32(xr + c, vaddq_f32(vld1q_f32(xr + c), vld1q_f32(b + c)));
#endif
        for (; c < cols; c++) xr[c] += b[c];
    }
}

size_t nn_accel_scratch_floats(const nn_accel *m) {
    int d = m->d, P = m->conv_k / 2, S = NN + P;
    size_t pad_rows = (size_t)NN_ACCEL_MAXB * S + P;
    int wide = 3 * d > m->ff ? 3 * d : m->ff;
    return 3 * pad_rows * (size_t)d + (size_t)NN_ACCEL_MAXB * NN * (size_t)wide + (size_t)(NN * NN);
}

void nn_accel_run(const nn_accel *m, const float *feat, int B, double *llr, float *scratch) {
    if (B <= 0) return;
    int d = m->d, P = m->conv_k / 2, S = NN + P;
    size_t pad_rows = (size_t)B * S + P;
    int wide = 3 * d > m->ff ? 3 * d : m->ff;
    float *buf_a = scratch;
    float *buf_b = buf_a + pad_rows * (size_t)d;
    float *buf_c = buf_b + pad_rows * (size_t)d;
    float *buf_qkv = buf_c + pad_rows * (size_t)d;
    float *buf_scores = buf_qkv + (size_t)B * NN * (size_t)wide;

    /* 1. h0 = feat @ inp_w + inp_bp */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B * NN, d, NFEAT, 1.0f, feat, NFEAT, m->inp_w, d, 0.0f,
                buf_a, d);
    for (int b = 0; b < B; b++) {
        float *hb = buf_a + (size_t)b * NN * d;
        for (int i = 0; i < NN * d; i++) hb[i] += m->inp_bp[i];
    }

    /* 2. Conv1d residual block: hc = h0 + gelu(conv2(gelu(conv1(h0)))) */
    memset(buf_b, 0, sizeof(float) * pad_rows * (size_t)d);
    for (int b = 0; b < B; b++)
        memcpy(buf_b + (size_t)(P + b * S) * d, buf_a + (size_t)b * NN * d, sizeof(float) * (size_t)(NN * d));

    int m_conv = (B - 1) * S + NN;
    for (int k = 0; k < m->conv_k; k++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m_conv, d, d, 1.0f, buf_b + (size_t)k * d, d,
                    m->c1_w[k], d, k == 0 ? 0.0f : 1.0f, buf_c, d);
    }
    const float *c1_b = m->c1_b, *c2_b = m->c2_b;
    for (int b = 0; b < B; b++)
        for (int s = 0; s < NN; s++)
            bias_gelu_copy(buf_c + (size_t)(b * S + s) * d, c1_b, d, buf_b + (size_t)(P + b * S + s) * d);
    for (int k = 0; k < m->conv_k; k++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m_conv, d, d, 1.0f, buf_b + (size_t)k * d, d,
                    m->c2_w[k], d, k == 0 ? 0.0f : 1.0f, buf_c, d);
    }
    for (int b = 0; b < B; b++)
        for (int s = 0; s < NN; s++)
            bias_gelu_accum(buf_c + (size_t)(b * S + s) * d, c2_b, d, buf_a + (size_t)(b * NN + s) * d);

    /* 3. Pre-LN Transformer Encoder layers (final layer runs Q and FFN on the 58 DATA_POS tokens only) */
    for (int li = 0; li < m->n_layers; li++) {
        const tf_layer *l = &m->lyr[li];
        int is_last = (li == m->n_layers - 1);
        int l_cur = is_last ? ND : NN;
        int m_cur = B * l_cur;
        int hd = d / m->heads;
        float scale = 1.0f / sqrtf((float)hd);

        layernorm(buf_a, B * NN, d, l->ln1_eps, l->ln1_w, l->ln1_b, buf_b);
        if (!is_last) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B * NN, 3 * d, d, 1.0f, buf_b, d, l->wqkv, 3 * d,
                        0.0f, buf_qkv, 3 * d);
            add_bias_rows(buf_qkv, B * NN, 3 * d, l->bqkv);
            for (int b = 0; b < B; b++) {
                const float *qkv_b = buf_qkv + (size_t)b * NN * 3 * d;
                float *att_b = buf_c + (size_t)b * NN * d;
                for (int h = 0; h < m->heads; h++) {
                    const float *q = qkv_b + h * hd;
                    const float *k = q + d;
                    const float *v = q + 2 * d;
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, NN, NN, hd, scale, q, 3 * d, k, 3 * d, 0.0f,
                                buf_scores, NN);
                    softmax_rows(buf_scores, NN, NN);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, NN, hd, NN, 1.0f, buf_scores, NN, v, 3 * d,
                                0.0f, att_b + h * hd, d);
                }
            }
        } else {
            float *q_buf = buf_qkv;
            float *kv_buf = buf_qkv + (size_t)B * ND * d;
            for (int b = 0; b < B; b++) {
                memcpy(buf_c + (size_t)(b * ND) * d, buf_b + (size_t)(b * NN + 7) * d, sizeof(float) * 29 * d);
                memcpy(buf_c + (size_t)(b * ND + 29) * d, buf_b + (size_t)(b * NN + 43) * d, sizeof(float) * 29 * d);
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B * ND, d, d, 1.0f, buf_c, d, l->wq, d, 0.0f, q_buf,
                        d);
            add_bias_rows(q_buf, B * ND, d, l->bqkv);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B * NN, 2 * d, d, 1.0f, buf_b, d, l->wkv, 2 * d,
                        0.0f, kv_buf, 2 * d);
            add_bias_rows(kv_buf, B * NN, 2 * d, l->bqkv + d);
            for (int b = 0; b < B; b++) {
                memmove(buf_a + (size_t)(b * ND) * d, buf_a + (size_t)(b * NN + 7) * d, sizeof(float) * 29 * d);
                memmove(buf_a + (size_t)(b * ND + 29) * d, buf_a + (size_t)(b * NN + 43) * d, sizeof(float) * 29 * d);
            }
            for (int b = 0; b < B; b++) {
                const float *qb = q_buf + (size_t)b * ND * d;
                const float *kvb = kv_buf + (size_t)b * NN * 2 * d;
                float *att_b = buf_c + (size_t)b * ND * d;
                for (int h = 0; h < m->heads; h++) {
                    const float *q = qb + h * hd;
                    const float *k = kvb + h * hd;
                    const float *v = k + d;
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ND, NN, hd, scale, q, d, k, 2 * d, 0.0f,
                                buf_scores, NN);
                    softmax_rows(buf_scores, ND, NN);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ND, hd, NN, 1.0f, buf_scores, NN, v, 2 * d,
                                0.0f, att_b + h * hd, d);
                }
            }
        }

        add_bias_rows(buf_a, m_cur, d, l->bo);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m_cur, d, d, 1.0f, buf_c, d, l->wo, d, 1.0f, buf_a, d);

        layernorm(buf_a, m_cur, d, l->ln2_eps, l->ln2_w, l->ln2_b, buf_b);
        int ff = m->ff;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m_cur, ff, d, 1.0f, buf_b, d, l->w1, ff, 0.0f, buf_qkv,
                    ff);
        const float *b1 = l->b1;
        for (int r = 0; r < m_cur; r++) bias_gelu_inplace(buf_qkv + (size_t)r * ff, b1, ff);
        add_bias_rows(buf_a, m_cur, d, l->b2);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m_cur, d, ff, 1.0f, buf_qkv, ff, l->w2, d, 1.0f, buf_a,
                    d);
    }

    if (m->n_layers == 0) {
        for (int b = 0; b < B; b++) {
            memmove(buf_a + (size_t)(b * ND) * d, buf_a + (size_t)(b * NN + 7) * d, sizeof(float) * 29 * d);
            memmove(buf_a + (size_t)(b * ND + 29) * d, buf_a + (size_t)(b * NN + 43) * d, sizeof(float) * 29 * d);
        }
    }

    /* 4. Final LayerNorm + output Linear (d -> 3) */
    layernorm(buf_a, B * ND, d, m->final_ln_eps, m->final_ln_w, m->final_ln_b, buf_b);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B * ND, 3, d, 1.0f, buf_b, d, m->out_w, 3, 0.0f, buf_c, 3);
    float ob0 = m->out_b[0], ob1 = m->out_b[1], ob2 = m->out_b[2];
    for (int r = 0; r < B * ND; r++) {
        llr[r * 3 + 0] = (double)(buf_c[r * 3 + 0] + ob0);
        llr[r * 3 + 1] = (double)(buf_c[r * 3 + 1] + ob1);
        llr[r * 3 + 2] = (double)(buf_c[r * 3 + 2] + ob2);
    }
}

#else
#include <stddef.h>
nn_accel *nn_accel_open(const char *onnx_path) {
    (void)onnx_path;
    return NULL;
}
void nn_accel_close(nn_accel *m) { (void)m; }
size_t nn_accel_scratch_floats(const nn_accel *m) {
    (void)m;
    return 0;
}
void nn_accel_run(const nn_accel *m, const float *feat, int b, double *llr, float *scratch) {
    (void)m;
    (void)feat;
    (void)b;
    (void)llr;
    (void)scratch;
}
#endif
