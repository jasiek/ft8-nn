/* Slot decoder (port of ft8nn/decoder.py decode_slot with nn_metric.make_llr_fn(also_classical=True)).
 *
 * Per pass: candidates -> fine_sync2 (2 hypotheses each) -> soft metrics [NN, classical nsym 1/2/3]
 * -> BP -> OSD-2 (NN metric only, gated; then on summed BP posteriors) -> context a-priori (decodes 30/60 s earlier,
 * ft8nn_decode_at only) -> CQ a-priori (NN metric only, gated) -> CRC.
 * Decodes are subtracted from the audio after every pass; on the final residual the co-channel retry, then (with
 * context) the repeat stage.
 */
#include <math.h>
#include <onnxruntime_c_api.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* One-time init and clocks: POSIX here; the _WIN32 branch is for native Windows builds without pthreads (built with
 * MinGW-w64 gcc in CI, .github/workflows/windows.yml). */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#undef near /* 16-bit leftovers defined empty by <windows.h>; `near` is a variable name below */
#undef far
typedef INIT_ONCE once_t;
#define ONCE_INIT INIT_ONCE_STATIC_INIT
static BOOL CALLBACK once_cb(PINIT_ONCE o, PVOID fn, PVOID *ctx) {
    ((void (*)(void))fn)();
    return TRUE;
}
static void run_once(once_t *o, void (*fn)(void)) { InitOnceExecuteOnce(o, once_cb, (PVOID)fn, NULL); }
static double tnow(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static double tcpu(void) { /* process CPU time, all threads */
    FILETIME c, e, k, u;
    GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
    return 1e-7 * ((double)(((uint64_t)k.dwHighDateTime << 32) | k.dwLowDateTime) +
                   (double)(((uint64_t)u.dwHighDateTime << 32) | u.dwLowDateTime));
}
static double tthr(void) { /* this thread's CPU time */
    FILETIME c, e, k, u;
    GetThreadTimes(GetCurrentThread(), &c, &e, &k, &u);
    return 1e-7 * ((double)(((uint64_t)k.dwHighDateTime << 32) | k.dwLowDateTime) +
                   (double)(((uint64_t)u.dwHighDateTime << 32) | u.dwLowDateTime));
}
#else
#include <pthread.h>
typedef pthread_once_t once_t;
#define ONCE_INIT PTHREAD_ONCE_INIT
static void run_once(once_t *o, void (*fn)(void)) { pthread_once(o, fn); }
static double tnow(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static double tcpu(void) { /* process CPU time, all threads */
    struct timespec t;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static double tthr(void) { /* this thread's CPU time */
    struct timespec t;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
#endif

static int prof = -1;

#include "../include/ft8nn.h"
#include "internal.h"

#define MAX_CAND 600
#define EARLY_DT (-2.5) /* early-DT candidate track down to this DT (clocks ~2 s early); jt9 searches to -2.5 */
#define EARLY_MIN 3.0  /* its score threshold (2 of 3 Costas arrays in the slot -> noisier score) */
#define N_HYP 2
#define PASSES 4 /* decode-subtract passes (decoder.decode_slot passes) */
#define OSD_ORDER 2
#define OSD_MAX_HARD 40
#define OSD_GATE 40.0
#define OSD_SNAPS 2 /* OSD on BP posterior sums after iterations 1..OSD_SNAPS when OSD fails (decoder.osd_snaps) */
#define AP_GATE 44.0
/* context a-priori (decoder._try_ctx): decodes 30 and 60 s earlier within CTX_DF Hz and CTX_DDT s of a hypothesis */
#define CTX_GATE 48.0
#define CTX_DF 10.0
#define CTX_DDT 0.5
#define CTX_HIST 4 /* slots of history kept per context */
/* repeat stage (decoder._repeat_stage): context message tested as a whole known codeword on the residual */
#define REPEAT_RHO 0.40
#define REPEAT_MARGIN 0.15
#define REPEAT_CLIP 2.0
#define REPEAT_SOLO 0.35 /* runner-up rho when a sender has no sendable competitor (decoder.REPEAT_SOLO) */
#define MAXGRID4 32400
#define NTOKENS_CQ 2063592u /* message.NTOKENS: call fields 2..NTOKENS-1 are CQ / CQ nnn / CQ ABCD */
/* co-channel retry (decoder._cochannel_stage; decoder.COCHANNEL, COCHANNEL_GRID, COCHANNEL_REFINE): at each decode's
 * cell, data-symbol sync over df +-CC_SPAN Hz x dt +-CC_TFAR baseband samples; the CC_NEAR best local maxima within
 * +-CC_TNEAR samples and the CC_FAR best beyond, scoring >= CC_MIN, each as 3 x 3 hypotheses (+-CC_RDF Hz, +-CC_RDT) */
#define CC_NEAR 2
#define CC_FAR 1
#define CC_SPAN 8.0
#define CC_TNEAR 24
#define CC_TFAR 50
#define CC_MIN 3.0
#define CC_GF 0.5 /* grid steps: Hz, baseband samples */
#define CC_GT 2
#define CC_RDF 0.4
#define CC_RDT 1
#define CC_NF 37 /* -(CC_SPAN + 2 CC_GF) .. CC_SPAN + 2 CC_GF */
#define CC_NT 53 /* -(CC_TFAR + CC_GT) .. CC_TFAR + CC_GT */
#define CC_MAXH ((CC_NEAR + CC_FAR) * 9)
#define DEC_CAP 512 /* decodes per slot */
#define ERASE_GATE 15.0 /* burst-erasure retry for confident NN metrics that still fail */
/* osd_er ghost test (decoder._is_ghost): neighbour within GHOST_MAX_DF Hz whose tones, shifted by whole symbols and
 * tones, equal the decode's data tones on >= GHOST_MATCH of >= GHOST_MIN_SYMS comparable symbols */
#define GHOST_MAX_DF 50.0
#define GHOST_MIN_SYMS 20
#define GHOST_MATCH 0.4
#define ERASE_OSD_DMAX 3.0 /* wide-window OSD-0 erasure stage: max distance on the kept bits (decoder.erase_osd_dmax) */
#define NMET 4 /* NN + classical nsym 1, 2, 3 */
#define NN_BATCH 128
#define BASE_MEDIAN 1 /* candidate score baseline: 1 = 1.77 * median of all cells (A/B 2026-11-04: real +281), 0 = old pct40 of per-freq max */
#define FMAX 3100.0 /* candidate search top (f0, Hz); jt9 decodes up to ~3050 (A/B 2026-11-04: validation run 2 +38) */
/* later passes skip a candidate (same bin, lag, track) tried in an earlier pass when no decode subtracted since lies
 * within REUSE_DF Hz (decoder.decode_slot reuse_df); 0 = off */
#define REUSE_DF 62.5
/* decoder.decode_slot dt_recenter: when the median DT of the decodes in hand (this slot's plus the context's from
 * 30/60 s earlier) is at least DT_RECENTER s from 0, the receiver's own clock or audio stream is off by that much
 * and every signal sits at the same wrong DT; later passes then search a window shifted by that median */
#define DT_RECENTER 0.8
#define DT_RECENTER_MIN 6   /* decodes needed before their median is trusted (decoder.DT_RECENTER_MIN) */
#define DT_RECENTER_MAX 2.5 /* largest shift applied, so the window cannot walk away (decoder.DT_RECENTER_MAX) */
#define TRIED_LAG0 160 /* lag offset in the tried table: lags -160..159 frames (early track -63, shift up to -62) */
#define TRIED_NL 320
#define TRIED_NB ((int)(FMAX / 3.125) + 1)
#define NOISE_BIN (3200.0 / 1000.0)
#define SNR_OFFSET 0.23

typedef struct {
    uint8_t bits77[77];
    double f0, dt, snr;
    int method, metric, pass;
    char text[48]; /* unpacked once the decode is final (ft8nn_decode_stream reports it then) */
    uint8_t final;
} dec;

struct ft8nn_ctx {
    const OrtApi *api;
    OrtEnv *env;
    OrtSession *sess;
    OrtMemoryInfo *mem;
    par *pool; /* decode threads (ft8nn_open n_threads) */
    hash_table *ht;
    struct {
        int64_t t; /* slot start, s; INT64_MIN = empty */
        int n;
        dec *d;
    } hist[CTX_HIST]; /* this context's own decodes of recent slots (ft8nn_decode_at) */
    int nv_bb;        /* NN features: baseband samples that carry audio during the current decode, 0 = all */
};

static once_t init_once = ONCE_INIT;
static void global_init(void) {
    ft8_init();
    ldpc_init();
    frontend_init();
    subtract_init();
}

static int ort_check(const OrtApi *api, OrtStatus *st) {
    if (!st) return 0;
    fprintf(stderr, "ft8nn: onnxruntime: %s\n", api->GetErrorMessage(st));
    api->ReleaseStatus(st);
    return -1;
}

ft8nn_ctx *ft8nn_open(const char *onnx_path, int n_threads) {
    run_once(&init_once, global_init);
    ft8nn_ctx *c = calloc(1, sizeof *c);
    c->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    const OrtApi *api = c->api;
    OrtSessionOptions *so = NULL;
#ifdef _WIN32 /* ONNX Runtime takes wide-character paths on Windows */
    wchar_t model[4096];
    if (!MultiByteToWideChar(CP_UTF8, 0, onnx_path, -1, model, 4096)) {
        free(c);
        return NULL;
    }
#else
    const char *model = onnx_path;
#endif
    if (ort_check(api, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ft8nn", &c->env)) ||
        ort_check(api, api->CreateSessionOptions(&so)) ||
        /* one ONNX Runtime thread per Run: our own threads call Run concurrently on chunks of hypotheses */
        ort_check(api, api->SetIntraOpNumThreads(so, 1)) || ort_check(api, api->SetInterOpNumThreads(so, 1)) ||
        ort_check(api, api->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL)) ||
        ort_check(api, api->CreateSession(c->env, model, so, &c->sess)) ||
        ort_check(api, api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &c->mem))) {
        if (so) api->ReleaseSessionOptions(so);
        ft8nn_close(c);
        return NULL;
    }
    api->ReleaseSessionOptions(so);
    c->pool = par_new(n_threads);
    c->ht = ht_new();
    for (int i = 0; i < CTX_HIST; i++) {
        c->hist[i].t = INT64_MIN;
        c->hist[i].d = malloc(sizeof(dec) * DEC_CAP);
    }
    return c;
}

void ft8nn_close(ft8nn_ctx *c) {
    if (!c) return;
    const OrtApi *api = c->api;
    if (c->mem) api->ReleaseMemoryInfo(c->mem);
    if (c->sess) api->ReleaseSession(c->sess);
    par_free(c->pool);
    if (c->env) api->ReleaseEnv(c->env);
    ht_free(c->ht);
    for (int i = 0; i < CTX_HIST; i++) free(c->hist[i].d);
    free(c);
}

void ft8nn_reset_hashes(ft8nn_ctx *c) { ht_clear(c->ht); }

int ft8nn_threads(const ft8nn_ctx *c) { return par_threads(c->pool); }

/* NN LLRs for nh hypotheses -> llr[h*174 ...]; feat holds NN_BATCH feature rows. Safe to call from several threads
 * at once (ONNX Runtime sessions allow concurrent Run). */
static int run_nn(ft8nn_ctx *c, sync_hyp *const *hyps, int nh, double *llr, float *feat) {
    const OrtApi *api = c->api;
    const char *in_names[] = {"features"}, *out_names[] = {"llr"};
    int rc = 0;
    for (int i0 = 0; i0 < nh && !rc; i0 += NN_BATCH) {
        int b = nh - i0 < NN_BATCH ? nh - i0 : NN_BATCH;
        for (int i = 0; i < b; i++) nn_features(hyps[i0 + i]->y, hyps[i0 + i]->start, c->nv_bb, feat + (size_t)i * NN * NFEAT);
        int64_t shape[3] = {b, NN, NFEAT};
        OrtValue *in = NULL, *out = NULL;
        if (ort_check(api, api->CreateTensorWithDataAsOrtValue(c->mem, feat, sizeof(float) * b * NN * NFEAT, shape, 3,
                                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in)) ||
            ort_check(api, api->Run(c->sess, NULL, in_names, (const OrtValue *const *)&in, 1, out_names, 1, &out))) {
            rc = -1;
        } else {
            float *o;
            if (ort_check(api, api->GetTensorMutableData(out, (void **)&o))) rc = -1;
            else
                for (int i = 0; i < b * 174; i++) llr[(size_t)i0 * 174 + i] = o[i];
        }
        if (in) api->ReleaseValue(in);
        if (out) api->ReleaseValue(out);
    }
    return rc;
}

static int same_payload(const uint8_t *a, const uint8_t *b) { return !memcmp(a, b, 77); }

/* d's data tones equal e's tones shifted by k symbols and m tones on >= GHOST_MATCH of the >= GHOST_MIN_SYMS data
 * symbols where e's shifted tone lies in d's 8-tone band: d was read off e's signal at a wrong lag/bin */
static int is_ghost(const dec *d, const dec *e) {
    double df = d->f0 - e->f0;
    if (fabs(df) >= GHOST_MAX_DF) return 0;
    uint8_t cw[174];
    int td[NN], te[NN];
    encode77(d->bits77, cw, td);
    encode77(e->bits77, cw, te);
    double x = df / 6.25, y = (d->dt - e->dt) * 12000.0 / 1920.0;
    for (int m = (int)floor(x); m <= (int)ceil(x); m++)
        for (int k = (int)floor(y) - 1; k <= (int)ceil(y) + 1; k++) {
            int n = 0, hit = 0;
            for (int i = 0; i < ND; i++) {
                int p = i < 29 ? 7 + i : 43 + (i - 29), j = p + k;
                if (j < 0 || j >= NN) continue;
                int t = te[j] - m;
                if (t < 0 || t >= 8) continue;
                n++;
                hit += td[p] == t;
            }
            if (n >= GHOST_MIN_SYMS && hit >= GHOST_MATCH * n) return 1;
        }
    return 0;
}

static double estimate_snr(const sync_hyp *h, const uint8_t *bits77) {
    uint8_t cw[174];
    int tones[NN];
    encode77(bits77, cw, tones);
    cplx cs[NN][8];
    symbol_spectra8(h->y, h->start, cs);
    double ps = 0;
    for (int s = 0; s < NN; s++) {
        double a = cabs(cs[s][tones[s]]);
        ps += a * a;
    }
    ps /= NN;
    double v = ps - NOISE_BIN;
    if (v < 1e-3 * NOISE_BIN) v = 1e-3 * NOISE_BIN;
    return 10 * log10(v / NOISE_BIN) - 26.0 + SNR_OFFSET;
}

/* CQ a-priori: n28 = 2 ('CQ'), ipa = 0, i3 = 1 on bits 0..28 and 74..76 */
static int ap_idx[32];
static uint8_t ap_val[32];
static void ap_init(void) {
    for (int i = 0; i < 29; i++) ap_idx[i] = i;
    for (int i = 0; i < 3; i++) ap_idx[29 + i] = 74 + i;
    for (int i = 0; i < 28; i++) ap_val[i] = (2 >> (27 - i)) & 1;
    ap_val[28] = 0;
    ap_val[29] = 0;
    ap_val[30] = 0;
    ap_val[31] = 1;
}

/* standard message (i3 = 1 or 2): call fields at bits 0..28 and 29..57 */
static int std_msg(const uint8_t *b) { return !b[74] && (b[75] ^ b[76]); }

/* Context a-priori (decoder.context_patterns + _try_ctx): for each context message within CTX_DF / CTX_DDT, pattern
 * pair (bits 0..57 + i3) then sender (29..57 + i3), deduplicated in order; pin, BP, else OSD-2; first CRC pass that
 * keeps the pinned bits wins. Returns FT8NN_BP_CTX, FT8NN_OSD_CTX or -1. */
static int try_ctx(const double *llr, const dec *cx, int ncx, double f0, double dt, uint8_t *a91) {
    enum { MAXP = 64 };
    const uint8_t *pb[MAXP];
    int pk[MAXP], np = 0;
    for (int i = 0; i < ncx && np < MAXP; i++) {
        if (fabs(cx[i].f0 - f0) > CTX_DF || fabs(cx[i].dt - dt) > CTX_DDT || !std_msg(cx[i].bits77)) continue;
        for (int kind = 0; kind < 2 && np < MAXP; kind++) {
            int lo = kind ? 29 : 0, dup = 0;
            for (int q = 0; q < np && !dup; q++)
                dup = pk[q] == kind && !memcmp(pb[q] + lo, cx[i].bits77 + lo, 58 - lo) && !memcmp(pb[q] + 74, cx[i].bits77 + 74, 3);
            if (!dup) {
                pb[np] = cx[i].bits77;
                pk[np++] = kind;
            }
        }
    }
    if (!np) return -1;
    double mag = 0, l2[174], dist;
    for (int i = 0; i < 174; i++) mag = fmax(mag, fabs(llr[i]));
    mag *= 1.01;
    uint8_t bits[174];
    int nh;
    for (int q = 0; q < np; q++) {
        int lo = pk[q] ? 29 : 0, method;
        memcpy(l2, llr, sizeof l2);
        for (int i = lo; i < 58; i++) l2[i] = pb[q][i] ? mag : -mag;
        for (int i = 74; i < 77; i++) l2[i] = pb[q][i] ? mag : -mag;
        if (bp_decode(l2, 30, bits, &nh) && crc_ok(bits)) method = FT8NN_BP_CTX;
        else if (osd_decode(l2, OSD_ORDER, bits, &dist, &nh)) method = FT8NN_OSD_CTX;
        else continue;
        if (memcmp(bits + lo, pb[q] + lo, 58 - lo) || memcmp(bits + 74, pb[q] + 74, 3)) continue;
        memcpy(a91, bits, 91);
        return method;
    }
    return -1;
}

static int is_ctx(int m) { return m == FT8NN_BP_CTX || m == FT8NN_OSD_CTX; }
static int same_sender(const dec *a, const dec *b) {
    return std_msg(a->bits77) && std_msg(b->bits77) && !memcmp(a->bits77 + 29, b->bits77 + 29, 29);
}

/* A station sends one message per slot (decoder._drop_ctx_dupes): drop a context decode in all[from..n) whose sender
 * a non-context decode in all[0..n) or an earlier kept context decode in all[from..) has. Returns the new n. */
static int drop_ctx_dupes(dec *all, int from, int n) {
    uint8_t drop[DEC_CAP] = {0}; /* n <= DEC_CAP (fixed size: not every compiler has variable-length arrays) */
    for (int i = from; i < n; i++) {
        if (!is_ctx(all[i].method)) continue;
        for (int j = 0; j < n && !drop[i]; j++)
            if (j != i && (!is_ctx(all[j].method) || (j >= from && j < i && !drop[j])) && same_sender(&all[i], &all[j])) drop[i] = 1;
    }
    int w = from;
    for (int i = from; i < n; i++)
        if (!drop[i]) all[w++] = all[i];
    return w;
}

static int ghost_prone(int m) { return m == FT8NN_OSD_ERASE || m == FT8NN_OSD_BP; }

/* OSD on the BP posterior sums (decoder._try_osd_bp); hard errors counted against the channel LLRs.
 * returns FT8NN_OSD_BP or -1 */
static int try_osd_bp(const double *llr, uint8_t *a91) {
    double zs[OSD_SNAPS * 174], dist;
    uint8_t cw[174];
    int nh;
    bp_posterior_sums(llr, OSD_SNAPS, zs);
    for (int s = 0; s < OSD_SNAPS; s++) {
        int crc = osd_decode(zs + s * 174, OSD_ORDER, cw, &dist, &nh);
        nh = 0;
        for (int i = 0; i < 174; i++) nh += cw[i] != (llr[i] > 0);
        if (crc && nh <= OSD_MAX_HARD) {
            memcpy(a91, cw, 91);
            return FT8NN_OSD_BP;
        }
    }
    return -1;
}

/* returns method (FT8NN_BP_AP / FT8NN_OSD_AP) or -1; osd_order < 0 = BP only */
static int try_ap(const double *llr, uint8_t *a91, int osd_order) {
    double mag = 0, l2[174];
    for (int i = 0; i < 174; i++) mag = fmax(mag, fabs(llr[i]));
    mag *= 1.01;
    memcpy(l2, llr, sizeof l2);
    for (int q = 0; q < 32; q++) l2[ap_idx[q]] = ap_val[q] ? mag : -mag;
    uint8_t bits[174];
    int nh, method = -1;
    double dist;
    if (bp_decode(l2, 30, bits, &nh) && crc_ok(bits)) method = FT8NN_BP_AP;
    else if (osd_order >= 0 && osd_decode(l2, osd_order, bits, &dist, &nh) && nh <= OSD_MAX_HARD) method = FT8NN_OSD_AP;
    if (method < 0) return -1;
    for (int q = 0; q < 32; q++)
        if (bits[ap_idx[q]] != ap_val[q]) return -1;
    memcpy(a91, bits, 91);
    return method;
}

/* Burst-erasure windows (first, end) over the 58 data symbols, sorted and deduplicated, as
 * decoder.ERASE_WINDOWS (BP stage) and decoder.ERASE_OSD_WINDOWS (OSD-0 stage) */
static int erase_win[128][2], n_erase, erase_osd_win[128][2], n_erase_osd;
static int build_windows(const int *widths, int nw, int win[][2]) {
    int n = 0;
    for (int wi = 0; wi < nw; wi++)
        for (int a = 0; a < ND - widths[wi] + 2; a += 2) {
            int b = a + widths[wi] < ND ? a + widths[wi] : ND, q = 0, dup = 0;
            for (int j = 0; j < n; j++) dup |= win[j][0] == a && win[j][1] == b;
            if (dup) continue;
            while (q < n && (win[q][0] < a || (win[q][0] == a && win[q][1] < b))) q++;
            memmove(win[q + 1], win[q], sizeof(win[0]) * (n - q));
            win[q][0] = a;
            win[q][1] = b;
            n++;
        }
    return n;
}
static void erase_init(void) {
    static const int w_bp[4] = {6, 10, 14, 18}, w_osd[3] = {20, 22, 24};
    n_erase = build_windows(w_bp, 4, erase_win);
    n_erase_osd = build_windows(w_osd, 3, erase_osd_win);
}

/* Sender changed the message mid-transmission: a burst of confidently wrong symbols. Zero each
 * window of data-symbol LLRs in turn, BP; first CRC pass wins. Then OSD-0 on each wide window
 * (too many erasures for BP); the smallest-distance CRC pass wins if below ERASE_OSD_DMAX.
 * Returns FT8NN_BP_ERASE, FT8NN_OSD_ERASE or -1. */
static int try_erase(const double *llr, uint8_t *a91) {
    double l2[174];
    uint8_t bits[174];
    int nh;
    for (int w = 0; w < n_erase; w++) {
        memcpy(l2, llr, sizeof l2);
        for (int i = 3 * erase_win[w][0]; i < 3 * erase_win[w][1]; i++) l2[i] = 0.0;
        if (bp_decode(l2, 30, bits, &nh) && crc_ok(bits)) {
            memcpy(a91, bits, 91);
            return FT8NN_BP_ERASE;
        }
    }
    double best = -1.0, dist;
    for (int w = 0; w < n_erase_osd; w++) {
        memcpy(l2, llr, sizeof l2);
        for (int i = 3 * erase_osd_win[w][0]; i < 3 * erase_osd_win[w][1]; i++) l2[i] = 0.0;
        if (osd_decode(l2, 0, bits, &dist, &nh) && dist < ERASE_OSD_DMAX && (best < 0 || dist < best)) {
            best = dist;
            memcpy(a91, bits, 91);
        }
    }
    return best >= 0 ? FT8NN_OSD_ERASE : -1;
}

static int any_bits(const uint8_t *b, int n) {
    for (int i = 0; i < n; i++)
        if (b[i]) return 1;
    return 0;
}

/* decoder._plausible: method if the payload could be a real transmission (callsign-block test only for OSD-family
 * methods), else -1 */
static int plausible(int method, const uint8_t *a91) {
    if (method < 0) return method;
    int deep = method == FT8NN_OSD || method == FT8NN_OSD_AP || method == FT8NN_OSD_ERASE || method == FT8NN_OSD_BP ||
               method == FT8NN_OSD_CTX || method == FT8NN_OSD_CC || method == FT8NN_OSD_BP_CC;
    return plausible77(a91, deep) ? method : -1;
}

/* tried-table slot of a candidate, or -1 if outside the table */
static int tried_idx(const candidate *cd) {
    long b = lround(cd->f0 / 3.125), l = lround(cd->dt * 25.0) + TRIED_LAG0;
    if (b < 0 || b >= TRIED_NB || l < 0 || l >= TRIED_NL) return -1;
    return (int)((b * TRIED_NL + l) * 2 + (cd->early != 0));
}

/* decoder.decode_slot reuse_df: tried before (pass q) and nothing subtracted since within REUSE_DF Hz */
static int reuse_skip(const candidate *cd, const signed char *tried, const dec *all, int nall) {
    int i = tried_idx(cd);
    if (i < 0 || tried[i] < 0) return 0;
    for (int q = 0; q < nall; q++)
        if (all[q].pass >= tried[i] && fabs(all[q].f0 - cd->f0) < REUSE_DF) return 0;
    return 1;
}

/* What the decoder stages make of one hypothesis on their own: the soft metrics in turn (NN, then classical nsym
 * 1..3), each through BP, OSD, OSD on BP sums, context and CQ a-priori, erasures, until one gives a plausible message.
 * It depends on nothing else in the pass, so the hypotheses of a pass are worked out in parallel and merged in order
 * afterwards (decode_pass). */
typedef struct {
    int8_t status; /* 0: no decode; 1: decoded (k, method, a91, snr); 2: all-zero payload (classic false decode) */
    int8_t k, method;
    uint8_t a91[91];
    double snr;
    int n_bp, n_osd;
} hyp_res;

static void decode_hyp(const sync_hyp *hy, const double *L, int early, const dec *cx, int ncx, hyp_res *r) {
    r->status = 0;
    r->n_bp = r->n_osd = 0;
    for (int k = 0; k < NMET; k++) {
        const double *l = L + (size_t)k * 174;
        double exp_err = 0;
        for (int i = 0; i < 174; i++) exp_err += 1.0 / (1.0 + exp(fabs(l[i])));
        uint8_t bits[174], a91[91];
        int nhard, method = -1;
        double dist;
        r->n_bp++;
        /* early-DT candidates have weaker sync evidence: no OSD on BP sums or under a-priori bits, no erasure stages */
        /* every stage's result must be a plausible message (decoder._plausible); if not, the next stage tries */
        if (bp_decode(l, 30, bits, &nhard) && crc_ok(bits)) {
            memcpy(a91, bits, 91);
            method = plausible(FT8NN_BP, a91);
        }
        if (method < 0 && k < 1 && exp_err < OSD_GATE && ++r->n_osd) {
            if (osd_decode(l, OSD_ORDER, bits, &dist, &nhard) && nhard <= OSD_MAX_HARD) {
                memcpy(a91, bits, 91);
                method = plausible(FT8NN_OSD, a91);
            }
            if (method < 0 && OSD_SNAPS > 0 && !early) method = plausible(try_osd_bp(l, a91), a91);
        }
        if (method < 0 && k == 0 && ncx > 0 && !early && exp_err < CTX_GATE)
            method = plausible(try_ctx(l, cx, ncx, hy->f0, hy->dt, a91), a91);
        if (method < 0 && k == 0 && exp_err < AP_GATE) method = plausible(try_ap(l, a91, early ? -1 : OSD_ORDER), a91);
        if (method < 0 && k == 0 && exp_err < ERASE_GATE && !early) method = plausible(try_erase(l, a91), a91);
        if (method < 0) continue;
        if (!any_bits(a91, 77)) {
            r->status = 2;
            return;
        }
        r->status = 1;
        r->k = (int8_t)k;
        r->method = (int8_t)method;
        memcpy(r->a91, a91, 91);
        r->snr = estimate_snr(hy, a91);
        return;
    }
}

/* per-thread CPU seconds of the stages, summed over threads for FT8NN_PROFILE */
enum { ST_SYNC, ST_NN, ST_CL, ST_DEC, N_ST };

typedef struct {
    const baseband *bb;
    const candidate *cands;
    const int *sel; /* candidates to sync, indices into cands */
    int ns;
    sync_hyp *hyps; /* N_HYP slots per selected candidate */
    int *nk;        /* hypotheses written per selected candidate */
    atomic_int next;
    double st[PAR_MAX_THREADS][N_ST];
} sync_job;

static void sync_worker(void *arg, int w) {
    sync_job *j = arg;
    double t0 = tthr();
    for (;;) {
        int i = atomic_fetch_add(&j->next, 1);
        if (i >= j->ns) break;
        j->nk[i] = fine_sync2(j->bb, &j->cands[j->sel[i]], N_HYP, j->hyps + (size_t)i * N_HYP);
    }
    j->st[w][ST_SYNC] += tthr() - t0;
}

/* hypotheses per work item (NN batch, classical metrics, decoding): small enough to keep all threads busy to the end
 * of a pass; ONNX Runtime's CPU time per hypothesis is the same for batches of 16, 32 and 128 */
#define NN_CHUNK 32

typedef struct {
    ft8nn_ctx *c;
    sync_hyp **hp;
    const uint8_t *early;
    int nh;
    const dec *cx;
    int ncx;
    hyp_res *res;
    atomic_int next, err;
    atomic_uchar *cdone;           /* per chunk: results written */
    void (*between)(void *, int);  /* run by the calling thread after each of its chunks, with the next chunk index */
    void *between_arg;
    double st[PAR_MAX_THREADS][N_ST];
} hyp_job;

static void hyp_worker(void *arg, int w) {
    hyp_job *j = arg;
    float *feat = malloc(sizeof(float) * NN_BATCH * NN * NFEAT);
    double *nnl = malloc(sizeof(double) * NN_CHUNK * 174), L[NMET * 174];
    for (;;) {
        int i0 = atomic_fetch_add(&j->next, NN_CHUNK);
        if (i0 >= j->nh || atomic_load(&j->err)) break;
        int b = j->nh - i0 < NN_CHUNK ? j->nh - i0 : NN_CHUNK;
        double t0 = tthr();
        if (run_nn(j->c, j->hp + i0, b, nnl, feat)) {
            atomic_store(&j->err, 1);
            break;
        }
        double t1 = tthr();
        j->st[w][ST_NN] += t1 - t0;
        for (int i = 0; i < b; i++) {
            const sync_hyp *hy = j->hp[i0 + i];
            t0 = tthr();
            memcpy(L, nnl + (size_t)i * 174, sizeof(double) * 174);
            cplx cs[NN][8];
            symbol_spectra8(hy->y, hy->start, cs);
            for (int k = 1; k < NMET; k++) coherent_llr(cs, k, L + k * 174);
            t1 = tthr();
            j->st[w][ST_CL] += t1 - t0;
            decode_hyp(hy, L, j->early[i0 + i], j->cx, j->ncx, &j->res[i0 + i]);
            j->st[w][ST_DEC] += tthr() - t1;
        }
        atomic_store_explicit(&j->cdone[i0 / NN_CHUNK], 1, memory_order_release);
        if (w == 0 && j->between) j->between(j->between_arg, (int)((j->nh + NN_CHUNK - 1) / NN_CHUNK));
    }
    free(feat);
    free(nnl);
}

static void report(ft8nn_ctx *c, dec *all, int from, int n, int ctx_too, int pass, ft8nn_pass_cb cb, void *user,
                   ft8nn_decode_t *batch);

/* The in-order merge of a pass's hypothesis results, run by the calling thread as chunks complete. */
typedef struct {
    ft8nn_ctx *c;
    hyp_job *j;
    sync_hyp **hp;
    const int *owner;
    uint8_t *done_cand;
    dec *all;
    int nall, cap, pass, added, h; /* next hypothesis to merge */
    int sent;                      /* all[nall .. nall + sent) handed to cb */
    long n_bp, n_osd, s_bp, s_osd;
    ft8nn_pass_cb cb;
    void *user;
    ft8nn_decode_t *batch;
} merger;

/* merge hypotheses h .. upto: skip a candidate once decoded and a hypothesis next to a decode of this pass, drop
 * payloads already known - exactly the serial decoder's choices */
static void merge_to(merger *m, int upto) {
    dec *all = m->all;
    int nall = m->nall;
    for (; m->h < upto; m->h++) {
        int h = m->h, ci = m->owner[h];
        const hyp_res *r = &m->j->res[h];
        m->s_bp += r->n_bp;
        m->s_osd += r->n_osd;
        if (m->done_cand[ci]) continue;
        int near = 0;
        for (int q = nall; q < nall + m->added; q++)
            if (fabs(m->hp[h]->f0 - all[q].f0) < 3 && fabs(m->hp[h]->dt - all[q].dt) < 0.05) near = 1;
        if (near) {
            m->done_cand[ci] = 1;
            continue;
        }
        m->n_bp += r->n_bp;
        m->n_osd += r->n_osd;
        if (r->status != 1) continue;
        int known = 0;
        for (int q = 0; q < nall + m->added; q++)
            if (same_payload(all[q].bits77, r->a91)) known = 1;
        if (!known && nall + m->added < m->cap) {
            dec *d = &all[nall + m->added++];
            d->final = 0;
            memcpy(d->bits77, r->a91, 77);
            d->f0 = m->hp[h]->f0;
            d->dt = m->hp[h]->dt;
            d->method = r->method;
            d->metric = r->k;
            d->pass = m->pass;
            d->snr = r->snr;
        }
        m->done_cand[ci] = 1;
    }
}

/* Between chunks: merge what has completed in order, and hand the merged decodes to cb up to the first that the end
 * of the pass may still drop (an osd_er / osd_bp ghost). That keeps texts unpacked in the serial order, so the hash
 * table learns callsigns in the same order. Context decodes are passed over, as report() does. */
static void merge_between(void *arg, int nchunks) {
    merger *m = arg;
    int ch = m->h / NN_CHUNK;
    while (ch < nchunks && atomic_load_explicit(&m->j->cdone[ch], memory_order_acquire)) ch++;
    int upto = ch * NN_CHUNK < m->j->nh ? ch * NN_CHUNK : m->j->nh;
    if (upto <= m->h) return;
    merge_to(m, upto);
    if (!m->cb) return;
    int q = m->sent;
    while (q < m->added && !ghost_prone(m->all[m->nall + q].method)) q++;
    if (q == m->sent) return;
    int any = 0;
    for (int i = m->sent; i < q; i++) any |= !is_ctx(m->all[m->nall + i].method);
    if (any) report(m->c, m->all, m->nall + m->sent, m->nall + q, 0, m->pass, m->cb, m->user, m->batch);
    m->sent = q;
}

/* One pass. Appends new decodes (payload not in all[0..nall)) to all; returns count added, -1 on error. Decodes that
 * are final before the pass ends go to cb already (report: pass, then marked final). */
static int decode_pass(ft8nn_ctx *c, const double *x, int n, dec *all, int nall, int cap, int pass, const dec *cx, int ncx,
                       signed char *tried, double dt_shift, ft8nn_pass_cb cb, void *user, ft8nn_decode_t *batch) {
    if (prof < 0) prof = getenv("FT8NN_PROFILE") != NULL;
    double t0 = tnow(), c0 = tcpu(), t1, t2, t3;
    long n_bp = 0, n_osd = 0, s_bp = 0, s_osd = 0;
    double st[N_ST] = {0};
    candidate *cands = malloc(sizeof(candidate) * (MAX_CAND + (int)(FMAX / 3.125) + 1));
    int nc = find_candidates(x, n, 150.0, FMAX, MAX_CAND, 1.3, BASE_MEDIAN, EARLY_DT, EARLY_MIN, dt_shift, cands, c->pool);
    t1 = tnow();
    /* candidates to try, in order; the tried table is updated as the serial loop always did */
    int *sel = malloc(sizeof(int) * (size_t)(nc + 1)), ns = 0, nskip = 0;
    for (int ci = 0; ci < nc; ci++) {
        if (REUSE_DF > 0 && pass > 0 && reuse_skip(&cands[ci], tried, all, nall)) {
            nskip++;
            continue;
        }
        int ti = tried_idx(&cands[ci]);
        if (ti >= 0) tried[ti] = (signed char)pass;
        sel[ns++] = ci;
    }
    baseband bb;
    baseband_init(&bb, x, n, c->pool);
    sync_hyp *hyps = malloc(sizeof(sync_hyp) * (size_t)(ns * N_HYP + 1));
    int *nk = malloc(sizeof(int) * (size_t)(ns + 1));
    {
        sync_job *sj = calloc(1, sizeof *sj);
        *sj = (sync_job){.bb = &bb, .cands = cands, .sel = sel, .ns = ns, .hyps = hyps, .nk = nk};
        atomic_init(&sj->next, 0);
        par_run(c->pool, sync_worker, sj);
        for (int w = 0; w < PAR_MAX_THREADS; w++) st[ST_SYNC] += sj->st[w][ST_SYNC];
        free(sj);
    }
    baseband_free(&bb);
    /* hypotheses in candidate order, as the serial loop produced them */
    sync_hyp **hp = malloc(sizeof(sync_hyp *) * (size_t)(ns * N_HYP + 1));
    int *owner = malloc(sizeof(int) * (size_t)(ns * N_HYP + 1)), nh = 0;
    uint8_t *early = malloc((size_t)(ns * N_HYP + 1));
    for (int i = 0; i < ns; i++)
        for (int q = 0; q < nk[i]; q++) {
            hp[nh] = hyps + (size_t)i * N_HYP + q;
            owner[nh] = i;
            early[nh++] = (uint8_t)(cands[sel[i]].early != 0);
        }
    t2 = tnow();
    t3 = t2;
    int added = 0;
    if (nh == 0) goto done;
    hyp_res *res = malloc(sizeof(hyp_res) * (size_t)nh);
    int nchunk = (nh + NN_CHUNK - 1) / NN_CHUNK;
    atomic_uchar *cdone = malloc(sizeof(atomic_uchar) * (size_t)nchunk);
    for (int i = 0; i < nchunk; i++) atomic_init(&cdone[i], 0);
    uint8_t *done_cand = calloc((size_t)ns + 1, 1);
    hyp_job *hj = calloc(1, sizeof *hj);
    merger m = {.c = c, .j = hj, .hp = hp, .owner = owner, .done_cand = done_cand, .all = all, .nall = nall, .cap = cap,
                .pass = pass, .cb = cb, .user = user, .batch = batch};
    *hj = (hyp_job){.c = c, .hp = hp, .early = early, .nh = nh, .cx = cx, .ncx = ncx, .res = res, .cdone = cdone,
                    .between = merge_between, .between_arg = &m};
    atomic_init(&hj->next, 0);
    atomic_init(&hj->err, 0);
    par_run(c->pool, hyp_worker, hj);
    int err = atomic_load(&hj->err);
    for (int w = 0; w < PAR_MAX_THREADS; w++)
        for (int s = ST_NN; s < N_ST; s++) st[s] += hj->st[w][s];
    t3 = tnow();
    if (!err) merge_to(&m, nh);
    added = m.added;
    n_bp = m.n_bp;
    n_osd = m.n_osd;
    s_bp = m.s_bp;
    s_osd = m.s_osd;
    free(hj);
    free(cdone);
    free(done_cand);
    free(res);
    if (err) {
        added = -1;
        goto done;
    }
    /* drop osd_er / osd_bp ghosts of another decode (earlier pass, or this pass and not ghost-prone), as
     * decoder._decode_pass (GHOST_PRONE) */
    {
        uint8_t drop[DEC_CAP] = {0}; /* added <= DEC_CAP */
        for (int i = 0; i < added; i++) {
            const dec *d = &all[nall + i];
            if (!ghost_prone(d->method)) continue;
            for (int q = 0; q < nall + added && !drop[i]; q++) {
                if (q >= nall && ghost_prone(all[q].method)) continue;
                if (is_ghost(d, &all[q])) drop[i] = 1;
            }
        }
        int w = 0;
        for (int i = 0; i < added; i++)
            if (!drop[i]) all[nall + w++] = all[nall + i];
        added = w;
    }
    added = drop_ctx_dupes(all, nall, nall + added) - nall;
done:
    if (prof)
        fprintf(stderr,
                "pass %d: cand %d skip %d hyp %d | wall cand %.2fs sync %.2fs decode %.2fs merge %.2fs = %.2fs, cpu %.2fs"
                " (sync %.2f nn %.2f classical %.2f bp/osd %.2f) | bp %ld osd %ld (+%ld/%ld speculative) -> %d\n",
                pass, nc, nskip, nh, t1 - t0, t2 - t1, t3 - t2, tnow() - t3, tnow() - t0, tcpu() - c0, st[ST_SYNC],
                st[ST_NN], st[ST_CL], st[ST_DEC], n_bp, n_osd, s_bp - n_bp, s_osd - n_osd, added);
    free(cands); /* candidate flags (early) are read in the decode loop */
    free(sel);
    free(hyps);
    free(nk);
    free(hp);
    free(owner);
    free(early);
    return added;
}

/* Co-channel retry (decoder._cochannel_stage). Two stations within a few Hz and tens of ms of each other share one
 * candidate cell (every FT8 signal carries the same Costas arrays); subtracting the stronger takes most of the weaker
 * one's sync with it, while its data symbols survive. So after the last pass, at every decode's own cell, the residual
 * is searched with the data-symbol sync metric and the best peaks go to the NN as 3 x 3 hypotheses: NN metric -> BP ->
 * OSD-2 (gated) -> OSD on BP sums, plausibility rules. In order: one decode per peak; skip a payload already decoded and
 * a ghost of any decode in the slot. Three steps as decode_pass: the grid search per cell and the per-hypothesis
 * decoding in parallel, then a serial merge in the order the Python loop takes. */
typedef struct {
    double df; /* Hz from the cell's f0 */
    int start, cell, group;
} cc_spec;

static void cc_decode_hyp(const sync_hyp *hy, const double *l, hyp_res *r) {
    r->status = 0;
    double exp_err = 0, dist;
    for (int i = 0; i < 174; i++) exp_err += 1.0 / (1.0 + exp(fabs(l[i])));
    uint8_t bits[174], a91[91];
    int nhard, method = -1;
    if (bp_decode(l, 30, bits, &nhard) && crc_ok(bits)) {
        memcpy(a91, bits, 91);
        method = plausible(FT8NN_BP_CC, a91);
    }
    if (method < 0 && exp_err < OSD_GATE) {
        if (osd_decode(l, OSD_ORDER, bits, &dist, &nhard) && nhard <= OSD_MAX_HARD) {
            memcpy(a91, bits, 91);
            method = plausible(FT8NN_OSD_CC, a91);
        }
        if (method < 0 && OSD_SNAPS > 0 && try_osd_bp(l, a91) >= 0) method = plausible(FT8NN_OSD_BP_CC, a91);
    }
    if (method < 0) return;
    if (!any_bits(a91, 77)) {
        r->status = 2;
        return;
    }
    r->status = 1;
    r->k = 0;
    r->method = (int8_t)method;
    memcpy(r->a91, a91, 91);
    r->snr = estimate_snr(hy, a91);
}

typedef struct {
    ft8nn_ctx *c;
    const dec *cells;
    const baseband *bb;
    cplx *y0;      /* NBB per cell */
    cc_spec *spec; /* CC_MAXH per cell */
    int *nspec, *npeak;
    const cc_spec **flat;
    int nflat;
    hyp_res *res;
    atomic_int next, err;
    double dfs[CC_NF];
    int dts[CC_NT];
} cc_job;

static void cc_search(void *arg, int ci) {
    cc_job *j = arg;
    const dec *d = &j->cells[ci];
    cplx *y0 = j->y0 + (size_t)ci * NBB;
    baseband_downsample(j->bb, d->f0, y0);
    int start0 = (int)nearbyint((NOMINAL_START + d->dt) * FS_BB);
    const double *cc_dfs = j->dfs;
    const int *cc_dts = j->dts;
    double M[CC_NF * CC_NT];
    data_sync(y0, start0, cc_dfs, CC_NF, cc_dts, CC_NT, M);
    /* local maxima over a 3 x 3 window (scipy maximum_filter, reflect mode == clipped window), in row-major order */
    int pk[CC_NF * CC_NT], np_ = 0;
    for (int i = 0; i < CC_NF; i++)
        for (int t = 0; t < CC_NT; t++) {
            double v = M[i * CC_NT + t];
            if (v < CC_MIN || fabs(cc_dfs[i]) > CC_SPAN + 1e-9 || abs(cc_dts[t]) > CC_TFAR) continue;
            int ok = 1;
            for (int di = -1; di <= 1 && ok; di++)
                for (int dj = -1; dj <= 1; dj++) {
                    int ii = i + di, jj = t + dj;
                    if (ii >= 0 && ii < CC_NF && jj >= 0 && jj < CC_NT && M[ii * CC_NT + jj] > v) {
                        ok = 0;
                        break;
                    }
                }
            if (ok) pk[np_++] = i * CC_NT + t;
        }
    int ns = 0, ng = 0;
    for (int z = 0; z < 2; z++) { /* near zone, then the far band */
        uint8_t used[CC_NF * CC_NT] = {0};
        for (int r = 0; r < (z ? CC_FAR : CC_NEAR); r++) {
            int bi = -1; /* best remaining peak of the zone; ties: earlier in row-major order */
            for (int q = 0; q < np_; q++) {
                int a = abs(cc_dts[pk[q] % CC_NT]);
                if (used[q] || (z ? a <= CC_TNEAR : a > CC_TNEAR)) continue;
                if (bi < 0 || M[pk[q]] > M[pk[bi]]) bi = q;
            }
            if (bi < 0) break;
            used[bi] = 1;
            static const double oa[9] = {0, -CC_RDF, -CC_RDF, -CC_RDF, 0, 0, CC_RDF, CC_RDF, CC_RDF};
            static const int ob[9] = {0, -CC_RDT, 0, CC_RDT, -CC_RDT, CC_RDT, -CC_RDT, 0, CC_RDT};
            for (int o = 0; o < 9; o++) {
                cc_spec *s = &j->spec[(size_t)ci * CC_MAXH + ns++];
                s->df = cc_dfs[pk[bi] / CC_NT] + oa[o];
                s->start = start0 + cc_dts[pk[bi] % CC_NT] + ob[o];
                s->cell = ci;
                s->group = ng;
            }
            ng++;
        }
    }
    j->nspec[ci] = ns;
    j->npeak[ci] = ng;
}

static void cc_worker(void *arg, int w) {
    (void)w;
    cc_job *j = arg;
    float *feat = malloc(sizeof(float) * NN_BATCH * NN * NFEAT);
    double *llr = malloc(sizeof(double) * NN_CHUNK * 174);
    sync_hyp *hy = malloc(sizeof(sync_hyp) * NN_CHUNK), *hp[NN_CHUNK];
    for (;;) {
        int i0 = atomic_fetch_add(&j->next, NN_CHUNK);
        if (i0 >= j->nflat || atomic_load(&j->err)) break;
        int b = j->nflat - i0 < NN_CHUNK ? j->nflat - i0 : NN_CHUNK;
        for (int i = 0; i < b; i++) {
            const cc_spec *s = j->flat[i0 + i];
            memcpy(hy[i].y, j->y0 + (size_t)s->cell * NBB, sizeof hy[i].y);
            bb_rotate(hy[i].y, s->df);
            hy[i].start = s->start;
            hy[i].f0 = j->cells[s->cell].f0 + s->df;
            hy[i].dt = s->start / FS_BB - NOMINAL_START;
            hy[i].metric = 0;
            hp[i] = &hy[i];
        }
        if (run_nn(j->c, hp, b, llr, feat)) {
            atomic_store(&j->err, 1);
            break;
        }
        for (int i = 0; i < b; i++) cc_decode_hyp(&hy[i], llr + (size_t)i * 174, &j->res[i0 + i]);
    }
    free(feat);
    free(llr);
    free(hy);
}

/* Appends the stage's decodes to all[nall..); returns the new count, or -1 on an ONNX Runtime error. */
static int cochannel_stage(ft8nn_ctx *c, const double *x, int n, dec *all, int nall) {
    double t0 = tnow();
    baseband bb;
    baseband_init(&bb, x, n, c->pool);
    cc_job *j = calloc(1, sizeof *j);
    for (int i = 0; i < CC_NF; i++) j->dfs[i] = -(CC_SPAN + 2 * CC_GF) + i * CC_GF; /* numpy arange: exact here */
    for (int t = 0; t < CC_NT; t++) j->dts[t] = -(CC_TFAR + CC_GT) + t * CC_GT;
    j->c = c;
    j->cells = all;
    j->bb = &bb;
    j->y0 = malloc(sizeof(cplx) * NBB * (size_t)nall);
    j->spec = malloc(sizeof(cc_spec) * CC_MAXH * (size_t)nall);
    j->nspec = calloc((size_t)nall, sizeof(int));
    j->npeak = calloc((size_t)nall, sizeof(int));
    par_for(c->pool, nall, cc_search, j);
    int ngroup = 0, *goff = malloc(sizeof(int) * (size_t)nall);
    for (int ci = 0; ci < nall; ci++) {
        goff[ci] = ngroup;
        ngroup += j->npeak[ci];
        j->nflat += j->nspec[ci];
    }
    j->flat = malloc(sizeof(cc_spec *) * (size_t)(j->nflat + 1));
    for (int ci = 0, q = 0; ci < nall; ci++)
        for (int s = 0; s < j->nspec[ci]; s++) j->flat[q++] = &j->spec[(size_t)ci * CC_MAXH + s];
    j->res = malloc(sizeof(hyp_res) * (size_t)(j->nflat + 1));
    atomic_init(&j->next, 0);
    atomic_init(&j->err, 0);
    if (j->nflat) par_run(c->pool, cc_worker, j);
    int rc = atomic_load(&j->err) ? -1 : nall;
    uint8_t *done = calloc((size_t)ngroup + 1, 1);
    for (int i = 0; rc >= 0 && i < j->nflat; i++) {
        const cc_spec *s = j->flat[i];
        const hyp_res *r = &j->res[i];
        int g = goff[s->cell] + s->group;
        if (done[g] || r->status != 1) continue;
        int skip = 0;
        for (int q = 0; q < rc && !skip; q++) skip = same_payload(all[q].bits77, r->a91);
        if (skip) continue;
        dec d = {.f0 = all[s->cell].f0 + s->df, .dt = s->start / FS_BB - NOMINAL_START, .snr = r->snr,
                 .method = r->method, .metric = 0, .pass = PASSES, .final = 0};
        memcpy(d.bits77, r->a91, 77);
        for (int q = 0; q < rc && !skip; q++) skip = is_ghost(&d, &all[q]);
        if (skip || rc >= DEC_CAP) continue;
        all[rc++] = d;
        done[g] = 1;
    }
    if (prof) fprintf(stderr, "cochannel: cells %d hyp %d -> %d, %.2fs\n", nall, j->nflat, rc - nall, tnow() - t0);
    free(done);
    free(goff);
    free(j->y0);
    free(j->spec);
    free(j->nspec);
    free(j->npeak);
    free(j->flat);
    free(j->res);
    free(j);
    baseband_free(&bb);
    return rc;
}

/* Repeat stage (decoder._repeat_stage): for each sender of a standard context message with no decode in this slot,
 * test its earlier messages as whole known codewords on the residual x at their (f0, dt): rho = sum(l*s)/sum(|l|) on
 * NN LLRs clipped to +-REPEAT_CLIP, s = +-1 per codeword bit, against competitors (the sender's CQ with any grid it
 * sent, then each earlier message with every R flag x grid/RRR/RR73/73/report -30..+30 that a station could send:
 * sendable()). Accept the best if it is an earlier message, rho >= REPEAT_RHO and it beats the runner-up of the same
 * hypothesis by REPEAT_MARGIN; with no competitor left the runner-up is REPEAT_SOLO. Appends to
 * all; returns the new count, or -1 on an ONNX Runtime error. */
static void set_bits(uint8_t *b, int v, int n) {
    for (int i = 0; i < n; i++) b[i] = (v >> (n - 1 - i)) & 1;
}
static int get_bits(const uint8_t *b, int n) {
    int v = 0;
    for (int i = 0; i < n; i++) v = v << 1 | b[i];
    return v;
}
/* decoder._sendable: no competitor a station never transmits - what plausible77 rejects (a CQ with the R flag), and a
 * CQ carrying RRR/RR73/73 or a report in place of its locator */
static int sendable(const uint8_t *b) {
    if (!plausible77(b, 0)) return 0;
    uint32_t n28a = (uint32_t)get_bits(b, 28);
    return !(n28a >= 2 && n28a < NTOKENS_CQ && get_bits(b + 59, 15) > MAXGRID4 + 1);
}
static int add_alt(uint8_t (*alt)[77], int na, int cap, const uint8_t *b) {
    for (int i = 0; i < na; i++)
        if (same_payload(alt[i], b)) return na;
    if (na < cap) memcpy(alt[na++], b, 77);
    return na;
}

static int repeat_stage(ft8nn_ctx *c, const double *x, int n, dec *all, int nall, const dec *cx, int ncx) {
    enum { MAXPRI = 16, MAXALT = 2048 };
    uint8_t *done = calloc((size_t)ncx, 1); /* context entry already in an earlier sender group */
    uint8_t(*alt)[77] = malloc(sizeof *alt * MAXALT);
    double *S = malloc(sizeof(double) * MAXALT * 174), *rho = malloc(sizeof(double) * MAXALT);
    sync_hyp *hyps = malloc(sizeof(sync_hyp) * N_HYP), *hp[N_HYP];
    float *feat = malloc(sizeof(float) * NN_BATCH * NN * NFEAT);
    double llr[N_HYP * 174];
    baseband bb;
    int bb_ready = 0, rc = nall;
    for (int g = 0; g < ncx; g++) {
        const uint8_t *b0 = cx[g].bits77;
        if (done[g] || !std_msg(b0)) continue;
        int skip = 0;
        for (int q = 0; q < nall && !skip; q++) skip = same_payload(all[q].bits77, b0);
        if (skip) continue;
        for (int q = 0; q < nall && !skip; q++) skip = std_msg(all[q].bits77) && !memcmp(all[q].bits77 + 29, b0 + 29, 29);
        if (skip) continue;
        /* the sender group: g and every later eligible context entry with the same sender, in context order */
        int mem[2 * DEC_CAP], nm = 0;
        for (int i = g; i < ncx; i++) {
            if (done[i] || !std_msg(cx[i].bits77) || memcmp(cx[i].bits77 + 29, b0 + 29, 29)) continue;
            int known = 0;
            for (int q = 0; q < nall && !known; q++) known = same_payload(all[q].bits77, cx[i].bits77);
            done[i] = 1;
            if (!known) mem[nm++] = i;
        }
        /* alternatives: earlier messages first (deduplicated), then the competitors */
        int na = 0;
        for (int i = 0; i < nm; i++) na = add_alt(alt, na, MAXPRI, cx[mem[i]].bits77);
        int npri = na, grids[MAXPRI], ng = 0;
        for (int i = 0; i < npri; i++) {
            int gv = get_bits(alt[i] + 59, 15), dup = 0;
            for (int j = 0; j < ng; j++) dup |= grids[j] == gv;
            if (gv < MAXGRID4 && !dup) grids[ng++] = gv;
        }
        for (int i = 1; i < ng; i++) /* sorted, as decoder.py */
            for (int j = i; j > 0 && grids[j - 1] > grids[j]; j--) {
                int t = grids[j];
                grids[j] = grids[j - 1];
                grids[j - 1] = t;
            }
        for (int i = 0; i < npri; i++) {
            uint8_t a[77];
            for (int j = 0; j < ng; j++) { /* the sender's own CQ */
                memcpy(a, alt[i], 77);
                set_bits(a, 2, 28);
                a[28] = a[58] = 0;
                set_bits(a + 59, grids[j], 15);
                na = add_alt(alt, na, MAXALT, a);
            }
            int gv0 = get_bits(alt[i] + 59, 15);
            for (int k = gv0 < MAXGRID4 ? -1 : 0; k < 64; k++) {
                int gv = k < 0 ? gv0 : k < 3 ? MAXGRID4 + 2 + k : MAXGRID4 + 35 + (k - 3 - 30);
                for (int r = 0; r < 2; r++) {
                    memcpy(a, alt[i], 77);
                    a[58] = (uint8_t)r;
                    set_bits(a + 59, gv, 15);
                    if (sendable(a)) na = add_alt(alt, na, MAXALT, a);
                }
            }
        }
        for (int i = 0; i < na; i++) {
            uint8_t cw[174];
            int tones[NN];
            encode77(alt[i], cw, tones);
            for (int j = 0; j < 174; j++) S[(size_t)i * 174 + j] = cw[j] ? 1.0 : -1.0;
        }
        if (!bb_ready) {
            baseband_init(&bb, x, n, c->pool);
            bb_ready = 1;
        }
        double r1 = 0, r2 = 0;
        int k1 = -1;
        sync_hyp best_h;
        for (int i = 0; i < nm; i++) { /* each distinct location, in context order */
            int dup = 0;
            for (int j = 0; j < i && !dup; j++) dup = cx[mem[j]].f0 == cx[mem[i]].f0 && cx[mem[j]].dt == cx[mem[i]].dt;
            if (dup) continue;
            candidate loc = {cx[mem[i]].f0, cx[mem[i]].dt, 0.0, 0};
            int nh = fine_sync2(&bb, &loc, N_HYP, hyps);
            if (nh <= 0) continue;
            for (int h = 0; h < nh; h++) hp[h] = &hyps[h];
            if (run_nn(c, hp, nh, llr, feat)) {
                rc = -1;
                goto out;
            }
            for (int h = 0; h < nh; h++) {
                double l[174], sa = 0;
                for (int j = 0; j < 174; j++) {
                    l[j] = fmax(-REPEAT_CLIP, fmin(REPEAT_CLIP, llr[h * 174 + j]));
                    sa += fabs(l[j]);
                }
                int b1 = 0, b2 = -1;
                for (int a = 0; a < na; a++) {
                    double s = 0;
                    for (int j = 0; j < 174; j++) s += S[(size_t)a * 174 + j] * l[j];
                    rho[a] = s / (sa + 1e-9);
                }
                for (int a = 1; a < na; a++)
                    if (rho[a] > rho[b1]) b1 = a;
                for (int a = 0; a < na; a++)
                    if (a != b1 && (b2 < 0 || rho[a] > rho[b2])) b2 = a;
                if (k1 < 0 || rho[b1] > r1) {
                    r1 = rho[b1];
                    r2 = b2 >= 0 ? rho[b2] : REPEAT_SOLO; /* no sendable competitor (a CQ-only sender): rho >= 0.50 */
                    k1 = b1;
                    best_h = hyps[h];
                }
            }
        }
        if (k1 >= 0 && k1 < npri && r1 >= REPEAT_RHO && r1 - r2 >= REPEAT_MARGIN && rc < DEC_CAP) {
            dec *d = &all[rc++];
            d->final = 0;
            memcpy(d->bits77, alt[k1], 77);
            d->f0 = best_h.f0;
            d->dt = best_h.dt;
            d->method = FT8NN_REPEAT;
            d->metric = 0;
            d->pass = PASSES;
            d->snr = estimate_snr(&best_h, alt[k1]);
        }
    }
out:
    if (bb_ready) baseband_free(&bb);
    free(done);
    free(alt);
    free(S);
    free(rho);
    free(hyps);
    free(feat);
    return rc;
}

/* decoder._dt_shift: receiver DT offset from decoded (CRC-checked) DTs - their median, snapped to the lag grid.
 * Senders' clocks scatter by a few tenths of a second, so a median this far from 0 over this many decodes is the
 * receiver's own offset (a KiwiSDR stream that slips, or a sound card / clock that is off in an app). */
static int cmp_dbl(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

static double dt_recenter_shift(const dec *all, int nall, const dec *cx, int ncx) {
    int n = nall + ncx;
    if (n < DT_RECENTER_MIN) return 0.0;
    double *d = malloc(sizeof(double) * (size_t)n);
    for (int i = 0; i < nall; i++) d[i] = all[i].dt;
    for (int i = 0; i < ncx; i++) d[nall + i] = cx[i].dt;
    qsort(d, (size_t)n, sizeof(double), cmp_dbl);
    double med = n % 2 ? d[n / 2] : (d[n / 2 - 1] + d[n / 2]) / 2;
    free(d);
    if (med > DT_RECENTER_MAX) med = DT_RECENTER_MAX;
    if (med < -DT_RECENTER_MAX) med = -DT_RECENTER_MAX;
    return fabs(med) < DT_RECENTER ? 0.0 : nearbyint(med * 25.0) / 25.0; /* snap to the lag grid (25 frames/s) */
}

static once_t ap_once = ONCE_INIT, erase_once = ONCE_INIT;

static void to_public(const dec *d, ft8nn_decode_t *o) {
    memset(o, 0, sizeof *o);
    memcpy(o->text, d->text, sizeof o->text);
    o->f0 = d->f0;
    o->dt = d->dt;
    o->snr = d->snr;
    for (int i = 0; i < 77; i++) o->bits77[i / 8] |= d->bits77[i] << (7 - i % 8);
    o->method = d->method;
    o->metric = d->metric;
    o->pass = d->pass;
}

/* Unpack the decodes in all[from..n) that are final - every one if ctx_too, else all but context decodes, which a
 * later pass can still drop (drop_ctx_dupes) - in order, so the hash table learns them, and hand them to cb.
 * Only the final drop_ctx_dupes removes decodes from earlier passes, and only context ones, so everything else
 * is final when its pass ends. */
static void report(ft8nn_ctx *c, dec *all, int from, int n, int ctx_too, int pass, ft8nn_pass_cb cb, void *user,
                   ft8nn_decode_t *batch) {
    int nb = 0;
    for (int q = from; q < n; q++) {
        if (all[q].final || (!ctx_too && is_ctx(all[q].method))) continue;
        unpack77(all[q].bits77, c->ht, all[q].text, sizeof all[q].text);
        all[q].final = 1;
        if (cb) to_public(&all[q], &batch[nb++]);
    }
    if (cb) cb(batch, nb, pass, user);
}

static int decode(ft8nn_ctx *c, const float *audio, int n, int64_t t, ft8nn_decode_t *out, int max_out,
                  ft8nn_pass_cb cb, void *user) {
    run_once(&ap_once, ap_init);
    run_once(&erase_once, erase_init);
    if (n > SLOT_N) n = SLOT_N;
    if (n < NSPS) {
        if (cb) cb(NULL, 0, -1, user);
        return 0;
    }
    double *x = malloc(sizeof(double) * n);
    double m = 0, v = 0;
    for (int i = 0; i < n; i++) m += audio[i];
    m /= n;
    for (int i = 0; i < n; i++) v += (audio[i] - m) * (audio[i] - m);
    double sc = 1.0 / (sqrt(v / n) + 1e-9);
    for (int i = 0; i < n; i++) x[i] = audio[i] * sc;
    dec *all = malloc(sizeof(dec) * DEC_CAP), *cx = NULL;
    signed char *tried = malloc((size_t)TRIED_NB * TRIED_NL * 2); /* candidate -> pass last tried, -1 = never */
    memset(tried, -1, (size_t)TRIED_NB * TRIED_NL * 2);
    ft8nn_decode_t *batch = cb ? malloc(sizeof(ft8nn_decode_t) * DEC_CAP) : NULL;
    int nall = 0, ncx = 0;
    if (t != INT64_MIN) { /* context: this context's decodes 30 then 60 s earlier */
        cx = malloc(sizeof(dec) * 2 * DEC_CAP);
        for (int b = 30; b <= 60; b += 30)
            for (int i = 0; i < CTX_HIST; i++)
                if (c->hist[i].t == t - b) {
                    memcpy(cx + ncx, c->hist[i].d, sizeof(dec) * c->hist[i].n);
                    ncx += c->hist[i].n;
                }
    }
    for (int p = 0; p < PASSES; p++) {
        double shift = DT_RECENTER > 0 ? dt_recenter_shift(all, nall, cx, ncx) : 0.0;
        int k = decode_pass(c, x, n, all, nall, DEC_CAP, p, cx, ncx, tried, shift, cb, user, batch);
        if (k < 0) {
            free(x);
            free(all);
            free(cx);
            free(tried);
            free(batch);
            return -1;
        }
        /* report first: subtraction only prepares the next pass, and takes ~20 ms per decode */
        report(c, all, nall, nall + k, 0, p, cb, user, batch);
        double ts = tnow();
        for (int q = nall; q < nall + k; q++) { /* the last pass too: the co-channel and repeat stages work on the residual */
            uint8_t cw[174];
            int tones[NN];
            encode77(all[q].bits77, cw, tones);
            subtract_signal(x, n, tones, all[q].f0, all[q].dt, c->pool);
        }
        if (prof && k) fprintf(stderr, "subtract %d: %.2fs\n", k, tnow() - ts);
        nall += k;
        if (!k) break;
    }
    if (nall > 0) { /* x is now the residual */
        int r = cochannel_stage(c, x, n, all, nall);
        if (r < 0) {
            free(x);
            free(all);
            free(cx);
            free(tried);
            free(batch);
            return -1;
        }
        for (int q = nall; q < r; q++) {
            uint8_t cw[174];
            int tones[NN];
            encode77(all[q].bits77, cw, tones);
            subtract_signal(x, n, tones, all[q].f0, all[q].dt, c->pool);
        }
        nall = r;
    }
    if (ncx > 0) {
        int r = repeat_stage(c, x, n, all, nall, cx, ncx);
        if (r < 0) {
            free(x);
            free(all);
            free(cx);
            free(tried);
            free(batch);
            return -1;
        }
        nall = r;
    }
    nall = drop_ctx_dupes(all, 0, nall);
    report(c, all, 0, nall, 1, -1, cb, user, batch); /* the context decodes that survived, the co-channel and repeat stages' */
    free(batch);
    free(cx);
    free(tried);
    if (t != INT64_MIN) { /* remember: replace this slot's entry or the oldest */
        int w = 0;
        for (int i = 0; i < CTX_HIST; i++)
            if (c->hist[i].t == t || (c->hist[w].t != t && c->hist[i].t < c->hist[w].t)) w = i;
        c->hist[w].t = t;
        c->hist[w].n = nall;
        memcpy(c->hist[w].d, all, sizeof(dec) * nall);
    }
    for (int q = 0; q < nall && q < max_out; q++) to_public(&all[q], &out[q]);
    free(x);
    free(all);
    return nall;
}

int ft8nn_decode(ft8nn_ctx *c, const float *audio, int n, ft8nn_decode_t *out, int max_out) {
    return decode(c, audio, n, INT64_MIN, out, max_out, NULL, NULL);
}

int ft8nn_decode_at(ft8nn_ctx *c, const float *audio, int n, int64_t slot_time, ft8nn_decode_t *out, int max_out) {
    return decode(c, audio, n, slot_time, out, max_out, NULL, NULL);
}

int ft8nn_decode_stream(ft8nn_ctx *c, const float *audio, int n, int64_t slot_time, ft8nn_pass_cb cb, void *user,
                        ft8nn_decode_t *out, int max_out) {
    return decode(c, audio, n, slot_time, out, max_out, cb, user);
}
