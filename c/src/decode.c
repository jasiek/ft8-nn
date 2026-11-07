/* Slot decoder (port of ft8nn/decoder.py decode_slot with nn_metric.make_llr_fn(also_classical=True)).
 *
 * Per pass: candidates -> fine_sync2 (2 hypotheses each) -> soft metrics [NN, classical nsym 1/2/3]
 * -> BP -> OSD-2 (NN metric only, gated; then on summed BP posteriors) -> context a-priori (decodes 30/60 s earlier,
 * ft8nn_decode_at only) -> CQ a-priori (NN metric only, gated) -> CRC.
 * Decodes are subtracted from the audio between passes.
 */
#include <math.h>
#include <onnxruntime/onnxruntime_c_api.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int prof = -1;
static double tnow(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

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
#define TRIED_LAG0 80 /* lag offset in the tried table: lags -80..79 frames (early track starts at -2.5 s = -63) */
#define TRIED_NL 160
#define TRIED_NB ((int)(FMAX / 3.125) + 1)
#define NOISE_BIN (3200.0 / 1000.0)
#define SNR_OFFSET 0.23

typedef struct {
    uint8_t bits77[77];
    double f0, dt, snr;
    int method, metric, pass;
} dec;

struct ft8nn_ctx {
    const OrtApi *api;
    OrtEnv *env;
    OrtSession *sess;
    OrtMemoryInfo *mem;
    hash_table *ht;
    struct {
        int64_t t; /* slot start, s; INT64_MIN = empty */
        int n;
        dec *d;
    } hist[CTX_HIST]; /* this context's own decodes of recent slots (ft8nn_decode_at) */
};

static pthread_once_t init_once = PTHREAD_ONCE_INIT;
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
    pthread_once(&init_once, global_init);
    ft8nn_ctx *c = calloc(1, sizeof *c);
    c->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    const OrtApi *api = c->api;
    OrtSessionOptions *so = NULL;
    if (ort_check(api, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ft8nn", &c->env)) ||
        ort_check(api, api->CreateSessionOptions(&so)) ||
        (n_threads > 0 && ort_check(api, api->SetIntraOpNumThreads(so, n_threads))) ||
        (n_threads > 0 && ort_check(api, api->SetInterOpNumThreads(so, 1))) ||
        ort_check(api, api->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL)) ||
        ort_check(api, api->CreateSession(c->env, onnx_path, so, &c->sess)) ||
        ort_check(api, api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &c->mem))) {
        if (so) api->ReleaseSessionOptions(so);
        ft8nn_close(c);
        return NULL;
    }
    api->ReleaseSessionOptions(so);
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
    if (c->env) api->ReleaseEnv(c->env);
    ht_free(c->ht);
    for (int i = 0; i < CTX_HIST; i++) free(c->hist[i].d);
    free(c);
}

void ft8nn_reset_hashes(ft8nn_ctx *c) { ht_clear(c->ht); }

/* NN LLRs for nh hypotheses -> llr[h*174 ...] */
static int run_nn(ft8nn_ctx *c, const sync_hyp *hyps, int nh, double *llr) {
    const OrtApi *api = c->api;
    float *feat = malloc(sizeof(float) * NN_BATCH * NN * NFEAT);
    const char *in_names[] = {"features"}, *out_names[] = {"llr"};
    int rc = 0;
    for (int i0 = 0; i0 < nh && !rc; i0 += NN_BATCH) {
        int b = nh - i0 < NN_BATCH ? nh - i0 : NN_BATCH;
        for (int i = 0; i < b; i++) nn_features(hyps[i0 + i].y, hyps[i0 + i].start, feat + (size_t)i * NN * NFEAT);
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
    free(feat);
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
    uint8_t drop[n > 0 ? n : 1];
    memset(drop, 0, sizeof drop);
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

/* One pass. Appends new decodes (payload not in all[0..nall)) to all; returns count added, -1 on error. */
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

static int decode_pass(ft8nn_ctx *c, const double *x, int n, dec *all, int nall, int cap, int pass, const dec *cx, int ncx,
                       signed char *tried) {
    if (prof < 0) prof = getenv("FT8NN_PROFILE") != NULL;
    double t0 = tnow(), t1, t2, t3, t4, t5;
    long n_bp = 0, n_osd = 0;
    candidate *cands = malloc(sizeof(candidate) * (MAX_CAND + (int)(FMAX / 3.125) + 1));
    int nc = find_candidates(x, n, 150.0, FMAX, MAX_CAND, 1.3, BASE_MEDIAN, EARLY_DT, EARLY_MIN, cands);
    t1 = tnow();
    baseband bb;
    baseband_init(&bb, x, n);
    sync_hyp *hyps = malloc(sizeof(sync_hyp) * (size_t)(nc * N_HYP + 1));
    int *owner = malloc(sizeof(int) * (size_t)(nc * N_HYP + 1));
    int nh = 0, nskip = 0;
    for (int ci = 0; ci < nc; ci++) {
        if (REUSE_DF > 0 && pass > 0 && reuse_skip(&cands[ci], tried, all, nall)) {
            nskip++;
            continue;
        }
        int ti = tried_idx(&cands[ci]);
        if (ti >= 0) tried[ti] = (signed char)pass;
        int k = fine_sync2(&bb, &cands[ci], N_HYP, hyps + nh);
        for (int q = 0; q < k; q++) owner[nh + q] = ci;
        nh += k;
    }
    baseband_free(&bb);
    t2 = tnow();
    t3 = t4 = t5 = t2;
    int added = 0;
    if (nh == 0) goto done;
    double *L = malloc(sizeof(double) * (size_t)nh * NMET * 174);
    double *nnl = malloc(sizeof(double) * (size_t)nh * 174);
    int nn_rc = run_nn(c, hyps, nh, nnl);
    t3 = tnow();
    if (nn_rc) {
        free(L);
        free(nnl);
        added = -1;
        goto done;
    }
    for (int h = 0; h < nh; h++) {
        double *row = L + (size_t)h * NMET * 174;
        memcpy(row, nnl + (size_t)h * 174, sizeof(double) * 174);
        cplx cs[NN][8];
        symbol_spectra8(hyps[h].y, hyps[h].start, cs);
        for (int k = 1; k < NMET; k++) coherent_llr(cs, k, row + k * 174);
    }
    free(nnl);
    t4 = tnow();
    uint8_t *done_cand = calloc(nc, 1);
    int pass_start = nall;
    for (int h = 0; h < nh; h++) {
        int ci = owner[h];
        if (done_cand[ci]) continue;
        int near = 0;
        for (int q = pass_start; q < nall + added; q++)
            if (fabs(hyps[h].f0 - all[q].f0) < 3 && fabs(hyps[h].dt - all[q].dt) < 0.05) near = 1;
        if (near) {
            done_cand[ci] = 1;
            continue;
        }
        for (int k = 0; k < NMET; k++) {
            const double *l = L + ((size_t)h * NMET + k) * 174;
            double exp_err = 0;
            for (int i = 0; i < 174; i++) exp_err += 1.0 / (1.0 + exp(fabs(l[i])));
            uint8_t bits[174], a91[91];
            int nhard, method = -1;
            double dist;
            n_bp++;
            /* early-DT candidates have weaker sync evidence: no OSD on BP sums or under a-priori bits, no erasure stages */
            int early = cands[ci].early;
            if (bp_decode(l, 30, bits, &nhard) && crc_ok(bits)) {
                method = FT8NN_BP;
                memcpy(a91, bits, 91);
            } else if (k < 1 && exp_err < OSD_GATE && ++n_osd) {
                if (osd_decode(l, OSD_ORDER, bits, &dist, &nhard) && nhard <= OSD_MAX_HARD) {
                    method = FT8NN_OSD;
                    memcpy(a91, bits, 91);
                } else if (OSD_SNAPS > 0 && !early) {
                    method = try_osd_bp(l, a91);
                }
            }
            if (method < 0 && k == 0 && ncx > 0 && !early && exp_err < CTX_GATE)
                method = try_ctx(l, cx, ncx, hyps[h].f0, hyps[h].dt, a91);
            if (method < 0 && k == 0 && exp_err < AP_GATE) method = try_ap(l, a91, early ? -1 : OSD_ORDER);
            if (method < 0 && k == 0 && exp_err < ERASE_GATE && !early) method = try_erase(l, a91);
            if (method < 0) continue;
            if (!any_bits(a91, 77)) break; /* all-zero payload: classic false decode */
            int known = 0;
            for (int q = 0; q < nall + added; q++)
                if (same_payload(all[q].bits77, a91)) known = 1;
            if (!known && nall + added < cap) {
                dec *d = &all[nall + added++];
                memcpy(d->bits77, a91, 77);
                d->f0 = hyps[h].f0;
                d->dt = hyps[h].dt;
                d->method = method;
                d->metric = k;
                d->pass = pass;
                d->snr = estimate_snr(&hyps[h], a91);
            }
            done_cand[ci] = 1;
            break;
        }
    }
    /* drop osd_er / osd_bp ghosts of another decode (earlier pass, or this pass and not ghost-prone), as
     * decoder._decode_pass (GHOST_PRONE) */
    {
        uint8_t drop[added > 0 ? added : 1];
        memset(drop, 0, sizeof drop);
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
    free(done_cand);
    free(L);
    t5 = tnow();
done:
    if (prof)
        fprintf(stderr, "pass %d: cand %d skip %d hyp %d | cand %.2fs sync %.2fs nn %.2fs classical %.2fs bp/osd %.2fs (bp %ld osd %ld) -> %d\n",
                pass, nc, nskip, nh, t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, n_bp, n_osd, added);
    free(cands); /* candidate flags (early) are read in the decode loop */
    free(hyps);
    free(owner);
    return added;
}

static pthread_once_t ap_once = PTHREAD_ONCE_INIT, erase_once = PTHREAD_ONCE_INIT;

static int decode(ft8nn_ctx *c, const float *audio, int n, int64_t t, ft8nn_decode_t *out, int max_out) {
    pthread_once(&ap_once, ap_init);
    pthread_once(&erase_once, erase_init);
    if (n > SLOT_N) n = SLOT_N;
    if (n < NSPS) return 0;
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
        int k = decode_pass(c, x, n, all, nall, DEC_CAP, p, cx, ncx, tried);
        if (k < 0) {
            free(x);
            free(all);
            free(cx);
            free(tried);
            return -1;
        }
        for (int q = nall; q < nall + k; q++) {
            uint8_t cw[174];
            int tones[NN];
            encode77(all[q].bits77, cw, tones);
            subtract_signal(x, n, tones, all[q].f0, all[q].dt);
        }
        nall += k;
        if (!k) break;
    }
    nall = drop_ctx_dupes(all, 0, nall);
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
    for (int q = 0; q < nall; q++) {
        char text[48];
        unpack77(all[q].bits77, c->ht, text, sizeof text); /* always, so the hash table learns */
        if (q >= max_out) continue;
        ft8nn_decode_t *o = &out[q];
        memset(o, 0, sizeof *o);
        memcpy(o->text, text, sizeof text);
        o->f0 = all[q].f0;
        o->dt = all[q].dt;
        o->snr = all[q].snr;
        for (int i = 0; i < 77; i++) o->bits77[i / 8] |= all[q].bits77[i] << (7 - i % 8);
        o->method = all[q].method;
        o->metric = all[q].metric;
        o->pass = all[q].pass;
    }
    free(x);
    free(all);
    return nall;
}

int ft8nn_decode(ft8nn_ctx *c, const float *audio, int n, ft8nn_decode_t *out, int max_out) {
    return decode(c, audio, n, INT64_MIN, out, max_out);
}

int ft8nn_decode_at(ft8nn_ctx *c, const float *audio, int n, int64_t slot_time, ft8nn_decode_t *out, int max_out) {
    return decode(c, audio, n, slot_time, out, max_out);
}
