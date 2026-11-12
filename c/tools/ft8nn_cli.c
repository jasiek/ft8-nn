/* ft8nn_cli: decode 15 s WAV slots with libft8nn.
 *
 *   ft8nn_cli [--threads N] [--time] [--stream] model.onnx slot1.wav [slot2.wav ...]
 *
 * One context for all files, so callsign hashes carry over in the order given (pass slots in
 * time order). Files named yymmdd_HHMMSS.wav are decoded with ft8nn_decode_at (context a-priori from the
 * slots 30 and 60 s earlier); other names with ft8nn_decode. Output, one line per decode, same format as bench/run_real.py decodes_<tag>.txt:
 *   <slot>\t<f0 %.1f>\t<dt %.2f>\t<method><metric>\t<text>\t<77 bits as 20 hex digits>\t<snr %.1f>
 * --threads N: decode threads (default 1; 0 = one per performance core).
 * --time also writes "time\t<slot>\t<decode wall s %.3f>\t<decodes>" per slot to stderr.
 * --stream decodes with ft8nn_decode_stream and prints the decodes in the order they are handed out, writing
 * "batch\t<slot>\t<pass>\t<s since the decode started %.3f>\t<decodes>" to stderr for every callback.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../include/ft8nn.h"

#ifdef _WIN32 /* native Windows (MinGW-w64 in CI): the C runtime's names */
#define timegm _mkgmtime
static void mono_now(struct timespec *t) { timespec_get(t, TIME_UTC); }
#else
static void mono_now(struct timespec *t) { clock_gettime(CLOCK_MONOTONIC, t); }
#endif

/* Minimal RIFF/WAVE reader: mono PCM16 or float32. Returns samples (malloc'd), or NULL. */
static float *read_wav(const char *path, int *n_out, int *rate) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint8_t h[12];
    float *x = NULL;
    int fmt = 0, ch = 0, bits = 0;
    if (fread(h, 1, 12, f) != 12 || memcmp(h, "RIFF", 4) || memcmp(h + 8, "WAVE", 4)) goto out;
    for (;;) {
        uint8_t ck[8];
        if (fread(ck, 1, 8, f) != 8) break;
        uint32_t sz = ck[4] | ck[5] << 8 | ck[6] << 16 | (uint32_t)ck[7] << 24;
        if (!memcmp(ck, "fmt ", 4)) {
            uint8_t b[40] = {0};
            size_t rd = sz < sizeof b ? sz : sizeof b;
            if (fread(b, 1, rd, f) != rd) goto out;
            if (sz > rd) fseek(f, (long)(sz - rd), SEEK_CUR);
            fmt = b[0] | b[1] << 8;
            ch = b[2] | b[3] << 8;
            *rate = b[4] | b[5] << 8 | b[6] << 16 | b[7] << 24;
            bits = b[14] | b[15] << 8;
        } else if (!memcmp(ck, "data", 4)) {
            if (ch != 1 || !((fmt == 1 && bits == 16) || (fmt == 3 && bits == 32))) goto out;
            int n = (int)(sz / (bits / 8));
            x = malloc(sizeof(float) * n);
            if (bits == 16) {
                int16_t *s = malloc(sz);
                n = (int)(fread(s, 2, n, f));
                for (int i = 0; i < n; i++) x[i] = s[i];
                free(s);
            } else {
                n = (int)fread(x, 4, n, f);
            }
            *n_out = n;
            break;
        } else {
            fseek(f, (long)(sz + (sz & 1)), SEEK_CUR);
        }
    }
out:
    fclose(f);
    return x;
}

static const char *method_name(int m) {
    static const char *names[] = {"bp",     "osd",    "bp_ap0", "osd_ap0", "bp_er",  "osd_er", "osd_bp",
                                  "bp_cx",  "osd_cx", "rp",     "bp_cc",   "osd_cc", "osd_bp_cc"};
    return m >= 0 && m < 13 ? names[m] : "?";
}

static void print_decode(const char *slot, const ft8nn_decode_t *d) {
    /* 77-bit payload as a 20-digit hex number (Python: f"{int(bits, 2):020x}"): 4-bit groups from the
     * least significant end, the top digit holding the first bit */
    char hx[21];
    for (int q = 19, e = 77; q >= 0; q--, e -= 4) {
        int v = 0;
        for (int b = e - 4 < 0 ? 0 : e - 4; b < e; b++) v = 2 * v + ((d->bits77[b / 8] >> (7 - b % 8)) & 1);
        hx[q] = "0123456789abcdef"[v];
    }
    hx[20] = 0;
    printf("%s\t%.1f\t%.2f\t%s%d\t%s\t%s\t%.1f\n", slot, d->f0, d->dt, method_name(d->method), d->metric, d->text, hx,
           d->snr);
}

static double secs(const struct timespec *a, const struct timespec *b) {
    return b->tv_sec - a->tv_sec + 1e-9 * (b->tv_nsec - a->tv_nsec);
}

typedef struct {
    const char *slot;
    struct timespec t0;
} stream_st;

static void on_batch(const ft8nn_decode_t *d, int n, int pass, void *user) {
    stream_st *s = user;
    struct timespec t;
    mono_now(&t);
    fprintf(stderr, "batch\t%s\t%d\t%.3f\t%d\n", s->slot, pass, secs(&s->t0, &t), n);
    for (int k = 0; k < n; k++) print_decode(s->slot, &d[k]);
    fflush(stdout);
}

/* slot start from a yymmdd_HHMMSS name on a 15 s boundary, s since 1970 (UTC); INT64_MIN otherwise (e.g. the
 * numbered synthetic test files) */
static int64_t slot_time(const char *slot) {
    struct tm tm = {0};
    char rest;
    if (sscanf(slot, "%2d%2d%2d_%2d%2d%2d%c", &tm.tm_year, &tm.tm_mon, &tm.tm_mday, &tm.tm_hour, &tm.tm_min, &tm.tm_sec,
               &rest) != 6 || strlen(slot) != 13 || tm.tm_mon < 1 || tm.tm_mon > 12 || tm.tm_mday < 1 || tm.tm_sec % 15)
        return INT64_MIN;
    tm.tm_year += 100;
    tm.tm_mon -= 1;
    return (int64_t)timegm(&tm);
}

int main(int argc, char **argv) {
    int threads = 1, timing = 0, stream = 0, a = 1;
    for (;;) {
        if (a + 1 < argc && !strcmp(argv[a], "--threads")) {
            threads = atoi(argv[a + 1]);
            a += 2;
        } else if (a < argc && !strcmp(argv[a], "--time")) {
            timing = 1;
            a++;
        } else if (a < argc && !strcmp(argv[a], "--stream")) {
            stream = 1;
            a++;
        } else
            break;
    }
    if (argc - a < 2) {
        fprintf(stderr, "usage: %s [--threads N] [--time] [--stream] model.onnx slot.wav...\n", argv[0]);
        return 2;
    }
    ft8nn_ctx *ctx = ft8nn_open(argv[a], threads);
    if (!ctx) return 1;
    ft8nn_decode_t out[256];
    for (int i = a + 1; i < argc; i++) {
        int n = 0, rate = 0;
        float *x = read_wav(argv[i], &n, &rate);
        if (!x || rate != 12000) {
            fprintf(stderr, "%s: need mono 12 kHz PCM16/float32 WAV\n", argv[i]);
            free(x);
            continue;
        }
        /* slot name = file name without directory and extension; Windows paths may also use '\' and 'D:' */
        const char *base = argv[i];
        for (const char *p = argv[i]; *p; p++)
#ifdef _WIN32
            if (*p == '/' || *p == '\\' || *p == ':') base = p + 1;
#else
            if (*p == '/') base = p + 1;
#endif
        char slot[256];
        snprintf(slot, sizeof slot, "%s", base);
        char *dot = strrchr(slot, '.');
        if (dot) *dot = 0;
        int64_t t = slot_time(slot);
        struct timespec t0, t1;
        mono_now(&t0);
        stream_st st = {slot, t0};
        int nd = stream ? ft8nn_decode_stream(ctx, x, n, t, on_batch, &st, NULL, 0)
                 : t == INT64_MIN ? ft8nn_decode(ctx, x, n, out, 256)
                                  : ft8nn_decode_at(ctx, x, n, t, out, 256);
        mono_now(&t1);
        if (timing) fprintf(stderr, "time\t%s\t%.3f\t%d\n", slot, secs(&t0, &t1), nd);
        free(x);
        if (nd < 0) {
            fprintf(stderr, "%s: decode failed\n", argv[i]);
            continue;
        }
        for (int k = 0; !stream && k < nd && k < 256; k++) print_decode(slot, &out[k]);
        fflush(stdout);
    }
    ft8nn_close(ctx);
    return 0;
}
