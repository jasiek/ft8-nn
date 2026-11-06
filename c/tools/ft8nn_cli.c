/* ft8nn_cli: decode 15 s WAV slots with libft8nn.
 *
 *   ft8nn_cli [--threads N] model.onnx slot1.wav [slot2.wav ...]
 *
 * One context for all files, so callsign hashes carry over in the order given (pass slots in
 * time order). Files named yymmdd_HHMMSS.wav are decoded with ft8nn_decode_at (context a-priori from the
 * slots 30 and 60 s earlier); other names with ft8nn_decode. Output, one line per decode, same format as bench/run_real.py decodes_<tag>.txt:
 *   <slot>\t<f0 %.1f>\t<dt %.2f>\t<method><metric>\t<text>\t<77 bits as 20 hex digits>\t<snr %.1f>
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../include/ft8nn.h"

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
    static const char *names[] = {"bp", "osd", "bp_ap0", "osd_ap0", "bp_er", "osd_er", "osd_bp", "bp_cx", "osd_cx"};
    return m >= 0 && m < 9 ? names[m] : "?";
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
    int threads = 1, a = 1;
    if (a + 1 < argc && !strcmp(argv[a], "--threads")) {
        threads = atoi(argv[a + 1]);
        a += 2;
    }
    if (argc - a < 2) {
        fprintf(stderr, "usage: %s [--threads N] model.onnx slot.wav...\n", argv[0]);
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
        const char *base = strrchr(argv[i], '/');
        base = base ? base + 1 : argv[i];
        char slot[256];
        snprintf(slot, sizeof slot, "%s", base);
        char *dot = strrchr(slot, '.');
        if (dot) *dot = 0;
        int64_t t = slot_time(slot);
        int nd = t == INT64_MIN ? ft8nn_decode(ctx, x, n, out, 256) : ft8nn_decode_at(ctx, x, n, t, out, 256);
        free(x);
        if (nd < 0) {
            fprintf(stderr, "%s: decode failed\n", argv[i]);
            continue;
        }
        for (int k = 0; k < nd && k < 256; k++) {
            const ft8nn_decode_t *d = &out[k];
            /* 77-bit payload as a 20-digit hex number (Python: f"{int(bits, 2):020x}") */
            unsigned __int128 v = 0;
            for (int b = 0; b < 77; b++) v = (v << 1) | ((d->bits77[b / 8] >> (7 - b % 8)) & 1);
            char hx[21];
            for (int q = 19; q >= 0; q--) {
                hx[q] = "0123456789abcdef"[(int)(v & 15)];
                v >>= 4;
            }
            hx[20] = 0;
            printf("%s\t%.1f\t%.2f\t%s%d\t%s\t%s\t%.1f\n", slot, d->f0, d->dt, method_name(d->method), d->metric, d->text, hx,
                   d->snr);
        }
        fflush(stdout);
    }
    ft8nn_close(ctx);
    return 0;
}
