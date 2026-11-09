/* Minimal RIFF/WAVE reader: mono PCM16 or float32 (same as c/tools/ft8nn_cli.c), plus the
 * yymmdd_HHMMSS slot-name parser. Used by jt9nn's file mode and by fake_wsjtx. */
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#define timegm _mkgmtime
#endif

/* Returns malloc'd samples (raw PCM values, any scale) and sets *n_out, *rate; NULL on failure. */
static float *wav_read(const char *path, int *n_out, int *rate) {
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
            int n = (int)(sz / (uint32_t)(bits / 8));
            x = (float *)malloc(sizeof(float) * (size_t)n);
            if (!x) goto out;
            if (bits == 16) {
                int16_t *s = (int16_t *)malloc(sz);
                if (!s) {
                    free(x);
                    x = NULL;
                    goto out;
                }
                n = (int)fread(s, 2, (size_t)n, f);
                for (int i = 0; i < n; i++) x[i] = s[i];
                free(s);
            } else {
                n = (int)fread(x, 4, (size_t)n, f);
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

/* Slot start of a yymmdd_HHMMSS basename (15 s boundary) as seconds since 1970 UTC, or INT64_MIN. */
static int64_t wav_slot_time(const char *name) {
    struct tm tm;
    char rest;
    memset(&tm, 0, sizeof tm);
    if (sscanf(name, "%2d%2d%2d_%2d%2d%2d%c", &tm.tm_year, &tm.tm_mon, &tm.tm_mday, &tm.tm_hour, &tm.tm_min,
               &tm.tm_sec, &rest) != 6 ||
        strlen(name) != 13 || tm.tm_mon < 1 || tm.tm_mon > 12 || tm.tm_mday < 1 || tm.tm_sec % 15)
        return INT64_MIN;
    tm.tm_year += 100;
    tm.tm_mon -= 1;
    return (int64_t)timegm(&tm);
}
