/* FT8 77-bit message unpacking (port of ft8nn/message.py, itself after WSJT-X packjt77). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "call_blocks.h"
#include "internal.h"


#define NTOKENS 2063592u
#define MAX22 4194304u
#define MAXGRID4 32400

static const char C1[] = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
static const char C2[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
static const char C3[] = "0123456789";
static const char C4[] = " ABCDEFGHIJKLMNOPQRSTUVWXYZ";
static const char C38[] = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/";
static const char C42[] = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-./?";

/* ---- hash table: newest entry wins, like the Python dicts (later add overwrites) */
typedef struct {
    char call[12];
    uint32_t h10, h12, h22;
} ht_ent;

struct hash_table {
    ht_ent *e;
    int n, cap;
};

hash_table *ht_new(void) { return calloc(1, sizeof(hash_table)); }
void ht_free(hash_table *h) {
    if (!h) return;
    free(h->e);
    free(h);
}
void ht_clear(hash_table *h) { h->n = 0; }

static uint32_t ihashcall(const char *call, int m) {
    char c[12];
    /* strip, upper, pad to 11 with spaces */
    while (*call == ' ') call++;
    int len = (int)strlen(call);
    while (len > 0 && call[len - 1] == ' ') len--;
    for (int i = 0; i < 11; i++) {
        char ch = i < len ? call[i] : ' ';
        if (ch >= 'a' && ch <= 'z') ch -= 32;
        c[i] = ch;
    }
    uint64_t n = 0;
    for (int i = 0; i < 11; i++) {
        const char *p = strchr(C38, c[i]);
        int v = (p && c[i]) ? (int)(p - C38) : 0;
        n = 38 * n + (uint64_t)v;
    }
    return (uint32_t)((47055833459ULL * n) >> (64 - m));
}

static void ht_add(hash_table *h, const char *call) {
    int len = (int)strlen(call);
    if (len < 3 || call[0] == '<' || !strcmp(call, "CQ") || !strcmp(call, "DE") || !strcmp(call, "QRZ")) return;
    for (int i = 0; i < h->n; i++)
        if (!strcmp(h->e[i].call, call)) { /* move to newest */
            ht_ent t = h->e[i];
            memmove(h->e + i, h->e + i + 1, sizeof(ht_ent) * (h->n - i - 1));
            h->e[h->n - 1] = t;
            return;
        }
    if (h->n == h->cap) {
        h->cap = h->cap ? 2 * h->cap : 256;
        h->e = realloc(h->e, sizeof(ht_ent) * h->cap);
    }
    ht_ent *e = &h->e[h->n++];
    snprintf(e->call, sizeof e->call, "%s", call);
    e->h10 = ihashcall(call, 10);
    e->h12 = ihashcall(call, 12);
    e->h22 = ihashcall(call, 22);
}

static void ht_lookup(const hash_table *h, int bits, uint32_t v, char *out, int outlen) {
    if (h)
        for (int i = h->n - 1; i >= 0; i--) {
            uint32_t hv = bits == 10 ? h->e[i].h10 : bits == 12 ? h->e[i].h12 : h->e[i].h22;
            if (hv == v) {
                snprintf(out, outlen, "<%s>", h->e[i].call);
                return;
            }
        }
    snprintf(out, outlen, "<...>");
}

/* bits b[lo..hi) as an unsigned integer, MSB first; hi - lo <= 64 */
static uint64_t bint(const uint8_t *b, int lo, int hi) {
    uint64_t v = 0;
    for (int i = lo; i < hi; i++) v = (v << 1) | b[i];
    return v;
}

/* v[0..n) (bits, MSB first) as a big unsigned integer: divide it by d in place (bit-serial long division) and
 * return the remainder. Replaces 128-bit arithmetic, which not every C compiler has. */
static int divmod_bits(uint8_t *v, int n, int d) {
    int r = 0;
    for (int i = 0; i < n; i++) {
        r = 2 * r + v[i];
        v[i] = r >= d;
        if (v[i]) r -= d;
    }
    return r;
}

static void strip(char *s) {
    int len = (int)strlen(s), a = 0;
    while (len > 0 && s[len - 1] == ' ') s[--len] = 0;
    while (s[a] == ' ') a++;
    if (a) memmove(s, s + a, len - a + 1);
}

static void unpack28(uint32_t n28, hash_table *ht, char *out, int outlen) {
    if (n28 < NTOKENS) {
        if (n28 <= 2) {
            snprintf(out, outlen, "%s", n28 == 0 ? "DE" : n28 == 1 ? "QRZ" : "CQ");
            return;
        }
        if (n28 <= 1002) {
            snprintf(out, outlen, "CQ %03u", n28 - 3);
            return;
        }
        uint32_t n = n28 - 1003;
        char s[5];
        for (int i = 3; i >= 0; i--) {
            s[i] = C4[n % 27];
            n /= 27;
        }
        s[4] = 0;
        strip(s);
        snprintf(out, outlen, "CQ %s", s);
        return;
    }
    n28 -= NTOKENS;
    if (n28 < MAX22) {
        ht_lookup(ht, 22, n28, out, outlen);
        return;
    }
    uint32_t n = n28 - MAX22;
    uint32_t i1 = n / (36 * 10 * 27 * 27 * 27);
    n %= 36 * 10 * 27 * 27 * 27;
    uint32_t i2 = n / (10 * 27 * 27 * 27);
    n %= 10 * 27 * 27 * 27;
    uint32_t i3 = n / (27 * 27 * 27);
    n %= 27 * 27 * 27;
    uint32_t i4 = n / (27 * 27);
    n %= 27 * 27;
    uint32_t i5 = n / 27, i6 = n % 27;
    if (i1 > 36) { /* out of range for a standard call (Python would raise) */
        snprintf(out, outlen, "<...>");
        return;
    }
    char c[7] = {C1[i1], C2[i2], C3[i3], C4[i4], C4[i5], C4[i6], 0};
    strip(c);
    if (ht) ht_add(ht, c);
    snprintf(out, outlen, "%s", c);
}

static void report(int ir, uint32_t igrid4, char *out, int outlen) {
    if (igrid4 <= MAXGRID4) {
        uint32_t j1 = igrid4 / 1800, r = igrid4 % 1800, j2 = r / 100;
        r %= 100;
        snprintf(out, outlen, "%s%c%c%u%u", ir ? "R " : "", 'A' + j1, 'A' + j2, r / 10, r % 10);
        return;
    }
    int irpt = (int)igrid4 - MAXGRID4;
    if (irpt == 1) {
        out[0] = 0;
        return;
    }
    if (irpt >= 2 && irpt <= 4) {
        snprintf(out, outlen, "%s", irpt == 2 ? "RRR" : irpt == 3 ? "RR73" : "73");
        return;
    }
    int isnr = irpt - 35;
    if (isnr > 50) isnr -= 101;
    snprintf(out, outlen, "%s%+03d", ir ? "R" : "", isnr);
}

/* hex of the integer b[0..nb) (bits, MSB first), no leading zeros, then zero padded to at least width digits */
static void hex_bits(const uint8_t *b, int nb, int width, int upper, char *out) {
    char t[40];
    int n = 0;
    for (int e = nb; e > 0; e -= 4) { /* 4-bit groups from the least significant end */
        int d = 0;
        for (int i = e - 4 < 0 ? 0 : e - 4; i < e; i++) d = 2 * d + b[i];
        t[n++] = (char)(d < 10 ? '0' + d : (upper ? 'A' : 'a') + d - 10);
    }
    while (n > 1 && t[n - 1] == '0') n--;
    while (n < width) t[n++] = '0';
    for (int i = 0; i < n; i++) out[i] = t[n - 1 - i];
    out[n] = 0;
}

void unpack77(const uint8_t *b, hash_table *ht, char *out, int outlen) {
    int i3 = (int)bint(b, 74, 77);
    char c1[32], c2[32], rpt[16], h[32];
    if (i3 == 1 || i3 == 2) {
        uint32_t n28a = (uint32_t)bint(b, 0, 28), n28b = (uint32_t)bint(b, 29, 57), igrid4 = (uint32_t)bint(b, 59, 74);
        int ipa = b[28], ipb = b[57], ir = b[58];
        const char *sfx = i3 == 1 ? "/R" : "/P";
        unpack28(n28a, ht, c1, sizeof c1);
        unpack28(n28b, ht, c2, sizeof c2);
        if (ipa && c1[0] != '<') strncat(c1, sfx, sizeof c1 - strlen(c1) - 1);
        if (ipb && c2[0] != '<') strncat(c2, sfx, sizeof c2 - strlen(c2) - 1);
        report(ir, igrid4, rpt, sizeof rpt);
        out[0] = 0;
        const char *parts[3] = {c1, c2, rpt};
        for (int i = 0; i < 3; i++) {
            if (!parts[i][0]) continue;
            if (out[0]) strncat(out, " ", outlen - strlen(out) - 1);
            strncat(out, parts[i], outlen - strlen(out) - 1);
        }
        return;
    }
    if (i3 == 4) {
        uint32_t n12 = (uint32_t)bint(b, 0, 12);
        uint64_t n58 = (uint64_t)bint(b, 12, 70);
        int iflip = b[70], nrpt = (int)bint(b, 71, 73), icq = b[73];
        char s[12];
        for (int i = 10; i >= 0; i--) {
            s[i] = C38[n58 % 38];
            n58 /= 38;
        }
        s[11] = 0;
        strip(s);
        if (ht) ht_add(ht, s);
        ht_lookup(ht, 12, n12, h, sizeof h);
        if (icq) {
            snprintf(out, outlen, "CQ %s", s);
            return;
        }
        static const char *rr[4] = {"", " RRR", " RR73", " 73"};
        if (iflip) snprintf(out, outlen, "%s %s%s", s, h, rr[nrpt]);
        else snprintf(out, outlen, "%s %s%s", h, s, rr[nrpt]);
        return;
    }
    char hx[40];
    if (i3 == 0) {
        int n3 = (int)bint(b, 71, 74);
        if (n3 == 0) {
            uint8_t v[71];
            memcpy(v, b, sizeof v);
            char s[14];
            for (int i = 12; i >= 0; i--) s[i] = C42[divmod_bits(v, 71, 42)];
            s[13] = 0;
            strip(s);
            snprintf(out, outlen, "%s", s);
            return;
        }
        if (n3 == 1) {
            unpack28((uint32_t)bint(b, 0, 28), ht, c1, sizeof c1);
            unpack28((uint32_t)bint(b, 28, 56), ht, c2, sizeof c2);
            ht_lookup(ht, 10, (uint32_t)bint(b, 56, 66), h, sizeof h);
            snprintf(out, outlen, "%s RR73; %s %s %+03d", c1, c2, h, 2 * (int)bint(b, 66, 71) - 30);
            return;
        }
        if (n3 == 5) {
            hex_bits(b, 71, 18, 1, hx);
            const char *p = hx;
            while (*p == '0') p++;
            snprintf(out, outlen, "%s", *p ? p : "0");
            return;
        }
        hex_bits(b, 71, 18, 0, hx);
        snprintf(out, outlen, "<0.%d:%s>", n3, hx);
        return;
    }
    hex_bits(b, 74, 19, 0, hx);
    snprintf(out, outlen, "<%d:%s>", i3, hx);
}

/* first character a whole call block (K, W, ...) or the first two a block in use (tools/gen_call_blocks.py) */
static int call_block_ok(const char *c) {
    if (c[0] && strchr(CALL_BLOCKS1, c[0])) return 1;
    for (int i = 0; i < N_CALL_BLOCKS2; i++)
        if (c[0] == CALL_BLOCKS2[i][0] && c[1] == CALL_BLOCKS2[i][1]) return 1;
    return 0;
}

/* message.plausible: 0 if a standard message (i3 = 1, 2) could not be a real transmission (CQ with the R flag, locator
 * past RR99, report code above 135; with deep also a malformed full callsign or one outside the call blocks) */
int plausible77(const uint8_t *b, int deep) {
    int i3 = (int)bint(b, 74, 77);
    if (i3 != 1 && i3 != 2) return 1;
    uint32_t n28a = (uint32_t)bint(b, 0, 28), n28b = (uint32_t)bint(b, 29, 57), igrid4 = (uint32_t)bint(b, 59, 74);
    if (igrid4 == MAXGRID4) return 0;
    int irpt = (int)igrid4 - MAXGRID4;
    if (irpt > 135) return 0; /* encoders use 5..135 (negative reports as 86..135 or 5..34) */
    if (n28a >= 2 && n28a < NTOKENS && b[58]) return 0;                    /* CQ ... with the R flag */
    if (deep) {
        const uint32_t ns[2] = {n28a, n28b};
        for (int i = 0; i < 2; i++) {
            if (ns[i] < NTOKENS + MAX22) continue; /* token or hashed call */
            if ((ns[i] - NTOKENS - MAX22) / (36u * 10 * 27 * 27 * 27) > 36) return 0;
            char c[16];
            unpack28(ns[i], NULL, c, sizeof c);
            if (strchr(c, ' ') || !call_block_ok(c)) return 0;
        }
    }
    return 1;
}
