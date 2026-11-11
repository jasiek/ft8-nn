/* libft8nn: FT8 decoder with a neural soft demodulator.
 *
 * One context per thread. A context owns the ONNX Runtime session and the callsign hash table,
 * which learns from every decoded full callsign (like a running WSJT-X session), so feed slots
 * in time order.
 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ft8nn_ctx ft8nn_ctx;

enum { FT8NN_BP = 0, FT8NN_OSD = 1, FT8NN_BP_AP = 2, FT8NN_OSD_AP = 3, FT8NN_BP_ERASE = 4, FT8NN_OSD_ERASE = 5, FT8NN_OSD_BP = 6,
       FT8NN_BP_CTX = 7, FT8NN_OSD_CTX = 8 };

typedef struct {
    char text[48];      /* decoded message, WSJT-X formatting; hashed calls as <CALL> or <...> */
    double f0;          /* Hz, frequency of tone 0 */
    double dt;          /* s, start time relative to the nominal 0.5 s into the slot */
    double snr;         /* dB in 2500 Hz */
    uint8_t bits77[10]; /* payload, MSB first; bit i is (bits77[i/8] >> (7 - i%8)) & 1 */
    int method;         /* FT8NN_BP, FT8NN_OSD, FT8NN_BP_AP (CQ a-priori), FT8NN_OSD_AP,
                           FT8NN_BP_ERASE (BP with a burst of symbols erased: message changed mid-frame),
                           FT8NN_OSD_ERASE (OSD-0 with a wider burst of 20-24 symbols erased),
                           FT8NN_OSD_BP (OSD on the summed BP posteriors of the first iterations),
                           FT8NN_BP_CTX / FT8NN_OSD_CTX (context a-priori: calls of a message decoded 30 or
                           60 s earlier near this frequency; ft8nn_decode_at only) */
    int metric;         /* soft metric that decoded it: 0 = NN, 1..3 = classical nsym 1..3 */
    int pass;           /* subtraction pass, 0-based */
} ft8nn_decode_t;

/* onnx_path: model exported by tools/export_onnx.py. n_threads: threads a decode runs on (0 = one per core; on
 * Apple silicon one per performance core). The context keeps them, idle between decodes; the decodes do not depend
 * on the number. Returns NULL on failure (message on stderr). */
ft8nn_ctx *ft8nn_open(const char *onnx_path, int n_threads);

/* The number of threads the context decodes on. */
int ft8nn_threads(const ft8nn_ctx *ctx);

/* Decode one 15 s slot of 12 kHz mono audio (any scale; the slot starts at audio12k[0]).
 * Uses at most the first 180000 samples. Writes up to max_out decodes, returns the number of
 * decodes found (which may exceed max_out), or -1 on error. */
int ft8nn_decode(ft8nn_ctx *ctx, const float *audio12k, int n, ft8nn_decode_t *out, int max_out);

/* As ft8nn_decode, plus context a-priori decoding: slot_time is the slot start in seconds (any fixed epoch, e.g.
 * Unix time; slots 15 s apart). The context remembers its own decodes of the last 4 slots and uses those from
 * slot_time - 30 and - 60 s (the same stations' previous transmissions) to try likely next messages.
 * Call it for every slot in time order. */
int ft8nn_decode_at(ft8nn_ctx *ctx, const float *audio12k, int n, int64_t slot_time, ft8nn_decode_t *out, int max_out);

/* Passed as slot_time when the slot's start time is not known: no context a-priori decoding. */
#define FT8NN_NO_SLOT_TIME INT64_MIN

/* Called by ft8nn_decode_stream as decodes become final, so a caller can show them before the slot is done.
 * During a subtraction pass, with pass = 0, 1, ... and the decodes that pass has made final so far (the strongest
 * candidates are tried first, so the first call comes after a fraction of the pass), and once more when the pass
 * ends, with the rest of it (n may be 0); then exactly once with pass = -1 and the context a-priori decodes
 * (FT8NN_BP_CTX / FT8NN_OSD_CTX), which wait for the end because a later pass may decode the same sender's
 * real message and drop them. Every decode is reported exactly once, and the decodes together equal what
 * ft8nn_decode_at returns. Calls come from the thread that called ft8nn_decode_stream. The array is only valid
 * during the call (NULL when n is 0). If the decode fails (returns -1) the pass = -1 call is not made. */
typedef void (*ft8nn_pass_cb)(const ft8nn_decode_t *decodes, int n, int pass, void *user);

/* As ft8nn_decode_at (or ft8nn_decode with slot_time = FT8NN_NO_SLOT_TIME), reporting decodes through cb as
 * they become final; out and max_out as there (out may be NULL with max_out 0). */
int ft8nn_decode_stream(ft8nn_ctx *ctx, const float *audio12k, int n, int64_t slot_time, ft8nn_pass_cb cb,
                        void *user, ft8nn_decode_t *out, int max_out);

/* Forget all learned callsign hashes. */
void ft8nn_reset_hashes(ft8nn_ctx *ctx);

void ft8nn_close(ft8nn_ctx *ctx);

#ifdef __cplusplus
}
#endif
