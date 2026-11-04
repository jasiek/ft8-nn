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

enum { FT8NN_BP = 0, FT8NN_OSD = 1, FT8NN_BP_AP = 2, FT8NN_OSD_AP = 3 };

typedef struct {
    char text[48];      /* decoded message, WSJT-X formatting; hashed calls as <CALL> or <...> */
    double f0;          /* Hz, frequency of tone 0 */
    double dt;          /* s, start time relative to the nominal 0.5 s into the slot */
    double snr;         /* dB in 2500 Hz */
    uint8_t bits77[10]; /* payload, MSB first; bit i is (bits77[i/8] >> (7 - i%8)) & 1 */
    int method;         /* FT8NN_BP, FT8NN_OSD, FT8NN_BP_AP (CQ a-priori), FT8NN_OSD_AP */
    int metric;         /* soft metric that decoded it: 0 = NN, 1..3 = classical nsym 1..3 */
    int pass;           /* subtraction pass, 0-based */
} ft8nn_decode_t;

/* onnx_path: model exported by tools/export_onnx.py. n_threads: ONNX Runtime intra-op threads
 * (0 = runtime default). Returns NULL on failure (message on stderr). */
ft8nn_ctx *ft8nn_open(const char *onnx_path, int n_threads);

/* Decode one 15 s slot of 12 kHz mono audio (any scale; the slot starts at audio12k[0]).
 * Uses at most the first 180000 samples. Writes up to max_out decodes, returns the number of
 * decodes found (which may exceed max_out), or -1 on error. */
int ft8nn_decode(ft8nn_ctx *ctx, const float *audio12k, int n, ft8nn_decode_t *out, int max_out);

/* Forget all learned callsign hashes. */
void ft8nn_reset_hashes(ft8nn_ctx *ctx);

void ft8nn_close(ft8nn_ctx *ctx);

#ifdef __cplusplus
}
#endif
