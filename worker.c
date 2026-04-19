/* ============================================================
 * worker.c – Worker Client
 * ============================================================
 * A worker connects to the master server, receives tasks one
 * at a time, computes the result, and sends it back.
 *
 * Supported tasks:
 *   TASK_DOT_PRODUCT  – Vector dot product (neural network core)
 *   TASK_DCT          – 1D Discrete Cosine Transform coefficient
 *   TASK_CONVOLUTION  – 1D signal convolution (edge detection)
 * ============================================================ */

#include "common.h"
#include <unistd.h>

/* ------------------------------------------------------------------
 * compute_dot_product() – Vector Dot Product (Neural Network Core)
 *
 * The master packs two vectors A and B into the single data[] array:
 *   data[0..n-1]   = Vector A
 *   data[n..2n-1]  = Vector B
 * where n = data_size / 2.
 *
 * Returns:  R = Σ A[i] * B[i]  for i = 0 .. n-1
 * ------------------------------------------------------------------ */
static long compute_dot_product(Task *t)
{
    int n = t->data_size / 2;    /* Each vector has n elements */
    long dot = 0;

    for (int i = 0; i < n; i++) {
        dot += (long)t->data[i] * (long)t->data[n + i];
    }

    return dot;
}

/* ------------------------------------------------------------------
 * compute_dct() – 1D Discrete Cosine Transform Coefficient
 *
 * Computes the DCT-II coefficient X_k for frequency index k:
 *
 *   X_k = Σ  x[n] * cos( π/N * (n + 0.5) * k )
 *         n=0..N-1
 *
 * The signal samples are in data[0..data_size-1].
 * The frequency index k is stored in t->param.
 *
 * Returns the coefficient scaled by 1000 (to preserve precision
 * as a long integer over the socket).
 * ------------------------------------------------------------------ */
static long compute_dct(Task *t)
{
    int N = t->data_size;
    int k = t->param;           /* Frequency index */
    double X_k = 0.0;

    for (int n = 0; n < N; n++) {
        double angle = (M_PI / N) * (n + 0.5) * k;
        X_k += t->data[n] * cos(angle);
    }

    /* Scale by 1000 to retain 3 decimal digits of precision */
    return (long)(X_k * 1000.0);
}

/* ------------------------------------------------------------------
 * compute_convolution() – 1D Signal Convolution (Edge Detection)
 *
 * Applies a fixed Sobel-like derivative kernel  g = {-1, 0, 1}
 * across the input signal f[0..N-1] and returns the index of the
 * strongest edge (maximum absolute convolution response).
 *
 * Convolution formula:
 *   (f * g)[i] = Σ  f[i - j] * g[j]    for j = 0..M-1
 *
 * With g = {-1, 0, 1} (M = 3), this simplifies to:
 *   (f * g)[i] = f[i+1] - f[i-1]       (a central difference)
 *
 * Returns: (strongest_edge_index * 10000) + max_abs_value
 *   – The upper digits encode the index of the strongest edge.
 *   – The lower 4 digits encode the absolute magnitude there.
 *   This packing lets both values travel in a single long.
 * ------------------------------------------------------------------ */
static long compute_convolution(Task *t)
{
    int N = t->data_size;

    /* Kernel: g = { -1, 0, 1 } */
    int kernel[3] = { -1, 0, 1 };
    int M = 3;

    long max_abs = 0;
    int  max_idx = 0;

    /*
     * Compute convolution for valid indices only (no zero-padding).
     * Valid range: i = 1 .. N-2  (so that i-1 >= 0 and i+1 <= N-1)
     */
    for (int i = 1; i <= N - 2; i++) {
        long conv = 0;
        for (int j = 0; j < M; j++) {
            int src = i - j + (M / 2);   /* Centre the kernel */
            if (src >= 0 && src < N) {
                conv += (long)t->data[src] * kernel[j];
            }
        }

        long abs_conv = (conv < 0) ? -conv : conv;
        if (abs_conv > max_abs) {
            max_abs = abs_conv;
            max_idx = i;
        }
    }

    /* Pack index and magnitude into one long value */
    return (long)max_idx * 10000L + max_abs;
}

/* ------------------------------------------------------------------
 * compute() – Dispatch to the correct computation function.
 * ------------------------------------------------------------------ */
long compute(Task *t)
{
    switch (t->task_type) {
        case TASK_DOT_PRODUCT:  return compute_dot_product(t);
        case TASK_DCT:          return compute_dct(t);
        case TASK_CONVOLUTION:  return compute_convolution(t);
        default:
            fprintf(stderr, "[Worker] Unknown task type %d\n", t->task_type);
            return -1;
    }
}

/* ------------------------------------------------------------------
 * main() – Entry point for the worker client.
 *
 * Usage:  ./worker [master_ip]
 *         Default master_ip = 127.0.0.1 (localhost)
 *
 * 1. Connects to the master via TCP.
 * 2. Receives Task structs in a loop.
 * 3. Calls compute() and sends back the Result.
 * 4. Exits when the master closes the connection.
 * ------------------------------------------------------------------ */
int main(int argc, char *argv[])
{
    /* Use command-line IP or default to localhost */
    const char *host = (argc > 1) ? argv[1] : "127.0.0.1";

    /* ---- Create TCP socket ---- */
    int sock = socket(AF_INET, SOCK_STREAM, 0);

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port   = htons(PORT)
    };
    inet_pton(AF_INET, host, &addr.sin_addr);   /* Convert IP string */

    /* ---- Connect to master ---- */
    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("connect failed");
        return 1;
    }
    printf("[Worker] Connected to master at %s:%d\n", host, PORT);

    /* ---- Receive tasks, compute, send results ---- */
    Task   t;
    Result r;

    /* Task type names for readable output */
    const char *type_names[] = { "???", "DOT_PRODUCT", "DCT", "CONVOLUTION" };

    while (recv(sock, &t, sizeof(Task), 0) > 0) {

        r.task_id = t.task_id;
        r.result  = compute(&t);

        const char *tname = (t.task_type >= 1 && t.task_type <= 3)
                            ? type_names[t.task_type] : "UNKNOWN";

        printf("[Worker] Task %d  type=%-12s  result=%ld\n",
               t.task_id, tname, r.result);

        send(sock, &r, sizeof(Result), 0);
    }

    printf("[Worker] No more tasks. Exiting.\n");
    close(sock);
    return 0;
}