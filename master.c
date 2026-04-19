/* ============================================================
 * master.c – Master Task Dispatcher Server
 * ============================================================
 * The master creates a pool of tasks, listens on a TCP socket,
 * and dispatches one task at a time to each connected worker.
 * Each worker connection is handled in its own POSIX thread.
 *
 * Task types dispatched:
 *   1. TASK_DOT_PRODUCT  – Two vectors packed in data[]
 *   2. TASK_DCT          – Signal data + frequency index k
 *   3. TASK_CONVOLUTION  – Signal data for edge detection
 * ============================================================ */

#include "common.h"

/* ---- Global shared state (protected by mutex) ---- */
Task    tasks[16];                                  /* Task pool       */
int     task_count  = 0;                            /* Total tasks     */
int     next_task   = 0;                            /* Next to assign  */
Result  results[16];                                /* Collected results */
int     result_count = 0;                           /* Results so far  */
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;  /* Thread safety   */

/* ------------------------------------------------------------------
 * create_tasks() – Populate the task pool with 6 sample tasks.
 *
 * Creates 6 tasks: 2 of each type, with realistic sample data.
 *   Task 1 → DOT_PRODUCT   Task 2 → DCT           Task 3 → CONVOLUTION
 *   Task 4 → DOT_PRODUCT   Task 5 → DCT           Task 6 → CONVOLUTION
 * ------------------------------------------------------------------ */
void create_tasks(void)
{
    int id = 0;

    /* ============================================================
     * TASK 1 – DOT PRODUCT
     * Vector A = { 1, 2, 3, 4, 5, 6, 7, 8 }
     * Vector B = { 8, 7, 6, 5, 4, 3, 2, 1 }
     * Expected: 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1
     *         = 8 + 14 + 18 + 20 + 20 + 18 + 14 + 8 = 120
     * ============================================================ */
    {
        int vecA[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        int vecB[] = { 8, 7, 6, 5, 4, 3, 2, 1 };
        int n = 8;

        tasks[id].task_id   = id + 1;
        tasks[id].task_type = TASK_DOT_PRODUCT;
        tasks[id].data_size = 2 * n;    /* Both vectors packed */
        tasks[id].param     = 0;        /* Not used for dot product */

        /* Pack: data[0..n-1] = A,  data[n..2n-1] = B */
        memcpy(&tasks[id].data[0], vecA, n * sizeof(int));
        memcpy(&tasks[id].data[n], vecB, n * sizeof(int));
        id++;
    }

    /* ============================================================
     * TASK 2 – DCT COEFFICIENT (k = 0, the "DC" component)
     * Signal: { 100, 120, 130, 110, 90, 80, 100, 115 }
     * Computing X_0 = Σ x[n]*cos(0) = Σ x[n] = 845
     * (Result will be 845000 because worker scales ×1000)
     * ============================================================ */
    {
        int signal[] = { 100, 120, 130, 110, 90, 80, 100, 115 };
        int n = 8;

        tasks[id].task_id   = id + 1;
        tasks[id].task_type = TASK_DCT;
        tasks[id].data_size = n;
        tasks[id].param     = 0;        /* Frequency index k = 0 */

        memcpy(tasks[id].data, signal, n * sizeof(int));
        id++;
    }

    /* ============================================================
     * TASK 3 – CONVOLUTION (Edge Detection)
     * Signal: { 10, 10, 10, 10, 50, 50, 50, 50 }
     *    (Step function — sharp edge between indices 3 and 4)
     * Kernel: { -1, 0, 1 }
     * Conv[4] = signal[5] - signal[3] = 50 - 10 = 40 (biggest)
     * Expected: strongest edge at index 4, magnitude 40
     *   → encoded as 4*10000 + 40 = 40040
     * ============================================================ */
    {
        int signal[] = { 10, 10, 10, 10, 50, 50, 50, 50 };
        int n = 8;

        tasks[id].task_id   = id + 1;
        tasks[id].task_type = TASK_CONVOLUTION;
        tasks[id].data_size = n;
        tasks[id].param     = 0;

        memcpy(tasks[id].data, signal, n * sizeof(int));
        id++;
    }

    /* ============================================================
     * TASK 4 – DOT PRODUCT (larger vectors, 16 elements each)
     * A = { 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 }
     * B = { 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5 }
     * Expected: 3*2+1*7+4*1+1*8+5*2+9*8+2*1+6*8+5*2+3*8+5*4+
     *           8*5+9*9+7*0+9*4+3*5
     *         = 6+7+4+8+10+72+2+48+10+24+20+40+81+0+36+15 = 383
     * ============================================================ */
    {
        int vecA[] = { 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
        int vecB[] = { 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5 };
        int n = 16;

        tasks[id].task_id   = id + 1;
        tasks[id].task_type = TASK_DOT_PRODUCT;
        tasks[id].data_size = 2 * n;
        tasks[id].param     = 0;

        memcpy(&tasks[id].data[0], vecA, n * sizeof(int));
        memcpy(&tasks[id].data[n], vecB, n * sizeof(int));
        id++;
    }

    /* ============================================================
     * TASK 5 – DCT COEFFICIENT (k = 2, a mid-frequency component)
     * Signal: { 255, 0, 128, 64, 200, 32, 180, 90 }
     * The worker computes:
     *   X_2 = Σ x[n]*cos(π/8 * (n+0.5) * 2)  for n=0..7
     * Result scaled ×1000 is returned.
     * ============================================================ */
    {
        int signal[] = { 255, 0, 128, 64, 200, 32, 180, 90 };
        int n = 8;

        tasks[id].task_id   = id + 1;
        tasks[id].task_type = TASK_DCT;
        tasks[id].data_size = n;
        tasks[id].param     = 2;        /* Frequency index k = 2 */

        memcpy(tasks[id].data, signal, n * sizeof(int));
        id++;
    }

    /* ============================================================
     * TASK 6 – CONVOLUTION (Edge Detection on a ramp + spike)
     * Signal: { 5, 10, 15, 20, 100, 25, 30, 35, 40, 45 }
     *   (Huge spike at index 4 → strong edges at indices 3–5)
     * Kernel: { -1, 0, 1 }
     * Conv[4] = signal[5] - signal[3] = 25 - 20 = 5
     * Conv[3] = signal[4] - signal[2] = 100 - 15 = 85  ← biggest
     * Expected: strongest edge at index 3, magnitude 85
     *   → encoded as 3*10000 + 85 = 30085
     * ============================================================ */
    {
        int signal[] = { 5, 10, 15, 20, 100, 25, 30, 35, 40, 45 };
        int n = 10;

        tasks[id].task_id   = id + 1;
        tasks[id].task_type = TASK_CONVOLUTION;
        tasks[id].data_size = n;
        tasks[id].param     = 0;

        memcpy(tasks[id].data, signal, n * sizeof(int));
        id++;
    }

    task_count = id;    /* = 6 */
}

/* ------------------------------------------------------------------
 * handle_worker() – Thread function: serves one connected worker.
 *
 * 1. Grabs the next available task (thread-safe via mutex).
 * 2. Sends the Task struct to the worker over the socket.
 * 3. Receives the Result struct back.
 * 4. Repeats until no tasks remain.
 * ------------------------------------------------------------------ */
void *handle_worker(void *arg)
{
    int sock = *(int *)arg;
    free(arg);                     /* Free the heap-allocated fd */

    Task   t;
    Result r;

    /* Task type names for readable output */
    const char *type_names[] = { "???", "DOT_PRODUCT", "DCT", "CONVOLUTION" };

    while (1) {
        /* ---- Grab the next task (critical section) ---- */
        pthread_mutex_lock(&mutex);

        if (next_task >= task_count) {
            pthread_mutex_unlock(&mutex);
            break;                 /* No more tasks */
        }
        t = tasks[next_task++];

        pthread_mutex_unlock(&mutex);

        const char *tname = (t.task_type >= 1 && t.task_type <= 3)
                            ? type_names[t.task_type] : "UNKNOWN";
        printf("  [Master] Dispatching Task %d (%-12s) ...\n",
               t.task_id, tname);

        /* ---- Send task to worker ---- */
        if (send(sock, &t, sizeof(Task), 0) < 0) {
            break;
        }

        /* ---- Receive result from worker ---- */
        if (recv(sock, &r, sizeof(Result), 0) <= 0) {
            break;
        }

        /* ---- Store result (critical section) ---- */
        pthread_mutex_lock(&mutex);
        results[result_count++] = r;
        pthread_mutex_unlock(&mutex);

        /* ---- Pretty-print the result ---- */
        if (t.task_type == TASK_DOT_PRODUCT) {
            printf("  [Master] Task %d (%s) → dot product = %ld\n",
                   r.task_id, tname, r.result);
        } else if (t.task_type == TASK_DCT) {
            printf("  [Master] Task %d (%s, k=%d) → X_k×1000 = %ld  "
                   "(actual ≈ %.3f)\n",
                   r.task_id, tname, t.param, r.result, r.result / 1000.0);
        } else if (t.task_type == TASK_CONVOLUTION) {
            long edge_idx = r.result / 10000L;
            long edge_mag = r.result % 10000L;
            printf("  [Master] Task %d (%s) → strongest edge at "
                   "index %ld, magnitude %ld\n",
                   r.task_id, tname, edge_idx, edge_mag);
        }
    }

    close(sock);
    return NULL;
}

/* ------------------------------------------------------------------
 * main() – Entry point for the master server.
 *
 * 1. Creates the task pool.
 * 2. Opens a TCP listening socket on PORT (8080).
 * 3. Accepts worker connections (up to MAX_WORKERS).
 * 4. Spawns a thread per worker.
 * 5. Waits for all threads to finish, then prints summary.
 * ------------------------------------------------------------------ */
int main(void)
{
    /* ---- Step 1: Create tasks ---- */
    create_tasks();

    printf("========================================\n");
    printf(" Task Dispatcher – Signal Processing\n");
    printf("========================================\n");
    printf(" Tasks loaded: %d\n", task_count);
    printf("   • DOT_PRODUCT  (Neural Network Core)\n");
    printf("   • DCT          (Image Frequency Analysis)\n");
    printf("   • CONVOLUTION  (Edge Detection)\n");
    printf("========================================\n\n");

    /* ---- Step 2: Create TCP socket ---- */
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {
        .sin_family      = AF_INET,
        .sin_addr.s_addr = INADDR_ANY,   /* Listen on all interfaces */
        .sin_port        = htons(PORT)
    };

    bind(server_fd, (struct sockaddr *)&addr, sizeof(addr));
    listen(server_fd, MAX_WORKERS);

    printf("[Master] Listening on port %d ...\n", PORT);

    /* ---- Step 3: Accept worker connections ---- */
    pthread_t tids[MAX_WORKERS];
    int worker_count = 0;

    while (next_task < task_count) {
        int *csock = malloc(sizeof(int));
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        *csock = accept(server_fd,
                        (struct sockaddr *)&client_addr,
                        &client_len);

        printf("[Master] Worker %d connected\n", worker_count + 1);

        pthread_create(&tids[worker_count++], NULL,
                       handle_worker, csock);
    }

    /* ---- Step 4: Wait for all worker threads ---- */
    for (int i = 0; i < worker_count; i++) {
        pthread_join(tids[i], NULL);
    }

    /* ---- Step 5: Print final summary ---- */
    printf("\n========================================\n");
    printf(" All %d tasks completed.\n", result_count);
    printf("========================================\n");

    for (int i = 0; i < result_count; i++) {
        printf("  Task %d → result = %ld\n",
               results[i].task_id, results[i].result);
    }
    printf("========================================\n");

    close(server_fd);
    return 0;
}