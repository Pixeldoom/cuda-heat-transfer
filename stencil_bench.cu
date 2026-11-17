// stencil_bench.cu — multi-GPU benchmark 5-точечного стенсила (strong scaling)
// с размерами сетки и типом данных float/double.
//
// Сборка: nvcc -O3 -Xcompiler -fopenmp -arch=native stencil_bench.cu -o stencil_bench
// Запуск: OMP_NUM_THREADS=<num GPUs> ./stencil_bench [dims=2048] [iters=90] [reps=5] [warmup=10] [types=float,double]
//   dims  — список размеров через запятую: "2048,4096,8192"
//   types — "float", "double", "float,double" (или "f,d", "all")
//
// Варианты обмена:
//   stencil_single    — 1 GPU, обмена нет (эталон для speedup и проверки корректности)
//   stencil_p2p       — гало-обмен напрямую между GPU (cudaMemcpyPeerAsync)
//   stencil_host_halo — гало через pinned-буфер хоста (2 строки на GPU за шаг)
//   stencil_host_full — наивный вариант: весь slab через хост каждый шаг
//
// Верификация: эталон свой для каждой пары (dim, precision); внутри одной точности
// ожидается ПОБИТОВОЕ совпадение (max_diff = 0). Поля float и double не сравниваются.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define SPEED    0.25    // предел устойчивости явной схемы; точен в двоичной записи для float и double
#define MAX_TEMP 1.0
#define MIN_TEMP 0.0001

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

typedef enum { MODE_P2P = 0, MODE_HOST_HALO, MODE_HOST_FULL } ExchangeMode;
typedef enum { PREC_FLOAT = 0, PREC_DOUBLE = 1 } Precision;

static const char* precName(Precision p) { return p == PREC_FLOAT ? "float" : "double"; }
static size_t precBytes(Precision p)     { return p == PREC_FLOAT ? sizeof(float) : sizeof(double); }

static const char* modeName(ExchangeMode mode, int workers) {
    if (workers == 1) return "stencil_single";
    switch (mode) {
        case MODE_P2P:       return "stencil_p2p";
        case MODE_HOST_HALO: return "stencil_host_halo";
        default:             return "stencil_host_full";
    }
}

// Единая раскладка для ВСЕХ GPU: строка 0 — верхнее гало, строки 1..local_h — свои,
// строка local_h+1 — нижнее гало. За пределами сетки — нулевые соседи.
// Индексы size_t: безопасно для больших сеток (dim^2 может превышать int).
template<typename T>
__global__ void stencil_kernel(T* __restrict__ out, const T* __restrict__ in,
                               int width, int local_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y > local_h) return;

    size_t off = (size_t)x + (size_t)y * width;
    T v = in[off];
    T top = in[off - width];
    T bot = in[off + width];
    T left  = (x > 0)         ? in[off - 1] : (T)0;
    T right = (x < width - 1) ? in[off + 1] : (T)0;
    out[off] = v + (T)SPEED * (top + bot + left + right - (T)4 * v);
}

static bool p2pAvailable(int workers) {
    for (int s = 0; s < workers; s++)
        for (int t = 0; t < workers; t++) {
            if (s == t) continue;
            int can = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can, s, t));
            if (!can) return false;
        }
    return true;
}

static void enableP2P(int workers) {
    for (int s = 0; s < workers; s++) {
        CUDA_CHECK(cudaSetDevice(s));
        for (int t = 0; t < workers; t++) {
            if (s == t) continue;
            cudaError_t e = cudaDeviceEnablePeerAccess(t, 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                fprintf(stderr, "cudaDeviceEnablePeerAccess(%d -> %d): %s\n",
                        s, t, cudaGetErrorString(e));
        }
    }
}

template<typename T>
struct Worker {
    int dim, local_h, workers, device;
    ExchangeMode mode;
    T* buf[2];     // ping-pong буферы данного GPU
    T** all_buf;   // буферы всех GPU (для P2P)
    T* h_stage;    // pinned-буфер хоста (host-варианты)
    dim3 grid, block;
};

// Один шаг: ядро + обмен гало. Результат (включая гало) оказывается в buf[cur^1].
template<typename T>
static void workerStep(const Worker<T>* w, int cur)
{
    T* in  = w->buf[cur];
    T* out = w->buf[cur ^ 1];
    const int dim = w->dim, lh = w->local_h, d = w->device;
    const size_t row_bytes = (size_t)dim * sizeof(T);

    stencil_kernel<<<w->grid, w->block>>>(out, in, dim, lh);
    CUDA_CHECK(cudaGetLastError());
    if (w->workers == 1) return;          // одному GPU обмен и синхронизация не нужны
    CUDA_CHECK(cudaDeviceSynchronize());  // ядро завершено — строки готовы к отправке

    if (w->mode == MODE_P2P) {
        if (d > 0)
            CUDA_CHECK(cudaMemcpyPeerAsync(
                w->all_buf[(d - 1) * 2 + (cur ^ 1)] + (size_t)(lh + 1) * dim, d - 1,
                out + dim, d, row_bytes, 0));              // моя первая строка -> нижнее гало соседа сверху
        if (d < w->workers - 1)
            CUDA_CHECK(cudaMemcpyPeerAsync(
                w->all_buf[(d + 1) * 2 + (cur ^ 1)], d + 1,
                out + (size_t)lh * dim, d, row_bytes, 0)); // моя последняя строка -> верхнее гало соседа снизу
        CUDA_CHECK(cudaDeviceSynchronize());
        #pragma omp barrier   // все гало доставлены до старта следующего ядра
    }
    else if (w->mode == MODE_HOST_HALO) {
        T* slot = w->h_stage + (size_t)d * 2 * dim;   // [первая строка | последняя строка]
        CUDA_CHECK(cudaMemcpy(slot,       out + dim,            row_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(slot + dim, out + (size_t)lh * dim, row_bytes, cudaMemcpyDeviceToHost));
        #pragma omp barrier
        if (d > 0)
            CUDA_CHECK(cudaMemcpy(out, w->h_stage + ((size_t)(d - 1) * 2 + 1) * dim,
                                  row_bytes, cudaMemcpyHostToDevice));
        if (d < w->workers - 1)
            CUDA_CHECK(cudaMemcpy(out + (size_t)(lh + 1) * dim, w->h_stage + (size_t)(d + 1) * 2 * dim,
                                  row_bytes, cudaMemcpyHostToDevice));
        #pragma omp barrier   // все прочитали слоты до перезаписи на следующем шаге
    }
    else { // MODE_HOST_FULL — намеренно неэффективный: весь slab D2H + H2D каждый шаг
        CUDA_CHECK(cudaMemcpy(w->h_stage + (size_t)d * lh * dim, out + dim,
                              row_bytes * lh, cudaMemcpyDeviceToHost));
        #pragma omp barrier
        if (d == 0)
            CUDA_CHECK(cudaMemcpy(out + dim, w->h_stage,
                                  row_bytes * (lh + 1), cudaMemcpyHostToDevice));
        else if (d == w->workers - 1)
            CUDA_CHECK(cudaMemcpy(out, w->h_stage + ((size_t)d * lh - 1) * dim,
                                  row_bytes * (lh + 1), cudaMemcpyHostToDevice));
        else
            CUDA_CHECK(cudaMemcpy(out, w->h_stage + ((size_t)d * lh - 1) * dim,
                                  row_bytes * (lh + 2), cudaMemcpyHostToDevice));
        #pragma omp barrier
    }
}

typedef struct {
    int iters, warmup, reps;
    double median_ms, min_ms, mcells_s;
    double max_diff;
} BenchResult;

static int cmpDouble(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

// h_init и h_result — в double (каноническое представление на хосте);
// конверсия в T и обратно выполняется поstripно через временный буфер.
template<typename T>
static void runStencil(int dim, int iters, int warmup, int reps, int workers,
                       ExchangeMode mode, const double* h_init, double* h_result,
                       BenchResult* res)
{
    const int local_h = dim / workers;
    const size_t row_bytes = (size_t)dim * sizeof(T);
    const size_t local_bytes = row_bytes * (local_h + 2);
    const int total_steps = warmup + reps * iters;

    if (mode == MODE_P2P && workers > 1) enableP2P(workers);

    T** d_buf = (T**)calloc((size_t)workers * 2, sizeof(T*));
    T* h_tmp = (T*)malloc(local_bytes);
    if (!d_buf || !h_tmp) { fputs("OOM\n", stderr); exit(EXIT_FAILURE); }

    T* h_stage = NULL;
    if (workers > 1 && mode != MODE_P2P) {
        size_t stage_bytes = (mode == MODE_HOST_FULL) ? row_bytes * dim
                                                      : row_bytes * 2 * workers;
        CUDA_CHECK(cudaHostAlloc(&h_stage, stage_bytes, cudaHostAllocDefault));
    }

    for (int d = 0; d < workers; d++) {
        CUDA_CHECK(cudaSetDevice(d));
        for (int b = 0; b < 2; b++) {
            CUDA_CHECK(cudaMalloc(&d_buf[d * 2 + b], local_bytes));
            CUDA_CHECK(cudaMemset(d_buf[d * 2 + b], 0, local_bytes));
        }
        // Заполняем буфер 0 целиком: строка r <-> глобальная строка d*local_h - 1 + r.
        // Вне сетки — нули (граничное условие); внутренние гало — настоящие строки соседей.
        for (int r = 0; r < local_h + 2; r++) {
            long long g = (long long)d * local_h - 1 + r;
            T* dst = h_tmp + (size_t)r * dim;
            if (g >= 0 && g < dim) {
                const double* src = h_init + (size_t)g * dim;
                for (int x = 0; x < dim; x++) dst[x] = (T)src[x];
            } else {
                memset(dst, 0, row_bytes);
            }
        }
        CUDA_CHECK(cudaMemcpy(d_buf[d * 2], h_tmp, local_bytes, cudaMemcpyHostToDevice));
        // Буфер 1: свои строки перезапишет ядро, внутренние гало доставит первый обмен,
        // внешние гало останутся нулями от memset.
    }

    double* rep_ms = (double*)malloc((size_t)reps * sizeof(double));
    if (!rep_ms) { fputs("OOM\n", stderr); exit(EXIT_FAILURE); }
    double t_start = 0.0;

    #pragma omp parallel num_threads(workers)
    {
        const int d = omp_get_thread_num();
        CUDA_CHECK(cudaSetDevice(d));

        Worker<T> w;
        w.dim = dim; w.local_h = local_h; w.workers = workers; w.device = d;
        w.mode = mode;
        w.buf[0] = d_buf[d * 2]; w.buf[1] = d_buf[d * 2 + 1];
        w.all_buf = d_buf;
        w.h_stage = h_stage;
        w.block = dim3(32, 8);   // warp вдоль x — коалесцированный доступ
        w.grid  = dim3((dim + 31) / 32, (local_h + 7) / 8);

        int cur = 0;

        for (int s = 0; s < warmup; s++) { workerStep(&w, cur); cur ^= 1; }
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int r = 0; r < reps; r++) {
            #pragma omp barrier
            #pragma omp single
            { t_start = omp_get_wtime(); }
            for (int s = 0; s < iters; s++) { workerStep(&w, cur); cur ^= 1; }
            CUDA_CHECK(cudaDeviceSynchronize());
            #pragma omp barrier
            #pragma omp single
            { rep_ms[r] = (omp_get_wtime() - t_start) * 1000.0 / iters; }
        }
    }

    // Сборка итогового состояния с конверсией в double
    const int final_buf = total_steps & 1;
    for (int d = 0; d < workers; d++) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaMemcpy(h_tmp, d_buf[d * 2 + final_buf] + dim,
                              row_bytes * local_h, cudaMemcpyDeviceToHost));
        double* dst = h_result + (size_t)d * local_h * dim;
        for (size_t i = 0; i < (size_t)local_h * dim; i++) dst[i] = (double)h_tmp[i];
        CUDA_CHECK(cudaFree(d_buf[d * 2]));
        CUDA_CHECK(cudaFree(d_buf[d * 2 + 1]));
    }
    free(d_buf); free(h_tmp);
    if (h_stage) CUDA_CHECK(cudaFreeHost(h_stage));

    qsort(rep_ms, (size_t)reps, sizeof(double), cmpDouble);
    res->iters = iters; res->warmup = warmup; res->reps = reps;
    res->median_ms = rep_ms[reps / 2];
    res->min_ms = rep_ms[0];
    res->mcells_s = (double)dim * dim / (res->median_ms * 1e3);
    res->max_diff = 0.0;
    free(rep_ms);
}

static void printRow(const char* variant, const char* prec, int dim, int workers,
                     const BenchResult* r, double base_ms, FILE* csv)
{
    const double speedup = base_ms / r->median_ms;
    const double eff_pct = 100.0 * speedup / workers;
    printf("%-18s %-7s %6d %5d %10.3f %10.3f %12.1f %8.2f %8.1f%% %11.2e\n",
           variant, prec, dim, workers, r->median_ms, r->min_ms, r->mcells_s,
           speedup, eff_pct, r->max_diff);
    if (csv) {
        fprintf(csv, "%s,%s,%d,%d,%d,%d,%d,%.4f,%.4f,%.2f,%.4f,%.2f,%.3e\n",
                variant, prec, workers, dim, r->iters, r->warmup, r->reps,
                r->median_ms, r->min_ms, r->mcells_s, speedup, eff_pct, r->max_diff);
        fflush(csv);
    }
}

// ---- Разбор списков аргументов ----

static int* parseDims(const char* arg, int* count) {
    char* copy = strdup(arg);
    if (!copy) { fputs("OOM\n", stderr); exit(EXIT_FAILURE); }
    int cap = 8, n = 0;
    int* dims = (int*)malloc((size_t)cap * sizeof(int));
    for (char* tok = strtok(copy, ","); tok; tok = strtok(NULL, ",")) {
        int v = atoi(tok);
        if (v < 4) { fprintf(stderr, "Некорректный размер сетки: %s\n", tok); exit(EXIT_FAILURE); }
        if (n == cap) {
            cap *= 2;
            dims = (int*)realloc(dims, (size_t)cap * sizeof(int));
            if (!dims) { fputs("OOM\n", stderr); exit(EXIT_FAILURE); }
        }
        dims[n++] = v;
    }
    free(copy);
    if (n == 0) { fprintf(stderr, "Пустой список размеров\n"); exit(EXIT_FAILURE); }
    *count = n;
    return dims;
}

static int parseTypes(const char* arg, Precision* out) {
    char* copy = strdup(arg);
    if (!copy) { fputs("OOM\n", stderr); exit(EXIT_FAILURE); }
    int wantF = 0, wantD = 0, n = 0;
    for (char* tok = strtok(copy, ","); tok; tok = strtok(NULL, ",")) {
        if (!strcmp(tok, "float") || !strcmp(tok, "f") || !strcmp(tok, "single")) wantF = 1;
        else if (!strcmp(tok, "double") || !strcmp(tok, "d")) wantD = 1;
        else if (!strcmp(tok, "all") || !strcmp(tok, "both")) { wantF = wantD = 1; }
        else { fprintf(stderr, "Неизвестный тип: %s (float|double|all)\n", tok); exit(EXIT_FAILURE); }
    }
    free(copy);
    if (wantF) out[n++] = PREC_FLOAT;
    if (wantD) out[n++] = PREC_DOUBLE;
    if (n == 0) { fprintf(stderr, "Пустой список типов\n"); exit(EXIT_FAILURE); }
    return n;
}

int main(int argc, char** argv)
{
    const char* dims_arg  = (argc > 1) ? argv[1] : "2048";
    int iters             = (argc > 2) ? atoi(argv[2]) : 90;
    int reps              = (argc > 3) ? atoi(argv[3]) : 5;
    int warmup            = (argc > 4) ? atoi(argv[4]) : 10;
    const char* types_arg = (argc > 5) ? argv[5] : "float,double";
    if (iters < 1 || reps < 1 || warmup < 0) {
        fprintf(stderr, "Usage: %s [dims=2048,4096,...] [iters>=1] [reps>=1] [warmup>=0] "
                        "[types=float|double|float,double]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int n_dims = 0;
    int* dims = parseDims(dims_arg, &n_dims);
    Precision precs[2];
    int n_prec = parseTypes(types_arg, precs);

    omp_set_dynamic(0);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) { fprintf(stderr, "CUDA GPU не найден\n"); return EXIT_FAILURE; }

    int max_workers = deviceCount;
    if (max_workers > omp_get_max_threads()) max_workers = omp_get_max_threads();

    int max_dim = 0;
    for (int i = 0; i < n_dims; i++) if (dims[i] > max_dim) max_dim = dims[i];

    for (int d = 0; d < deviceCount; d++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, d));
        printf("GPU %d: %s (SM %d.%d, %zu MB)\n",
               d, prop.name, prop.major, prop.minor, prop.totalGlobalMem >> 20);
    }
    printf("dims=%s, iters=%d, reps=%d, warmup=%d, types=%s, max GPUs=%d\n\n",
           dims_arg, iters, reps, warmup, types_arg, max_workers);

    // Хостовые буферы — в double, размер по максимальной сетке
    const size_t max_elems = (size_t)max_dim * max_dim;
    double* h_init = (double*)calloc(max_elems, sizeof(double));
    double* h_ref  = (double*)malloc(max_elems * sizeof(double));
    double* h_out  = (double*)malloc(max_elems * sizeof(double));
    if (!h_init || !h_ref || !h_out) { fputs("OOM\n", stderr); return EXIT_FAILURE; }

    FILE* csv = fopen("benchmark_results.csv", "w");
    if (csv)
        fprintf(csv, "variant,precision,gpus,dim,iters,warmup,reps,median_ms,min_ms,"
                     "mcells_s,speedup,efficiency_pct,max_diff\n");

    printf("%-18s %-7s %6s %5s %10s %10s %12s %8s %9s %11s\n",
           "VARIANT", "PREC", "DIM", "GPUs", "med_ms", "min_ms", "MCells/s",
           "speedup", "eff", "max_diff");

    for (int di = 0; di < n_dims; di++) {
        const int dim = dims[di];
        const size_t grid_elems = (size_t)dim * dim;

        // Начальное состояние (горячий прямоугольник + 2 точки), масштабируется под dim
        memset(h_init, 0, max_elems * sizeof(double));
        const int rx0 = dim * 600 / 2048, rx1 = dim * 1200 / 2048;
        const int ry0 = dim * 620 / 2048, ry1 = dim * 1202 / 2048;
        for (int y = 0; y < dim; y++)
            for (int x = 0; x < dim; x++)
                if (x > rx0 && x < rx1 && y > ry0 && y < ry1)
                    h_init[(size_t)y * dim + x] = MAX_TEMP;
        const int px = dim * 200 / 2048;
        h_init[(size_t)(dim * 200 / 2048) * dim + px]  = (MAX_TEMP + MIN_TEMP) / 2;
        h_init[(size_t)(dim * 1400 / 2048) * dim + px] = MIN_TEMP;

        for (int pi = 0; pi < n_prec; pi++) {
            const Precision prec = precs[pi];
            printf("\n=== dim=%d, %s (%zu B/cell) ===\n", dim, precName(prec), precBytes(prec));

            double base_ms = 0.0;
            int have_ref = 0;   // эталон свой для каждой пары (dim, precision)

            for (int workers = 1; workers <= max_workers; workers++) {
                if (dim % workers != 0) {
                    printf("[skip] %d GPU: dim=%d не делится на %d\n", workers, dim, workers);
                    continue;
                }

                ExchangeMode modes[3];
                int n_modes = 0;
                if (workers == 1) {
                    modes[n_modes++] = MODE_P2P;   // обмена нет, метка "stencil_single"
                } else {
                    modes[n_modes++] = MODE_HOST_FULL;
                    modes[n_modes++] = MODE_HOST_HALO;
                    if (p2pAvailable(workers)) modes[n_modes++] = MODE_P2P;
                    else printf("[skip] P2P недоступен для %d GPU\n", workers);
                }

                for (int m = 0; m < n_modes; m++) {
                    BenchResult r;
                    if (prec == PREC_FLOAT)
                        runStencil<float>(dim, iters, warmup, reps, workers, modes[m],
                                          h_init, h_out, &r);
                    else
                        runStencil<double>(dim, iters, warmup, reps, workers, modes[m],
                                           h_init, h_out, &r);

                    if (!have_ref) {   // первый прогон группы — 1 GPU: эталон поля и T1
                        memcpy(h_ref, h_out, grid_elems * sizeof(double));
                        base_ms = r.median_ms;
                        have_ref = 1;
                        r.max_diff = 0.0;
                    } else {
                        double md = 0.0;
                        for (size_t i = 0; i < grid_elems; i++) {
                            double diff = fabs(h_out[i] - h_ref[i]);
                            if (diff > md) md = diff;
                        }
                        r.max_diff = md;
                        if (md > 0.0)   // внутри одной точности ожидается побитовое совпадение
                            fprintf(stderr, "WARNING: %s (%s, %d GPU) отличается от эталона: "
                                            "max_diff=%.3e (ожидался 0)\n",
                                    modeName(modes[m], workers), precName(prec), workers, md);
                    }
                    printRow(modeName(modes[m], workers), precName(prec), dim, workers,
                             &r, base_ms, csv);
                }
            }
        }
    }

    if (csv) { fclose(csv); printf("\nРезультаты сохранены в benchmark_results.csv\n"); }
    free(h_init); free(h_ref); free(h_out); free(dims);
    return 0;
}