#include <bits/stdc++.h>
#include <omp.h>
#include <immintrin.h>
using namespace std;

static constexpr int BN = 240, BK = 256, BM = 128;

// 4x4 AVX2 micro-kernel (same as gemm_full_opt)
static inline void micro_kernel_4x4(
    const double* A, const double* B, double* C,
    int N, int i, int j, int k0, int kmax)
{
    __m256d c0 = _mm256_setzero_pd();
    __m256d c1 = _mm256_setzero_pd();
    __m256d c2 = _mm256_setzero_pd();
    __m256d c3 = _mm256_setzero_pd();

    for (int k = k0; k < kmax; ++k) {
        __m256d b = _mm256_loadu_pd(&B[k * N + j]);

        c0 = _mm256_fmadd_pd(_mm256_set1_pd(A[(i+0) * N + k]), b, c0);
        c1 = _mm256_fmadd_pd(_mm256_set1_pd(A[(i+1) * N + k]), b, c1);
        c2 = _mm256_fmadd_pd(_mm256_set1_pd(A[(i+2) * N + k]), b, c2);
        c3 = _mm256_fmadd_pd(_mm256_set1_pd(A[(i+3) * N + k]), b, c3);
    }

    _mm256_storeu_pd(&C[(i+0) * N + j],
        _mm256_add_pd(_mm256_loadu_pd(&C[(i+0) * N + j]), c0));
    _mm256_storeu_pd(&C[(i+1) * N + j],
        _mm256_add_pd(_mm256_loadu_pd(&C[(i+1) * N + j]), c1));
    _mm256_storeu_pd(&C[(i+2) * N + j],
        _mm256_add_pd(_mm256_loadu_pd(&C[(i+2) * N + j]), c2));
    _mm256_storeu_pd(&C[(i+3) * N + j],
        _mm256_add_pd(_mm256_loadu_pd(&C[(i+3) * N + j]), c3));
}

// Blocked + AVX2 GEMM (same as gemm_full_opt)
static void matmul_full_opt(const double* A, const double* B, double* C, int N)
{
    for (int k0 = 0; k0 < N; k0 += BK) {
        int kmax = min(k0 + BK, N);

        #pragma omp parallel for collapse(2) schedule(static)
        for (int i0 = 0; i0 < N; i0 += BM) {
            for (int j0 = 0; j0 < N; j0 += BN) {
                for (int i = i0; i + 4 <= min(i0 + BM, N); i += 4) {
                    for (int j = j0; j + 4 <= min(j0 + BN, N); j += 4) {

                        micro_kernel_4x4(A, B, C, N, i, j, k0, kmax);

                    }
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " N num_threads\n";
        return 1;
    }

    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    omp_set_num_threads(T);

    vector<double> A(N * N), B(N * N), C(N * N);

    // ---- Deterministic, single-thread RNG init (same for all binaries) ----
    {
        mt19937_64 rng(12345);
        normal_distribution<double> dist(0.0, 1.0);

        for (long i = 0; i < (long)N * (long)N; ++i) {
            A[i] = dist(rng);
            B[i] = dist(rng);
        }
    }

    // ---- NUMA first-touch: each thread touches its chunk of C ----
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)N * (long)N; ++i) {
        C[i] = 0.0;
    }

    // ---- Compute GEMM with optimized kernel ----
    double t0 = omp_get_wtime();
    matmul_full_opt(A.data(), B.data(), C.data(), N);
    double t1 = omp_get_wtime();

    double secs   = t1 - t0;
    double gflops = (2.0 * N * (double)N * (double)N) / (secs * 1e9);

    cout << "N=" << N << " T=" << T
         << " time=" << secs << "s GFLOPs=" << gflops << "\n";

    double s = 0.0;
    for (double x : C) s += x;
    cout << "checksum=" << s << "\n";

    return 0;
}
