//
// Created by 祈Inory on 2023/1/6.
// Copyright (c) 2023 BBK. All rights reserved.
//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <fstream>
#include <cmath>
#include <vector>

using namespace std;
using namespace chrono;

static float ELAPSED;
static float CPU_FLOPS;
static float CALC_OPS;
static float CALC_FLOPS;
static vector<string> FUNC_NAME_VEC;
static vector<float> ELAPSED_VEC, FLOPS_VEC, FLOPS_PERCENT_VEC;

static void calc_flops() {
    CALC_FLOPS = (CALC_OPS / ELAPSED);
}

class Timer {
public:
    Timer() = delete;

    Timer(char* func_name):
        func_name(func_name) {
        start = system_clock::now();
    }

    ~Timer() {
        auto end = system_clock::now();
        auto elapsed = duration_cast<duration<float>>(end - start);
        printf("%s cost %f seconds\n", func_name, elapsed.count());
        ELAPSED = elapsed.count();
        calc_flops();

        FUNC_NAME_VEC.emplace_back(func_name);
        ELAPSED_VEC.push_back(ELAPSED);
        FLOPS_VEC.push_back(CALC_FLOPS);
        FLOPS_PERCENT_VEC.push_back(CALC_FLOPS * 100 / CPU_FLOPS);
    }

private:
    char* func_name;
    time_point<system_clock> start;
};

static float get_clock_speed() {
//    ifstream ifs("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
//    unsigned int clock_speed;
//    ifs >> clock_speed;
//    return clock_speed / 1e6f;
    return 2.8;
}

/**
 * @brief 产生随机矩阵
 * @param mat
 * @param row
 * @param col
 * @return
 */
static void generate_matrix(float** mat, int row, int col) {
    *mat = (float*)malloc(row * col * sizeof(float));
    srand(time(NULL));
    for (int i = 0; i < row * col; ++i) {
        (*mat)[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
}

/**
 * @brief 产生全0矩阵
 * @param mat
 * @param row
 * @param col
 */
static void zero_matrix(float** mat, int row, int col) {
    if (*mat == NULL) {
        *mat = (float*)malloc(row * col * sizeof(float));
    }
    memset(*mat, 0, row * col * sizeof(float));
}

/**
 * @brief 销毁矩阵
 * @param mat
 */
static void destroy_matrix(float* mat) {
    free(mat);
}

/**
 * @brief C[i][j] = sum(A[i][k] * B[k][j])
 * @param A
 * @param B
 * @param C
 * @param row
 * @param col
 */
static void mm_ijk(float* A, float* B, float* C, int row, int col) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            for (int k = 0; k < row; ++k) {
                C[i * col + j] += A[i * col + k] * B[k * col + j];
            }
        }
    }
}

static void mm_ikj(float* A, float* B, float* C, int row, int col) {
    for (int i = 0; i < row; ++i) {
        for (int k = 0; k < row; ++k) {
            for (int j = 0; j < col; ++j) {
                C[i * col + j] += A[i * col + k] * B[k * col + j];
            }
        }
    }
}

static void mm_jik(float* A, float* B, float* C, int row, int col) {
    for (int j = 0; j < col; ++j) {
        for (int i = 0; i < row; ++i) {
            for (int k = 0; k < row; ++k) {
                C[i * col + j] += A[i * col + k] * B[k * col + j];
            }
        }
    }
}

static void mm_jki(float* A, float* B, float* C, int row, int col) {
    for (int j = 0; j < col; ++j) {
        for (int k = 0; k < row; ++k) {
            for (int i = 0; i < row; ++i) {
                C[i * col + j] += A[i * col + k] * B[k * col + j];
            }
        }
    }
}

static void mm_kij(float* A, float* B, float* C, int row, int col) {
    for (int k = 0; k < row; ++k) {
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                C[i * col + j] += A[i * col + k] * B[k * col + j];
            }
        }
    }
}

static void mm_kji(float* A, float* B, float* C, int row, int col) {
    for (int k = 0; k < row; ++k) {
        for (int j = 0; j < col; ++j) {
            for (int i = 0; i < row; ++i) {
                C[i * col + j] += A[i * col + k] * B[k * col + j];
            }
        }
    }
}

static void mm_ikj_tiled(float* A, float* B, float* C, int n, int s) {
    for (int ih = 0; ih < n; ih+=s) {
        for (int jh = 0; jh < n; jh+=s) {
            for (int kh = 0; kh < n; kh+=s) {
                for (int il = 0; il < s; il++) {
                    for (int kl = 0; kl < s; kl++) {
                        for (int jl = 0; jl < s; jl++) {
                            C[(ih + il) * n + jh + jl] = A[(ih + il) * n + kh + kl] * B[(kh + kl) * n + jh + jl];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    printf("Usage: mm [matrix size, default=1024]\n");

    int row, col;
    row = col = 1024;
    if (argc >= 2) {
        row = col = atoi(argv[1]);
    }

    int core_count = std::thread::hardware_concurrency();
    float clock_speed = get_clock_speed();
    // FLOPS: 频率 * CPU核数 * 2(超线程) * 16个FPU
    CPU_FLOPS = clock_speed * core_count * 2 * 16;

    printf("CPU Info: \n");
    printf("core count: %d\n", core_count);
    printf("clock speed: %f GHz\n", clock_speed);
    printf("cpu flops: %f GFLOPS\n", CPU_FLOPS);
    printf("\n");

    printf("row = col = %d\n", row);
    CALC_OPS = 2 * pow(row, 3) / 1e9;
    printf("calc ops: %f GFLOPS\n", CALC_OPS);

    float* A, *B, *C = NULL;
    generate_matrix(&A, row, col);
    generate_matrix(&B, row, col);
    zero_matrix(&C, row, col);

    {
        Timer timer("ijk");
        mm_ijk(A, B, C, row, col);
    }

    {
        Timer timer("ikj");
        mm_ikj(A, B, C, row, col);
    }

//    {
//        Timer timer("jik");
//        mm_jik(A, B, C, row, col);
//    }
//
//    {
//        Timer timer("jki");
//        mm_jki(A, B, C, row, col);
//    }
//
//    {
//        Timer timer("kij");
//        mm_kij(A, B, C, row, col);
//    }
//
//    {
//        Timer timer("kji");
//        mm_kji(A, B, C, row, col);
//    }

    {
        Timer timer("ikj_tiled");
        mm_ikj_tiled(A, B, C, row, 64);
    }

    destroy_matrix(A);
    destroy_matrix(B);
    destroy_matrix(C);

    printf("Loop Order |\t Latency(s) |\t GFLOPS |\t Percent of Peak(%)\n");
    for (size_t i = 0; i < FUNC_NAME_VEC.size(); ++i) {
        printf("%s \t %f \t %f \t %f\n", FUNC_NAME_VEC[i].c_str(), ELAPSED_VEC[i], FLOPS_VEC[i], FLOPS_PERCENT_VEC[i]);
    }

    return 0;
}