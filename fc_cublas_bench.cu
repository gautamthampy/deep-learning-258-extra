
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDA(call) do { cudaError_t status = (call); if (status != cudaSuccess) { std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at line " << __LINE__ << std::endl; std::exit(EXIT_FAILURE); } } while (0)
#define CHECK_CUBLAS(call) do { cublasStatus_t status = (call); if (status != CUBLAS_STATUS_SUCCESS) { std::cerr << "cuBLAS error: " << status << " at line " << __LINE__ << std::endl; std::exit(EXIT_FAILURE); } } while (0)

struct GemmCase {
    int batch;
    int in_features;
    int out_features;
};

struct Result {
    std::string backend;
    std::string mode;
    int batch;
    int in_features;
    int out_features;
    float avg_ms;
    double tflops;
};

__global__ void fill_kernel(float* data, size_t n, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int value = static_cast<int>(idx % 251);
        data[idx] = scale * (static_cast<float>(value) - 125.0f) / 125.0f;
    }
}

int iterations_for_case(int batch, int in_features, int out_features) {
    double flops = 2.0 * batch * in_features * out_features;
    return flops >= 2.0 * 1024.0 * 1024.0 * 1024.0 ? 50 : 100;
}

void run_baseline(cublasHandle_t handle, int m, int n, int k, const float* A, const float* B, float* C) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        A, m,
        B, k,
        &beta,
        C, m
    ));
}

void run_tf32(cublasHandle_t handle, int m, int n, int k, const float* A, const float* B, float* C) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        A, CUDA_R_32F, m,
        B, CUDA_R_32F, k,
        &beta,
        C, CUDA_R_32F, m,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

Result benchmark_case(cublasHandle_t handle, const GemmCase& cfg, bool use_tf32) {
    const int m = cfg.out_features;
    const int k = cfg.in_features;
    const int n = cfg.batch;

    const size_t a_elems = static_cast<size_t>(m) * k;
    const size_t b_elems = static_cast<size_t>(k) * n;
    const size_t c_elems = static_cast<size_t>(m) * n;

    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&A, a_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&B, b_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&C, c_elems * sizeof(float)));

    const int threads = 256;
    fill_kernel<<<(a_elems + threads - 1) / threads, threads>>>(A, a_elems, 1.0f);
    fill_kernel<<<(b_elems + threads - 1) / threads, threads>>>(B, b_elems, 0.5f);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemset(C, 0, c_elems * sizeof(float)));

    if (use_tf32) {
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    } else {
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    }

    const int warmup = 25;
    const int iters = iterations_for_case(cfg.batch, cfg.in_features, cfg.out_features);

    for (int i = 0; i < warmup; ++i) {
        if (use_tf32) {
            run_tf32(handle, m, n, k, A, B, C);
        } else {
            run_baseline(handle, m, n, k, A, B, C);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        if (use_tf32) {
            run_tf32(handle, m, n, k, A, B, C);
        } else {
            run_baseline(handle, m, n, k, A, B, C);
        }
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    const float avg_ms = total_ms / static_cast<float>(iters);
    const double tflops = (2.0 * cfg.batch * cfg.in_features * cfg.out_features) / (static_cast<double>(avg_ms) * 1e-3) / 1e12;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(A));
    CHECK_CUDA(cudaFree(B));
    CHECK_CUDA(cudaFree(C));

    return Result{"cuBLAS", use_tf32 ? "tf32_tensor_core" : "baseline_fp32", cfg.batch, cfg.in_features, cfg.out_features, avg_ms, tflops};
}

int main() {
    std::vector<GemmCase> sweep = {
        {256, 1024, 1024},
        {1024, 1024, 1024},
        {4096, 1024, 1024},
        {256, 4096, 4096},
        {1024, 4096, 4096},
        {4096, 4096, 4096},
        {256, 8192, 8192},
        {1024, 8192, 8192},
        {2048, 8192, 8192}
    };

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    std::ofstream csv("cublas_fc_results.csv");
    csv << "backend,mode,batch,in_features,out_features,avg_ms,tflops\n";

    std::cout << std::fixed << std::setprecision(3);

    for (const auto& cfg : sweep) {
        Result baseline = benchmark_case(handle, cfg, false);
        Result tf32 = benchmark_case(handle, cfg, true);

        csv << baseline.backend << "," << baseline.mode << "," << baseline.batch << "," << baseline.in_features << "," << baseline.out_features << "," << baseline.avg_ms << "," << baseline.tflops << "\n";
        csv << tf32.backend << "," << tf32.mode << "," << tf32.batch << "," << tf32.in_features << "," << tf32.out_features << "," << tf32.avg_ms << "," << tf32.tflops << "\n";

        std::cout
            << "B=" << cfg.batch
            << " K=" << cfg.in_features
            << " N=" << cfg.out_features
            << " | baseline=" << baseline.avg_ms << " ms"
            << " | tf32=" << tf32.avg_ms << " ms"
            << " | speedup=" << (baseline.avg_ms / tf32.avg_ms)
            << "x"
            << std::endl;
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    return 0;
}
