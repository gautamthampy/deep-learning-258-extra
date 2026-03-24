# FC GEMM Benchmark: CUDA Core FP32 vs Tensor Core TF32 on A100

This README provides a complete benchmark for measuring the performance difference between a traditional FP32 GEMM path and a Tensor Core TF32 GEMM path using a simple fully connected layer.

It includes two implementations:

1. PyTorch
2. CUDA C++ with cuBLAS

Both implementations use the same sweep of matrix sizes and the same timing methodology so the comparison is fair.

## Goal

Benchmark the forward pass of a bias-free fully connected layer under two modes:

1. Baseline CUDA Core path: FP32 matmul with TF32 disabled
2. Tensor Core path: FP32 input and output with TF32 enabled

The fully connected layer computes:

$$
Y = XW^T
$$

with:

- $X \in \mathbb{R}^{B \times K}$
- $W \in \mathbb{R}^{N \times K}$
- $Y \in \mathbb{R}^{B \times N}$

The total floating-point work per forward pass is:

$$
\text{FLOPs} = 2 \times B \times K \times N
$$

Effective throughput is reported as:

$$
\text{TFLOP/s} = \frac{2 \times B \times K \times N}{t \times 10^{12}}
$$

where $t$ is the average latency in seconds.

## Expected Observations

These are the observations you should expect when running on an NVIDIA A100.

1. TF32 Tensor Core mode should be faster than the baseline FP32 path for most medium and large GEMMs.
2. The speedup usually grows with problem size because larger GEMMs are more compute-bound and make better use of Tensor Cores.
3. Small GEMMs may show limited speedup because kernel launch overhead, framework overhead, and memory effects become more visible.
4. cuBLAS results are often cleaner than PyTorch results because cuBLAS is closer to the raw GEMM kernel path, while PyTorch adds framework dispatch overhead.
5. Throughput in TF32 mode should increase substantially relative to baseline FP32, especially for shapes that map well to Tensor Core kernels.
6. Numerical output will still be FP32 tensors, but the internal multiply path changes in TF32 mode, so small numerical differences relative to full FP32 are expected.

## Benchmark Methodology

To make the comparison correct:

1. Use the same shapes in both PyTorch and cuBLAS.
2. Warm up each case before timing.
3. Use GPU timing, not host wall-clock timing.
4. Synchronize after timing.
5. Report average latency over multiple iterations.
6. Compute effective throughput from measured latency.

## Sweep Sizes

The benchmark sweeps the following shapes:

- `(256, 1024, 1024)`
- `(1024, 1024, 1024)`
- `(4096, 1024, 1024)`
- `(256, 4096, 4096)`
- `(1024, 4096, 4096)`
- `(4096, 4096, 4096)`
- `(256, 8192, 8192)`
- `(1024, 8192, 8192)`
- `(2048, 8192, 8192)`

Each tuple is:

`(batch, in_features, out_features)`

## Colab Instructions

Use an A100 runtime in Colab, then paste the following sections into separate cells in order.

## Section 1: Setup and Sweep Definition

```python
!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

assert torch.cuda.is_available(), "Switch Colab to a GPU runtime."
print("PyTorch:", torch.__version__)
print("CUDA via torch:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

device = "cuda"
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True

# (batch, in_features, out_features)
# Chosen to show how TF32 Tensor Core benefit grows with problem size.
SWEEP_CASES = [
    (256, 1024, 1024),
    (1024, 1024, 1024),
    (4096, 1024, 1024),
    (256, 4096, 4096),
    (1024, 4096, 4096),
    (4096, 4096, 4096),
    (256, 8192, 8192),
    (1024, 8192, 8192),
    (2048, 8192, 8192),
]

WARMUP_ITERS = 25

def iterations_for_case(batch, in_features, out_features):
    flops = 2.0 * batch * in_features * out_features
    return 50 if flops >= 2.0 * (1024 ** 3) else 100

def case_label(batch, in_features, out_features):
    return f"B{batch}-K{in_features}-N{out_features}"
```

## Section 2: PyTorch FC Model and Benchmark Helpers

```python
class FCModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(
            in_features,
            out_features,
            bias=False,
            device=device,
            dtype=torch.float32,
        )
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.02)

    def forward(self, x):
        return self.fc(x)

def configure_torch_mode(use_tf32):
    # Baseline: full FP32 path
    # Tensor Core mode: FP32 input/output with TF32 compute allowed
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    torch.set_float32_matmul_precision("high" if use_tf32 else "highest")

@torch.inference_mode()
def benchmark_torch_case(batch, in_features, out_features, use_tf32):
    configure_torch_mode(use_tf32)

    model = FCModel(in_features, out_features).eval()
    x = torch.randn(batch, in_features, device=device, dtype=torch.float32)

    for _ in range(WARMUP_ITERS):
        _ = model(x)
    torch.cuda.synchronize()

    iters = iterations_for_case(batch, in_features, out_features)

    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        _ = model(x)
    stop_event.record()
    torch.cuda.synchronize()

    total_ms = start_event.elapsed_time(stop_event)
    avg_ms = total_ms / iters
    tflops = (2.0 * batch * in_features * out_features) / (avg_ms * 1e-3) / 1e12

    return {
        "backend": "PyTorch",
        "mode": "tf32_tensor_core" if use_tf32 else "baseline_fp32",
        "batch": batch,
        "in_features": in_features,
        "out_features": out_features,
        "avg_ms": avg_ms,
        "tflops": tflops,
    }
```

## Section 3: Run the PyTorch Sweep

```python
torch_rows = []

for batch, in_features, out_features in SWEEP_CASES:
    for use_tf32 in [False, True]:
        row = benchmark_torch_case(batch, in_features, out_features, use_tf32)
        torch_rows.append(row)
        print(
            f"[PyTorch] {row['mode']:>16} | "
            f"{case_label(batch, in_features, out_features):>18} | "
            f"{row['avg_ms']:8.3f} ms | {row['tflops']:8.2f} TFLOP/s"
        )

torch_results = pd.DataFrame(torch_rows)
torch_results.to_csv("torch_fc_results.csv", index=False)
torch_results
```

## Section 4: Generate the CUDA C++ cuBLAS Benchmark Source

```python
cases_cpp = ",\n        ".join(f"{{{b}, {k}, {n}}}" for b, k, n in SWEEP_CASES)

cuda_source = f"""
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDA(call) do {{ cudaError_t status = (call); if (status != cudaSuccess) {{ std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at line " << __LINE__ << std::endl; std::exit(EXIT_FAILURE); }} }} while (0)
#define CHECK_CUBLAS(call) do {{ cublasStatus_t status = (call); if (status != CUBLAS_STATUS_SUCCESS) {{ std::cerr << "cuBLAS error: " << status << " at line " << __LINE__ << std::endl; std::exit(EXIT_FAILURE); }} }} while (0)

struct GemmCase {{
    int batch;
    int in_features;
    int out_features;
}};

struct Result {{
    std::string backend;
    std::string mode;
    int batch;
    int in_features;
    int out_features;
    float avg_ms;
    double tflops;
}};

__global__ void fill_kernel(float* data, size_t n, float scale) {{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        int value = static_cast<int>(idx % 251);
        data[idx] = scale * (static_cast<float>(value) - 125.0f) / 125.0f;
    }}
}}

int iterations_for_case(int batch, int in_features, int out_features) {{
    double flops = 2.0 * batch * in_features * out_features;
    return flops >= 2.0 * 1024.0 * 1024.0 * 1024.0 ? 50 : 100;
}}

void run_baseline(cublasHandle_t handle, int m, int n, int k, const float* A, const float* B, float* C) {{
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
}}

void run_tf32(cublasHandle_t handle, int m, int n, int k, const float* A, const float* B, float* C) {{
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
}}

Result benchmark_case(cublasHandle_t handle, const GemmCase& cfg, bool use_tf32) {{
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

    if (use_tf32) {{
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    }} else {{
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    }}

    const int warmup = 25;
    const int iters = iterations_for_case(cfg.batch, cfg.in_features, cfg.out_features);

    for (int i = 0; i < warmup; ++i) {{
        if (use_tf32) {{
            run_tf32(handle, m, n, k, A, B, C);
        }} else {{
            run_baseline(handle, m, n, k, A, B, C);
        }}
    }}
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {{
        if (use_tf32) {{
            run_tf32(handle, m, n, k, A, B, C);
        }} else {{
            run_baseline(handle, m, n, k, A, B, C);
        }}
    }}
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

    return Result{{"cuBLAS", use_tf32 ? "tf32_tensor_core" : "baseline_fp32", cfg.batch, cfg.in_features, cfg.out_features, avg_ms, tflops}};
}}

int main() {{
    std::vector<GemmCase> sweep = {{
        {cases_cpp}
    }};

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    std::ofstream csv("cublas_fc_results.csv");
    csv << "backend,mode,batch,in_features,out_features,avg_ms,tflops\\n";

    std::cout << std::fixed << std::setprecision(3);

    for (const auto& cfg : sweep) {{
        Result baseline = benchmark_case(handle, cfg, false);
        Result tf32 = benchmark_case(handle, cfg, true);

        csv << baseline.backend << "," << baseline.mode << "," << baseline.batch << "," << baseline.in_features << "," << baseline.out_features << "," << baseline.avg_ms << "," << baseline.tflops << "\\n";
        csv << tf32.backend << "," << tf32.mode << "," << tf32.batch << "," << tf32.in_features << "," << tf32.out_features << "," << tf32.avg_ms << "," << tf32.tflops << "\\n";

        std::cout
            << "B=" << cfg.batch
            << " K=" << cfg.in_features
            << " N=" << cfg.out_features
            << " | baseline=" << baseline.avg_ms << " ms"
            << " | tf32=" << tf32.avg_ms << " ms"
            << " | speedup=" << (baseline.avg_ms / tf32.avg_ms)
            << "x"
            << std::endl;
    }}

    CHECK_CUBLAS(cublasDestroy(handle));
    return 0;
}}
"""

Path("fc_cublas_bench.cu").write_text(cuda_source)
print("Wrote fc_cublas_bench.cu")
```

## Section 5: Compile and Run the CUDA C++ Benchmark on A100

```python
!nvcc --version
!nvcc -O3 -std=c++17 -arch=sm_80 fc_cublas_bench.cu -lcublas -o fc_cublas_bench
!./fc_cublas_bench
```

## Section 6: Load Both Result Sets, Compute Speedups, and Plot

```python
torch_df = pd.read_csv("torch_fc_results.csv")
cublas_df = pd.read_csv("cublas_fc_results.csv")

results = pd.concat([torch_df, cublas_df], ignore_index=True)
results["case"] = results.apply(
    lambda row: case_label(int(row["batch"]), int(row["in_features"]), int(row["out_features"])),
    axis=1,
)

order = [case_label(b, k, n) for b, k, n in SWEEP_CASES]

summary_rows = []
for backend in ["PyTorch", "cuBLAS"]:
    sub = results[results["backend"] == backend].copy()
    latency = sub.pivot(index="case", columns="mode", values="avg_ms").reindex(order)
    throughput = sub.pivot(index="case", columns="mode", values="tflops").reindex(order)
    speedup = latency["baseline_fp32"] / latency["tf32_tensor_core"]

    for case in order:
        summary_rows.append({
            "backend": backend,
            "case": case,
            "baseline_ms": latency.loc[case, "baseline_fp32"],
            "tf32_ms": latency.loc[case, "tf32_tensor_core"],
            "speedup_x": speedup.loc[case],
            "baseline_tflops": throughput.loc[case, "baseline_fp32"],
            "tf32_tflops": throughput.loc[case, "tf32_tensor_core"],
        })

summary_df = pd.DataFrame(summary_rows)
display(summary_df)

print("\nSpeedup summary by backend")
display(summary_df.groupby("backend")["speedup_x"].agg(["mean", "median", "max", "min"]))

sns.set_theme(style="whitegrid", context="talk")
palette = {
    "baseline_fp32": "#4c78a8",
    "tf32_tensor_core": "#f58518",
}

fig, axes = plt.subplots(2, 2, figsize=(24, 14), constrained_layout=True)

for row_index, backend in enumerate(["PyTorch", "cuBLAS"]):
    sub = results[results["backend"] == backend].copy()
    sub["case"] = pd.Categorical(sub["case"], categories=order, ordered=True)

    sns.barplot(
        data=sub,
        x="case",
        y="avg_ms",
        hue="mode",
        palette=palette,
        ax=axes[row_index, 0],
    )
    axes[row_index, 0].set_title(f"{backend}: average forward latency")
    axes[row_index, 0].set_xlabel("Shape")
    axes[row_index, 0].set_ylabel("Latency (ms)")
    axes[row_index, 0].tick_params(axis="x", rotation=45)

    sns.barplot(
        data=sub,
        x="case",
        y="tflops",
        hue="mode",
        palette=palette,
        ax=axes[row_index, 1],
    )
    axes[row_index, 1].set_title(f"{backend}: effective throughput")
    axes[row_index, 1].set_xlabel("Shape")
    axes[row_index, 1].set_ylabel("TFLOP/s")
    axes[row_index, 1].tick_params(axis="x", rotation=45)

plt.show()

fig, axes = plt.subplots(1, 2, figsize=(24, 6), constrained_layout=True)

for ax, backend in zip(axes, ["PyTorch", "cuBLAS"]):
    sub = results[results["backend"] == backend].copy()
    latency = sub.pivot(index="case", columns="mode", values="avg_ms").reindex(order)
    speedup = latency["baseline_fp32"] / latency["tf32_tensor_core"]

    sns.lineplot(
        x=speedup.index,
        y=speedup.values,
        marker="o",
        linewidth=3,
        color="#54a24b",
        ax=ax,
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{backend}: TF32 Tensor Core speedup over FP32 baseline")
    ax.set_xlabel("Shape")
    ax.set_ylabel("Speedup (x)")
    ax.tick_params(axis="x", rotation=45)

plt.show()
```

## Section 7: Optional Numerical-Difference Check for PyTorch

```python
batch, in_features, out_features = SWEEP_CASES[-2]

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

x = torch.randn(batch, in_features, device=device, dtype=torch.float32)
model = FCModel(in_features, out_features).eval()

configure_torch_mode(False)
with torch.inference_mode():
    y_fp32 = model(x)
torch.cuda.synchronize()

configure_torch_mode(True)
with torch.inference_mode():
    y_tf32 = model(x)
torch.cuda.synchronize()

abs_diff = (y_fp32 - y_tf32).abs()
rel_diff = abs_diff / y_fp32.abs().clamp_min(1e-6)

print("Case:", case_label(batch, in_features, out_features))
print("Max abs diff :", abs_diff.max().item())
print("Mean abs diff:", abs_diff.mean().item())
print("Max rel diff :", rel_diff.max().item())
print("Mean rel diff:", rel_diff.mean().item())
```

## How to Interpret the Results

Use the following interpretation when you review the plots and printed summaries.

1. Baseline mode is the traditional FP32 GEMM path.
   In PyTorch, TF32 is disabled. In cuBLAS, the benchmark uses `cublasSgemm` with pedantic math so the baseline does not silently benefit from TF32 Tensor Core execution.

2. Tensor Core mode still uses FP32 tensors at the interface.
   Inputs and outputs remain FP32, but the internal GEMM compute path is allowed to use TF32 Tensor Core math.

3. Larger GEMMs should show stronger Tensor Core gains.
   As dimensions grow, the workload becomes more compute-dominated, which is where A100 Tensor Cores help most.

4. Throughput is usually the clearest signal.
   If TF32 mode delivers higher TFLOP/s and lower latency than the baseline, then Tensor Core utilization is providing real benefit.

5. Numerical differences should be present but typically small.
   That is expected because TF32 reduces mantissa precision in the multiply path while keeping the external tensors in FP32.

## Deliverables Produced by the Benchmark

Running the full workflow generates:

1. `torch_fc_results.csv`
2. `cublas_fc_results.csv`
3. Latency comparison plots
4. Throughput comparison plots
5. TF32 speedup plots

## Practical Summary

On A100, the main observation should be:

- Traditional FP32 GEMM is correct as a baseline but slower.
- TF32 Tensor Core GEMM preserves FP32 tensor interfaces while delivering lower latency and higher throughput.
- The benefit becomes more obvious as the fully connected layer size increases.

If you want to extend this, the next step is to compare `cublasGemmEx` against cuBLASLt or add backward-pass benchmarking.