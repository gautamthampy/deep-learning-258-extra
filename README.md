# FC GEMM Benchmark: CUDA Core FP32 vs Tensor Core TF32 on A100

This project benchmarks a simple bias-free fully connected layer in two modes on NVIDIA A100:

1. Baseline FP32 GEMM with TF32 disabled
2. FP32 input and output with TF32 Tensor Core compute enabled

The benchmark is implemented in both PyTorch and CUDA C++ with cuBLAS, using the same sweep sizes and timing methodology.

## Files

1. [README.md](README.md): benchmark overview, expected observations, and run instructions
2. [CMPE258-extra.ipynb](CMPE258-extra.ipynb): full Colab-ready notebook with the runnable code

## Goal

Measure the forward-pass performance difference for:

$$
Y = XW^T
$$

where:

- $X \in \mathbb{R}^{B \times K}$
- $W \in \mathbb{R}^{N \times K}$
- $Y \in \mathbb{R}^{B \times N}$

Per forward pass, the GEMM work is:

$$
	ext{FLOPs} = 2 \times B \times K \times N
$$

Effective throughput is reported as:

$$
	ext{TFLOP/s} = \frac{2 \times B \times K \times N}{t \times 10^{12}}
$$

where $t$ is average latency in seconds.

## Benchmark Modes

1. Baseline CUDA Core path
   FP32 GEMM with TF32 disabled.

2. Tensor Core path
   FP32 tensors at the API boundary, but TF32 Tensor Core compute enabled internally.

## Expected Observations

When run on an A100, the main observations should be:

1. TF32 Tensor Core mode is typically faster than strict FP32 for medium and large GEMMs.
2. The speedup generally increases with problem size because large GEMMs are more compute-bound.
3. Small GEMMs often show smaller gains because launch overhead and framework overhead matter more.
4. cuBLAS usually shows a cleaner Tensor Core advantage than PyTorch because it is closer to the raw GEMM path.
5. TF32 mode should deliver higher effective TFLOP/s while still using FP32 input and output tensors.
6. Numerical differences relative to strict FP32 are expected, but they are usually small.

## Benchmark Methodology

The benchmark is designed to make the comparison fair:

1. Same input sizes in PyTorch and cuBLAS
2. Warm-up iterations before timing
3. GPU event timing instead of host wall-clock timing
4. Explicit synchronization before reading timings
5. Average latency reported over multiple iterations
6. Throughput computed from measured average latency

## Sweep Sizes

The sweep uses these `(batch, in_features, out_features)` tuples:

- `(256, 1024, 1024)`
- `(1024, 1024, 1024)`
- `(4096, 1024, 1024)`
- `(256, 4096, 4096)`
- `(1024, 4096, 4096)`
- `(4096, 4096, 4096)`
- `(256, 8192, 8192)`
- `(1024, 8192, 8192)`
- `(2048, 8192, 8192)`

## How to Run

1. Use a Colab runtime with A100.
2. Open [CMPE258-extra.ipynb](CMPE258-extra.ipynb).
3. Run the notebook cells in order.
4. Review the generated CSV files and comparison plots.

## Outputs

The notebook produces:

1. `torch_fc_results.csv`
2. `cublas_fc_results.csv`
3. Latency plots
4. Throughput plots
5. TF32 speedup plots

## Practical Summary

On A100, the main takeaway should be:

- Strict FP32 GEMM is a valid baseline but slower.
- TF32 Tensor Core GEMM keeps FP32 tensors at the interface while reducing latency and increasing throughput.
- The advantage becomes more obvious as the FC layer size grows.
