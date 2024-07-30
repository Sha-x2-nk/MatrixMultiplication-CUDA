# CUDA Matrix Multiplication

This repository contains CUDA implementations of matrix multiplication on GPU, starting from basic, inefficient methods and progressively improving towards more optimized solutions all the way to WARP TILING ultimately aiming to outperform cuBLAS for large matrices of size 4096x4096.

## Introduction

Matrix multiplication is a fundamental operation in many scientific and engineering applications. CUDA (Compute Unified Device Architecture) offers a powerful platform for parallel computing on NVIDIA GPUs, enabling significant speedups for matrix multiplication compared to traditional CPU-based approaches.

This project explores various CUDA implementations of matrix multiplication, focusing on optimizing performance and efficiency.

## Implementation

Below are all implementations and their stats. Benchmarking was done using Nvidia Nsight Compute. 


NUM_FLOPS for MatMul= 2 x (4096 x 4096 x 4096)= 137.44 GFLOPS ( FOR OUR KERNELS )


NUM_FLOPS for cuBlas MatMul= 2 x (4096 x 4096 x 4096) + 4096 x 4096= 137.45 GFLOPS ( since CuBLAS uses alpha x C' + beta x C, where C'= A x B )


#### HARDWARE: RTX 3070Ti ( Compute Capablity 8.6 )

|Kernel | GFLOPs | % of CuBLAS |
|-------|--------|-------------|
|0. BASIC MATMUL. No COALESCING |  50.53 GFLOPs | 0.73 % |
|1. RESULTS IN REGISTER |  156.83 GFLOPs | 2.29 % |
|2. GMEM (GLOBAL MEMORY) COALESCING |  670.92 GFLOPs | 9.82 % |
|3. SHMEM (SHARED MEMORY) USED |  879.89 GFLOPs | 12.87 % |
|4. VECTORISED MEM ACCESSES (128-bit) |  917.42 GFLOPs | 13.42 % |
|5. THREAD TILE (1D- EACH THREAD COMPUTES TM ELEMENTS) |  2709.22 GFLOPs | 39.65 % |
|6. THREAD TILE (2D- EACH THREAD COMPUTES TMxTN ELEMENTS) |  4177.47 GFLOPs | 61.14 % |
|7. THREAD TILE 2D WITH s_A TRANSPOSED |  5327.09 GFLOPs | 77.97 % |
|8. WARP TILE (EACH WARP COMPUTES WMxWN ELEMENTS) |  6001.7 GFLOPs | 87.84 % |
|9. WARP TILE CUSTOMISED (ALL CONSTANTS AND LOOP UNROLL) |  7012.19 GFLOPs | 102.64 % |
| -  CuBLAS |  6831.79 GFLOPs | 100 % |


Although we are faster than CuBLAS but turns out, CuBLAS is much much better because we have fixed our kernel to work only for 4096x4096 matrix with the given hyperparameters, which turn out to be best on my GPU. CuBLAS will probably have 100s of implementations of matrix multiplication choosing what to use at runtime.

## Usage
* Compile using nvcc, link with cublas to bench cublas performance, or comment it out

    <code>nvcc main.cu -o main.exe -lcublas</code>

* Run

    <code>main.exe</code>

* Tune parameters like BLOCK_SIZE and all tile sizes for your hardware.

## Acknowledgments

This project is inspired by Si_Boehm's amazing article on CUDA matrix multiplication. I learned all concepts from there. The last kernel has been optimised by me. We acknowledge the valuable resources and documentation provided by NVIDIA and the CUDA community.
https://siboehm.com/articles/22/CUDA-MMM


