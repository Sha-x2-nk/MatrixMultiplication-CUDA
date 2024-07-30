#pragma once

#include "errorCheckUtils.cuh"

#include <cuda_runtime.h>

#include <iostream>

#define WARP_SIZE 32

/*
the worst possible implementation
    input 
        A = Matrix (M, K)
        B = Matrix (K, N)
    output 
        C = Matrix (M, N) [A dot B]
*/
template <int M, int K, int N>
__global__ void matMul0(float *A, float *B, float *C){
    const int br = blockIdx.x, bc = blockIdx.y;
    const int tr = threadIdx.x, tc = threadIdx.y;

    // moving to this block's A, B, C required locations 
    A += br * blockDim.x * K; // ( br, 0 )
    B += bc * blockDim.y;     // ( 0, bc )
    C += (br * blockDim.x) * N + (bc * blockDim.y);

    for (int k = 0; k < K; ++k)
        C[tr * N + tc] += A[tr * K + k] * B[k * N + tc];
}

/*
Adding results in register and writing later
    input 
        A = Matrix (M, K)
        B = Matrix (K, N)
    output 
        C = Matrix (M, N) [A dot B]
*/
template <int M, int K, int N>
__global__ void matMul1(float *A, float *B, float *C){
    const int br = blockIdx.x, bc = blockIdx.y;
    const int tr = threadIdx.x, tc = threadIdx.y;

    // moving to this block's A, B, C required locations 
    A += br * blockDim.x * K; // ( br, 0 )
    B += bc * blockDim.y;     // ( 0, bc )
    C += (br * blockDim.x) * N + (bc * blockDim.y);

    float res = 0.0f;
    for (int k = 0; k < K; ++k)
        res += A[tr * K + k] * B[k * N + tc];

    C[tr * N + tc] = res;
}

/*
Optimising memory access (COALESCING). consecutive threads access consecutive elements
    input 
        A = Matrix (M, K)
        B = Matrix (K, N)
    output 
        C = Matrix (M, N) [A dot B]
*/
template <int M, int K, int N>
__global__ void matMul2(float *A, float *B, float *C){
    const int br = blockIdx.y, bc = blockIdx.x;
    const int tr = threadIdx.y, tc = threadIdx.x;

    // moving to this block's A, B, C required locations 
    A += br * blockDim.y * K; // ( br, 0 )
    B += bc * blockDim.x;     // ( 0, bc )
    C += (br * blockDim.y) * N + (bc * blockDim.x);

    float res = 0.0f;
    for (int k = 0; k < K; ++k)
        res += A[tr * K + k] * B[k * N + tc];

    C[tr * N + tc] = res;
}

/*
Using shared memory to reduce global memory accesses
    input 
        A = Matrix (M, K)
        B = Matrix (K, N)
        BM, BK, BN = Block Tile Sizes for M, K, N respectively
            each Block computes BM x BN elements of C and BK is the tile size for k loop used inside
    output 
        C = Matrix (M, N) [A dot B]
*/
template <int M, int K, int N, int BM, int BK, int BN>
__global__ void matMul3(float *A, float *B, float *C){
    const int br = blockIdx.y, bc = blockIdx.x;
    const int tr = threadIdx.x / (BN), tc = threadIdx.x % (BN);

    // moving to this block's A, B, C required locations 
    A += br * BM * K; // ( br, 0 )
    B += bc * BN;     // ( 0, bc )
    C += (br * BM) * N + (bc * BN);

    __shared__ float s_A[BM * BK];
    __shared__ float s_B[BK * BN];


    /*
    variables to populate the entire shared mem. 
    Basic Idea is to arrange them such that, each thread loads at least one and unique element, and all elements are loaded. 
    */
    
    // const int s_a_r= ( threadIdx.x / (BN) );
    // const int s_a_c= ( threadIdx.x % (BN) );
    // const int strideA= (blockDim.x / (BN)); // worst case if each thread has to do more than 1 load to fully populate ShMem
    // yaha p ye index tr, tc k equal hi hain.

    float res = 0.0f;

    // tiling on K
    for (int tile = 0; tile < K; tile += BK){
        s_A[tr * BK + tc] = A[tr * K + tc];
        s_B[tr * BN + tc] = B[tr * N + tc];

        __syncthreads();

        A += BK;     // row const. col change
        B += BK * N; // col const. row change

        // computing result for this tile
        for (int k = 0; k < BK; ++k)
            res += s_A[tr * BK + k] * s_B[k * BN + tc];

        __syncthreads();
    }

    C[tr * N + tc] = res;
}

/*
Vectorised GMem accesses, using reinterpret_cast to perform 128-bit loads.
    input 
        A = Matrix (M, K)
        B = Matrix (K, N)
        BM, BK, BN = Block Tile Sizes for M, K, N respectively
            each Block computes BM x BN elements of C and BK is the tile size for k loop used inside
    output 
        C = Matrix (M, N) [A dot B]
*/
template <int M, int K, int N, int BM, int BK, int BN>
__global__ void matMul4(float *A, float *B, float *C){
    const int br = blockIdx.y, bc = blockIdx.x;
    const int tr = threadIdx.x / (BN), tc = threadIdx.x % (BN);

    // moving to this block's A, B, C required locations 
    A += br * BM * K; // ( br, 0 )
    B += bc * BN;     // ( 0, bc )
    C += (br * BM) * N + (bc * BN);

    __shared__ float s_A[BM * BK];
    __shared__ float s_B[BK * BN];

    float res = 0.0f;

    /*
    variables to populate the entire shared mem. 
    Basic Idea is to arrange them such that, each thread loads at least one and unique element, and all elements are loaded. 
    */
    const int s_a_r = (threadIdx.x) / (BK / 4); // 4, since we will load 4 floats at once
    const int s_a_c = (threadIdx.x) % (BK / 4);
    const int strideA = (blockDim.x) / (BK / 4);

    const int s_b_r = (threadIdx.x) / (BN / 4);
    const int s_b_c = (threadIdx.x) % (BN / 4);
    const int strideB = (blockDim.x) / (BN / 4);

    // tiling on K
    for (int tile = 0; tile < K; tile += BK){
        for (int rowOffset = 0; (s_a_r + rowOffset) < BM; rowOffset += strideA)
            reinterpret_cast<float4 *>(&s_A[(s_a_r + rowOffset) * BK + s_a_c * 4])[0] = reinterpret_cast<float4 *>(&A[(s_a_r + rowOffset) * K + s_a_c * 4])[0];

        for (int rowOffset = 0; (s_b_r + rowOffset) < BK; rowOffset += strideB)
            reinterpret_cast<float4 *>(&s_B[(s_b_r + rowOffset) * BN + s_b_c * 4])[0] = reinterpret_cast<float4 *>(&B[(s_b_r + rowOffset) * N + s_b_c * 4])[0];

        __syncthreads();

        A += BK;     // row const. col change
        B += BK * N; // col const. row change

        // calculate results
        for (int k = 0; k < BK; ++k)
            res += s_A[tr * BK + k] * s_B[k * BN + tc];

        __syncthreads();
    }

    C[tr * N + tc] = res;
}

/*
Additional Thread level tiling. why Along cols?? Row was slower.
    input 
        A = Matrix (M, K)
        B = Matrix (K, N)
        BM, BK, BN = Block Tile Sizes for M, K, N respectively
        TN = Thread Tile
            each Block computes BM x BN elements of C and BK is the tile size for k loop used inside. 
            each thread computes TN elements of C
    output 
        C = Matrix (M, N) [A dot B]
*/
template <int M, int K, int N, int BM, int BK, int BN, int TM>
__global__ void matMul5(float *A, float *B, float *C){
    const int br = blockIdx.y, bc = blockIdx.x;
    const int tr = threadIdx.x / (BN), tc = threadIdx.x % (BN);

    // moving to this block's A, B, C required locations 
    A += br * BM * K; // ( br, 0 )
    B += bc * BN;     // ( 0, bc )
    C += (br * BM) * N + (bc * BN);

    __shared__ float s_A[BM * BK];
    __shared__ float s_B[BK * BN];

    float res[TM] = {0.0f};

    /*
    variables to populate the entire shared mem. 
    Basic Idea is to arrange them such that, each thread loads at least one and unique element, and all elements are loaded. 
    */
    const int s_a_r = (threadIdx.x) / (BK / 4); // 4, since we will load 4 floats at once
    const int s_a_c = (threadIdx.x) % (BK / 4);
    const int strideA = (blockDim.x) / (BK / 4);

    const int s_b_r = (threadIdx.x) / (BN / 4);
    const int s_b_c = (threadIdx.x) % (BN / 4);
    const int strideB = (blockDim.x) / (BN / 4);

    // tiling on k
    for (int tile = 0; tile < K; tile += BK){
        // load krne ka simple logic. all threads should put atleast one element and entire shared me should be filled.
        for (int rowOffset = 0; (s_a_r + rowOffset) < BM; rowOffset += strideA)
            reinterpret_cast<float4 *>(&s_A[(s_a_r + rowOffset) * BK + s_a_c * 4])[0] = reinterpret_cast<float4 *>(&A[(s_a_r + rowOffset) * K + s_a_c * 4])[0];

        for (int rowOffset = 0; (s_b_r + rowOffset) < BK; rowOffset += strideB)
            reinterpret_cast<float4 *>(&s_B[(s_b_r + rowOffset) * BN + s_b_c * 4])[0] = reinterpret_cast<float4 *>(&B[(s_b_r + rowOffset) * N + s_b_c * 4])[0];

        __syncthreads();

        A += BK;     // row const. col change
        B += BK * N; // col const. row change


        // Compute results
        for (int k = 0; k < BK; ++k){
            float b_tmp = s_B[k * BN + tc]; // stored in register for fastest access possible.
            for (int m = 0; m < TM; ++m)
                res[m] += s_A[(tr * TM + m) * BK + k] * b_tmp;
        }

        __syncthreads();
    }

    for (int m = 0; m < TM; ++m)
        C[(tr * TM + m) * N + tc] = res[m];
}

/*
Additional Thread level tiling.
    input 
        A = Matrix (M, K)
        B = Matrix (K, N)
        BM, BK, BN = Block Tile Sizes for M, K, N respectively
        TM, TN = Thread Tile for M, N
            each Block computes BM x BN elements of C and BK is the tile size for k loop used inside. 
            each thread computes TM x TN elements of C
    output 
        C = Matrix (M, N) [A dot B]
*/
template <int M, int K, int N, int BM, int BK, int BN, int TM, int TN>
__global__ void matMul6(float *A, float *B, float *C){
    const int br = blockIdx.y, bc = blockIdx.x;
    const int tr = threadIdx.x / (BN / TN), tc = threadIdx.x % (BN / TN);

    // moving to this block's A, B, C required locations 
    A += br * BM * K; // ( br, 0 )
    B += bc * BN;     // ( 0, bc )
    C += (br * BM) * N + (bc * BN);

    __shared__ float s_A[BM * BK];
    __shared__ float s_B[BK * BN];

    float res[TM * TN] = {0.0f};
    float a_tmp[TM]; // for extremely fast accesses
    float b_tmp[TN];

    /*
    variables to populate the entire shared mem. 
    Basic Idea is to arrange them such that, each thread loads at least one and unique element, and all elements are loaded. 
    */
    const int s_a_r = (threadIdx.x) / (BK / 4); // 4, since we will load 4 floats at once
    const int s_a_c = (threadIdx.x) % (BK / 4);
    const int strideA = (blockDim.x) / (BK / 4);

    const int s_b_r = (threadIdx.x) / (BN / 4);
    const int s_b_c = (threadIdx.x) % (BN / 4);
    const int strideB = (blockDim.x) / (BN / 4);

    // tiling on k
    for (int tile = 0; tile < K; tile += BK){
        for (int rowOffset = 0; (s_a_r + rowOffset) < BM; rowOffset += strideA)
            reinterpret_cast<float4 *>(&s_A[(s_a_r + rowOffset) * BK + s_a_c * 4])[0] = reinterpret_cast<float4 *>(&A[(s_a_r + rowOffset) * K + s_a_c * 4])[0];

        for (int rowOffset = 0; (s_b_r + rowOffset) < BK; rowOffset += strideB)
            reinterpret_cast<float4 *>(&s_B[(s_b_r + rowOffset) * BN + s_b_c * 4])[0] = reinterpret_cast<float4 *>(&B[(s_b_r + rowOffset) * N + s_b_c * 4])[0];

        __syncthreads();

        A += BK;     // row const. col change
        B += BK * N; // col const. row change

        for (int k = 0; k < BK; ++k){
            // populating registers for fastest access
            for (int m = 0; m < TM; ++m)
                a_tmp[m] = s_A[(tr * TM + m) * BK + k];

            for (int n = 0; n < TN; ++n)
                b_tmp[n] = s_B[k * BN + (tc * TN + n)];

            // computing results
            for (int m = 0; m < TM; ++m)
                for (int n = 0; n < TN; ++n)
                    res[m * TN + n] += a_tmp[m] * b_tmp[n];
        }

        __syncthreads();
    }

    // Vectorised stores
    for (int m = 0; m < TM; ++m)
        for (int n = 0; n < TN; n += 4){
            int idx = m * TN + n;
            float4 tmp = make_float4(res[idx], res[idx + 1], res[idx + 2], res[idx + 3]);

            reinterpret_cast<float4 *>(&C[(tr * TM + m) * N + (tc * TN + n)])[0] = tmp;
        }
}

/*
Additional Thread level tiling + storing s_A as transposed to increase locality.
    input 
        A = Matrix (M, K)
        B = Matrix (K, N)
        BM, BK, BN = Block Tile Sizes for M, K, N respectively
        TM, TN = Thread Tile for M, N
            each Block computes BM x BN elements of C and BK is the tile size for k loop used inside. 
            each thread computes TM x TN elements of C
    output 
        C = Matrix (M, N) [A dot B]
*/
template <int M, int K, int N, int BM, int BK, int BN, int TM, int TN>
__global__ void matMul7(float *A, float *B, float *C){
    const int br = blockIdx.y, bc = blockIdx.x;
    const int tr = threadIdx.x / (BN / TN), tc = threadIdx.x % (BN / TN);

    // moving to this block's A, B, C required locations 
    A += br * BM * K; // ( br, 0 )
    B += bc * BN;     // ( 0, bc )
    C += (br * BM) * N + (bc * BN);

    __shared__ float s_A[BM * BK];
    __shared__ float s_B[BK * BN];

    float res[TM * TN] = {0.0f};
    float a_tmp[TM]; // for fastest accesses
    float b_tmp[TN];

    /*
    variables to populate the entire shared mem. 
    Basic Idea is to arrange them such that, each thread loads at least one and unique element, and all elements are loaded. 
    */
    const int s_a_r = (threadIdx.x) / (BK / 4); // 4, since we will load 4 floats at once
    const int s_a_c = (threadIdx.x) % (BK / 4);
    const int strideA = (blockDim.x) / (BK / 4);

    const int s_b_r = (threadIdx.x) / (BN / 4);
    const int s_b_c = (threadIdx.x) % (BN / 4);
    const int strideB = (blockDim.x) / (BN / 4);

    // tiling on k
    for (int tile = 0; tile < K; tile += BK){
        for (int rowOffset = 0; (s_a_r + rowOffset) < BM; rowOffset += strideA){
            float4 tmp = reinterpret_cast<float4 *>(&A[(s_a_r + rowOffset) * K + s_a_c * 4])[0];
            s_A[(s_a_c * 4) * BM + (s_a_r + rowOffset)] = tmp.x;
            s_A[(s_a_c * 4 + 1) * BM + (s_a_r + rowOffset)] = tmp.y;
            s_A[(s_a_c * 4 + 2) * BM + (s_a_r + rowOffset)] = tmp.z;
            s_A[(s_a_c * 4 + 3) * BM + (s_a_r + rowOffset)] = tmp.w;
        }

        for (int rowOffset = 0; (s_b_r + rowOffset) < BK; rowOffset += strideB)
            reinterpret_cast<float4 *>(&s_B[(s_b_r + rowOffset) * BN + s_b_c * 4])[0] = reinterpret_cast<float4 *>(&B[(s_b_r + rowOffset) * N + s_b_c * 4])[0];

        __syncthreads();

        A += BK;     // row const. col change
        B += BK * N; // col const. row change

        for (int k = 0; k < BK; ++k){
            // populating registers for fastest access
            for (int m = 0; m < TM; ++m)
                a_tmp[m] = s_A[k * BM + (tr * TM + m)];

            for (int n = 0; n < TN; ++n)
                b_tmp[n] = s_B[k * BN + (tc * TN + n)];

            // computing results
            for (int m = 0; m < TM; ++m)
                for (int n = 0; n < TN; ++n)
                    res[m * TN + n] += a_tmp[m] * b_tmp[n];
        }

        __syncthreads();
    }

    // Vectorised stores
    for (int m = 0; m < TM; ++m)
        for (int n = 0; n < TN; n += 4){
            int idx = m * TN + n;
            float4 tmp = make_float4(res[idx], res[idx + 1], res[idx + 2], res[idx + 3]);

            reinterpret_cast<float4 *>(&C[(tr * TM + m) * N + (tc * TN + n)])[0] = tmp;
        }
}

/*
Additional Warp level tiling.
    input 
        A = Matrix (M, K)
        B = Matrix (K, N)
        BM, BK, BN = Block Tile Sizes for M, K, N respectively
        TM, TN = Thread Tile for M, N
        WM, WN = Warp Tile for M, N
            each block computes BM x BN elements of C and BK is the tile size for k loop used inside. 
            each thread computes TM x TN elements of C
            each warp computes WM x WN elements of C
    output 
        C = Matrix (M, N) [A dot B]
*/
template <int M, int K, int N, int BM, int BK, int BN, int TM, int TN, int WM, int WN, int WNITER>
__global__ void matMul8(float *A, float *B, float *C){
    const int br = blockIdx.y, bc = blockIdx.x;

    const int W_IDX = threadIdx.x / WARP_SIZE;
    const int wr = W_IDX / (BN / WN);
    const int wc = W_IDX % (BN / WN);

    constexpr int WMITER = (WM * WN) / (WARP_SIZE * TM * TN * WNITER); // 2
    constexpr int WSUBM = WM / WMITER;                                 // 32
    constexpr int WSUBN = WN / WNITER;                                 // 32

    // thread location in warp
    const int threadIdxInWarp = threadIdx.x % WARP_SIZE;        // [0, 31]
    const int threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // threadIdxInWarp / (WN/(WNITER*TN))
    const int threadColInWarp = threadIdxInWarp % (WSUBN / TN); // threadIdxInWarp / (WN/(WNITER*TN))

    // moving to this block's A, B, C required locations 
    A += br * BM * K; // ( br, 0 )
    B += bc * BN;     // ( 0, bc )
    // C to its warp location
    C += (br * BM + wr * WM) * N + (bc * BN + wc * WN);

    __shared__ float s_A[BM * BK];
    __shared__ float s_B[BK * BN];

    float res[WMITER * TM * WNITER * TN] = {0.0f};
    float a_tmp[WMITER * TM]; // for fastest accesses
    float b_tmp[WNITER * TN];

    /*
    variables to populate the entire shared mem. 
    Basic Idea is to arrange them such that, each thread loads at least one and unique element, and all elements are loaded. 
    */
    const int s_a_r = (threadIdx.x) / (BK / 4); // 4, since we will load 4 floats at once
    const int s_a_c = (threadIdx.x) % (BK / 4);
    const int strideA = (blockDim.x) / (BK / 4);

    const int s_b_r = (threadIdx.x) / (BN / 4);
    const int s_b_c = (threadIdx.x) % (BN / 4);
    const int strideB = (blockDim.x) / (BN / 4);

    // tiling on k
    for (int tile = 0; tile < K; tile += BK){
        for (int rowOffset = 0; (s_a_r + rowOffset) < BM; rowOffset += strideA){
            float4 tmp = reinterpret_cast<float4 *>(&A[(s_a_r + rowOffset) * K + s_a_c * 4])[0];
            s_A[(s_a_c * 4) * BM + (s_a_r + rowOffset)] = tmp.x;
            s_A[(s_a_c * 4 + 1) * BM + (s_a_r + rowOffset)] = tmp.y;
            s_A[(s_a_c * 4 + 2) * BM + (s_a_r + rowOffset)] = tmp.z;
            s_A[(s_a_c * 4 + 3) * BM + (s_a_r + rowOffset)] = tmp.w;
        }

        for (int rowOffset = 0; (s_b_r + rowOffset) < BK; rowOffset += strideB)
            reinterpret_cast<float4 *>(&s_B[(s_b_r + rowOffset) * BN + s_b_c * 4])[0] = reinterpret_cast<float4 *>(&B[(s_b_r + rowOffset) * N + s_b_c * 4])[0];

        __syncthreads();

        A += BK;     // row const. col change
        B += BK * N; // col const. row change

        for (int k = 0; k < BK; ++k){
            // populating registers for fastest access
            for (int wm = 0; wm < WMITER; ++wm)
                for (int m = 0; m < TM; ++m)
                    a_tmp[wm * TM + m] = s_A[k * BM + (wr * WM + wm * WSUBM + threadRowInWarp * TM + m)];

            for (int wn = 0; wn < WNITER; ++wn)
                for (int n = 0; n < TN; ++n)
                    b_tmp[wn * TN + n] = s_B[k * BN + (wc * WN + wn * WSUBN + threadColInWarp * TN + n)];

            // computing results
            for (int wm = 0; wm < WMITER; ++wm)
                for (int wn = 0; wn < WNITER; ++wn)
                    for (int m = 0; m < TM; ++m)
                        for (int n = 0; n < TN; ++n)
                            res[(wm * TM + m) * (WNITER * TN) + (wn * TN + n)] += a_tmp[wm * TM + m] * b_tmp[wn * TN + n];
        }

        __syncthreads();
    }

    // Vectorised Loads
    for (int wm = 0; wm < WMITER; ++wm)
        for (int wn = 0; wn < WNITER; ++wn){
            float *C_interim = (C + (wm * WSUBM) * N + (wn * WSUBN));
            for (int m = 0; m < TM; ++m)
                for (int n = 0; n < TN; n += 4){
                    int idx = (wm * TM + m) * (WNITER * TN) + (wn * TN + n);
                    float4 tmp = make_float4(res[idx], res[idx + 1], res[idx + 2], res[idx + 3]);

                    reinterpret_cast<float4 *>(&C_interim[(threadRowInWarp * TM + m) * N + (threadColInWarp * TN + n)])[0] = tmp;
                }
        }
}

// Final Kernel, similar to above but all constants have been coded explicitly and loops have been unrolled
__global__ void __launch_bounds__(128) matMul9(float *A, float *B, float *C){
    const int br = blockIdx.y, bc = blockIdx.x;

    const int W_IDX = threadIdx.x / 32;
    const int wr = W_IDX >> 2;
    const int wc = W_IDX % 4;

    // thread location in warp
    const int threadIdxInWarp = threadIdx.x % WARP_SIZE;
    const int threadRowInWarp = threadIdxInWarp / 4;
    const int threadColInWarp = threadIdxInWarp % 4;

    // moving to this block's A, B, C required locations 
    A += br * 262144; // ( br, 0 )
    B += bc * 128;    // ( 0, bc )
    // C to its warp location
    C += (br + wr) * 262144 + (bc * 128 + wc * 32);

    __shared__ float s_A[512];
    __shared__ float s_B[1024];

    float res[64] = {0.0f};
    float a_tmp[8]; // for fastest accesses
    float b_tmp[8];

    /*
    variables to populate the entire shared mem. 
    Basic Idea is to arrange them such that, each thread loads at least one and unique element, and all elements are loaded. 
    */
    const int s_a_r = (threadIdx.x >> 1);
    const int s_a_c = (threadIdx.x & 1);
    // const int strideA = (128) / 2;

    const int s_b_r = (threadIdx.x) / 32;
    const int s_b_c = (threadIdx.x) % 32;
    // const int strideB = (128) / 32;

    // tile chlegi 4096 ki
    for (int tile = 0; tile < 4096; tile += 8){
        // load krne ka simple logic. all threads should put atleast one element and entire shared me should be filled.
        float4 tmp = reinterpret_cast<float4 *>(&A[s_a_r * 4096 + s_a_c * 4])[0];
        s_A[(s_a_c * 4) * 64 + s_a_r] = tmp.x;
        s_A[(s_a_c * 4 + 1) * 64 + s_a_r] = tmp.y;
        s_A[(s_a_c * 4 + 2) * 64 + s_a_r] = tmp.z;
        s_A[(s_a_c * 4 + 3) * 64 + s_a_r] = tmp.w;

        reinterpret_cast<float4 *>(&s_B[(s_b_r) * 128 + s_b_c * 4])[0] = reinterpret_cast<float4 *>(&B[(s_b_r) * 4096 + s_b_c * 4])[0];

        reinterpret_cast<float4 *>(&s_B[(s_b_r + 4) * 128 + s_b_c * 4])[0] = reinterpret_cast<float4 *>(&B[(s_b_r + 4) * 4096 + s_b_c * 4])[0];

        __syncthreads();

        for (char k = 0; k < 8; ++k){
            // populating registers
            a_tmp[0] = s_A[k * 64 + (wr * 64 + threadRowInWarp * 4)];
            a_tmp[1] = s_A[k * 64 + (wr * 64 + threadRowInWarp * 4 + 1)];
            a_tmp[2] = s_A[k * 64 + (wr * 64 + threadRowInWarp * 4 + 2)];
            a_tmp[3] = s_A[k * 64 + (wr * 64 + threadRowInWarp * 4 + 3)];
            a_tmp[4] = s_A[k * 64 + (wr * 64 + 32 + threadRowInWarp * 4)];
            a_tmp[5] = s_A[k * 64 + (wr * 64 + 33 + threadRowInWarp * 4)];
            a_tmp[6] = s_A[k * 64 + (wr * 64 + 34 + threadRowInWarp * 4)];
            a_tmp[7] = s_A[k * 64 + (wr * 64 + 35 + threadRowInWarp * 4)];

            for (char n = 0; n < 8; ++n)
                b_tmp[n] = s_B[k * 128 + (wc * 32 + threadColInWarp * 8 + n)];

            // computing results
            for (char n = 0; n < 8; ++n)
                res[n] += a_tmp[0] * b_tmp[n];

            for (char n = 0; n < 8; ++n)
                res[8 + n] += a_tmp[1] * b_tmp[n];

            for (char n = 0; n < 8; ++n)
                res[16 + n] += a_tmp[2] * b_tmp[n];

            for (char n = 0; n < 8; ++n)
                res[24 + n] += a_tmp[3] * b_tmp[n];

            for (char n = 0; n < 8; ++n)
                res[32 + n] += a_tmp[4] * b_tmp[n];

            for (char n = 0; n < 8; ++n)
                res[40 + n] += a_tmp[5] * b_tmp[n];

            for (char n = 0; n < 8; ++n)
                res[48 + n] += a_tmp[6] * b_tmp[n];

            for (char n = 0; n < 8; ++n)
                res[56 + n] += a_tmp[7] * b_tmp[n];
        }

        A += 8;     // row const. col change
        B += 32768; // col const. row change
        __syncthreads();
    }

    // Vectorised stores
    float4 tmp;

    tmp = make_float4(res[0], res[1], res[2], res[3]);
    reinterpret_cast<float4 *>(&C[(threadRowInWarp * 4) * 4096 + (threadColInWarp * 8)])[0] = tmp;

    tmp = make_float4(res[4], res[5], res[6], res[7]);
    reinterpret_cast<float4 *>(&C[(threadRowInWarp * 4) * 4096 + (threadColInWarp * 8 + 4)])[0] = tmp;

    tmp = make_float4(res[8], res[9], res[10], res[11]);
    reinterpret_cast<float4 *>(&C[(threadRowInWarp * 4 + 1) * 4096 + (threadColInWarp * 8)])[0] = tmp;

    tmp = make_float4(res[12], res[13], res[14], res[15]);
    reinterpret_cast<float4 *>(&C[(threadRowInWarp * 4 + 1) * 4096 + (threadColInWarp * 8 + 4)])[0] = tmp;

    tmp = make_float4(res[16], res[17], res[18], res[19]);
    reinterpret_cast<float4 *>(&C[(threadRowInWarp * 4 + 2) * 4096 + (threadColInWarp * 8)])[0] = tmp;

    tmp = make_float4(res[20], res[21], res[22], res[23]);
    reinterpret_cast<float4 *>(&C[(threadRowInWarp * 4 + 2) * 4096 + (threadColInWarp * 8 + 4)])[0] = tmp;

    tmp = make_float4(res[24], res[25], res[26], res[27]);
    reinterpret_cast<float4 *>(&C[(threadRowInWarp * 4 + 3) * 4096 + (threadColInWarp * 8)])[0] = tmp;

    tmp = make_float4(res[28], res[29], res[30], res[31]);
    reinterpret_cast<float4 *>(&C[(threadRowInWarp * 4 + 3) * 4096 + (threadColInWarp * 8 + 4)])[0] = tmp;

    tmp = make_float4(res[32], res[33], res[34], res[35]);
    reinterpret_cast<float4 *>(&C[(threadRowInWarp * 4 + 32) * 4096 + (threadColInWarp * 8)])[0] = tmp;

    tmp = make_float4(res[36], res[37], res[38], res[39]);
    reinterpret_cast<float4 *>(&C[(threadRowInWarp * 4 + 32) * 4096 + (threadColInWarp * 8 + 4)])[0] = tmp;

    tmp = make_float4(res[40], res[41], res[42], res[43]);
    reinterpret_cast<float4 *>(&C[(threadRowInWarp * 4 + 33) * 4096 + (threadColInWarp * 8)])[0] = tmp;

    tmp = make_float4(res[44], res[45], res[46], res[47]);
    reinterpret_cast<float4 *>(&C[(threadRowInWarp * 4 + 33) * 4096 + (threadColInWarp * 8 + 4)])[0] = tmp;

    tmp = make_float4(res[48], res[49], res[50], res[51]);
    reinterpret_cast<float4 *>(&C[(threadRowInWarp * 4 + 34) * 4096 + (threadColInWarp * 8)])[0] = tmp;

    tmp = make_float4(res[52], res[53], res[54], res[55]);
    reinterpret_cast<float4 *>(&C[(threadRowInWarp * 4 + 34) * 4096 + (threadColInWarp * 8 + 4)])[0] = tmp;

    tmp = make_float4(res[56], res[57], res[58], res[59]);
    reinterpret_cast<float4 *>(&C[(threadRowInWarp * 4 + 35) * 4096 + (threadColInWarp * 8)])[0] = tmp;

    tmp = make_float4(res[60], res[61], res[62], res[63]);
    reinterpret_cast<float4 *>(&C[(threadRowInWarp * 4 + 35) * 4096 + (threadColInWarp * 8 + 4)])[0] = tmp;
}
