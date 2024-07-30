#include "include/kernels.cuh"
#include "include/errorCheckUtils.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>

#define A_VAL 2.0f
#define B_VAL 3.0f

template <int M, int N>
inline void initMat(float *A, float val){ 
    // A is MxN matrix and fill it with val

    for (int i = 0; i < M * N; ++i)
        A[i] = val;
}

template <int M, int K, int N>
inline bool checkC(float *C){
    float C_VAL = A_VAL * B_VAL * K;

    for (int i = 0; i < M * N; ++i)
        if (C[i] != C_VAL)
            return false;

    return true;
}


template <int M, int K, int N>
void callMatMul0(float *d_A, float *d_B, float *d_C, float *h_C){
    const int BS = 16; // BLOCK_SIZE. 16 to maximumise occupancy
    dim3 block(16, 16);
    dim3 grid(M / BS, N / BS); // M x N
    matMul0<M, K, N><<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    CUDA_CALL(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    if (checkC<M, K, N>(h_C))
        printf("\n RESULT CORRECT.");
    else
        printf("\n INCORRECT RESULT.");

    // reset both h_C and d_C
    initMat<M, N>(h_C, 0);
    CUDA_CALL(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));
}


template <int M, int K, int N>
void callMatMul1(float *d_A, float *d_B, float *d_C, float *h_C){
    const int BS = 16; // BLOCK_SIZE. 16 to maximise occupancy
    dim3 block(16, 16);
    dim3 grid(M / BS, N / BS);
    matMul1<M, K, N><<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    CUDA_CALL(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    if (checkC<M, K, N>(h_C))
        printf("\n RESULT CORRECT.");
    else
        printf("\nINCORRECT RESULT.");

    // reset both h_C and d_C
    initMat<M, N>(h_C, 0);
    CUDA_CALL(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));
}


template <int M, int K, int N>
void callMatMul2(float *d_A, float *d_B, float *d_C, float *h_C){
    const int BS = 16; // BLOCK_SIZE. 16 to maximise occupancy
    dim3 block(BS, BS);
    dim3 grid(N / BS, M / BS);
    matMul2<M, K, N><<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    CUDA_CALL(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    if (checkC<M, K, N>(h_C))
        printf("\n RESULT CORRECT.");
    else
        printf("\n INCORRECT RESULT.");

    // reset both h_C and d_C
    initMat<M, N>(h_C, 0);
    CUDA_CALL(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));
}


template <int M, int K, int N>
void callMatMul3(float *d_A, float *d_B, float *d_C, float *h_C){
    const int BM = 16, BN = 16, BK = 16; // BLOCK_SIZE. 16 to maximise occupancy.

    dim3 block(BM * BN); // check code how thread rows and cols are partitioned.
    dim3 grid(N / BN, M / BM);

    matMul3<M, K, N, BM, BK, BN><<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    CUDA_CALL(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    if (checkC<M, K, N>(h_C))
        printf("\n RESULT CORRECT.");
    else
        printf("\n INCORRECT RESULT.");

    // reset both h_C and d_C
    initMat<M, N>(h_C, 0);
    CUDA_CALL(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));
}


template <int M, int K, int N>
void callMatMul4(float *d_A, float *d_B, float *d_C, float *h_C){
    const int BM = 16, BN = 16, BK = 16; // BLOCK_SIZE. 16 to maximise occupancy.
    // BK/ 4= BN such that stride is not needed.

    dim3 block(BM * BN); // check code how thread rows and cols are partitioned.
    dim3 grid(N / BN, M / BM);

    matMul4<M, K, N, BM, BK, BN><<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    CUDA_CALL(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    if (checkC<M, K, N>(h_C))
        printf("\n RESULT CORRECT.");
    else
        printf("\n INCORRECT RESULT.");

    // reset both h_C and d_C
    initMat<M, N>(h_C, 0);
    CUDA_CALL(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));
}


template <int M, int K, int N>
void callMatMul5(float *d_A, float *d_B, float *d_C, float *h_C){
    const int BM = 64, BN = 64, BK = 16, TM = 8; // BLOCK_SIZE. 16 to maximise occupancy.

    dim3 block((BM * BN) / TM); // check code how thread rows and cols are partitioned.
    dim3 grid(N / BN, M / BM);

    matMul5<M, K, N, BM, BK, BN, TM><<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    CUDA_CALL(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    if (checkC<M, K, N>(h_C))
        printf("\n RESULT CORRECT.");
    else
        printf("\n INCORRECT RESULT.");

    // reset both h_C and d_C
    initMat<M, N>(h_C, 0);
    CUDA_CALL(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));
}


template <int M, int K, int N>
void callMatMul6(float *d_A, float *d_B, float *d_C, float *h_C){
    const int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8; // BLOCK_SIZE. 16 to maximise occupancy.

    dim3 block((BM * BN) / (TM * TN)); // check code how thread rows and cols are partitioned.
    dim3 grid(N / BN, M / BM);

    matMul6<M, K, N, BM, BK, BN, TM, TN><<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    CUDA_CALL(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    if (checkC<M, K, N>(h_C))
        printf("\n RESULT CORRECT.");
    else
        printf("\n INCORRECT RESULT.");

    // reset both h_C and d_C
    initMat<M, N>(h_C, 0);
    CUDA_CALL(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));
}


template <int M, int K, int N>
void callMatMul7(float *d_A, float *d_B, float *d_C, float *h_C){
    const int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;

    dim3 block((BM * BN) / (TM * TN)); // check code how thread rows and cols are partitioned.
    dim3 grid(N / BN, M / BM);

    matMul7<M, K, N, BM, BK, BN, TM, TN><<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    CUDA_CALL(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    if (checkC<M, K, N>(h_C))
        printf("\n RESULT CORRECT.");
    else
        printf(" INCORRECT RESULT.");

    // reset both h_C and d_C
    initMat<M, N>(h_C, 0);
    CUDA_CALL(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));
}


template <int M, int K, int N>
void callMatMul8(float *d_A, float *d_B, float *d_C, float *h_C){
    const int BM = 64, BN = 128, BK = 8, TM = 4, TN = 8, WM = 64, WN = 32, WNITER = 1;

    dim3 block((BM * BN * 32) / (WM * WN)); // check code how thread rows and cols are partitioned.
    dim3 grid(N / BN, M / BM);

    matMul8<M, K, N, BM, BK, BN, TM, TN, WM, WN, WNITER><<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    CUDA_CALL(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    if (checkC<M, K, N>(h_C))
        printf("\n RESULT CORRECT.");
    else
        printf("\n INCORRECT RESULT.");

    // reset both h_C and d_C
    initMat<M, N>(h_C, 0);
    CUDA_CALL(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));
}


void callMatMul9(float *d_A, float *d_B, float *d_C, float *h_C){
    dim3 block(128);
    dim3 grid(4096 / 128, 4096 / 64);

    matMul9<<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    CUDA_CALL(cudaMemcpy(h_C, d_C, 4096 * 4096 * sizeof(float), cudaMemcpyDeviceToHost));

    if (checkC<4096, 4096, 4096>(h_C))
        printf("\n RESULT CORRECT.");
    else
        printf("\n INCORRECT RESULT.");

    // reset both h_C and d_C
    initMat<4096, 4096>(h_C, 0);
    CUDA_CALL(cudaMemcpy(d_C, h_C, 4096 * 4096 * sizeof(float), cudaMemcpyHostToDevice));
}


template <int M, int K, int N>
void callCuBlas(float *d_A, float *d_B, float *d_C, float *h_C){
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Matrix multiplication using cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
    cudaDeviceSynchronize();

    CUDA_CALL(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    if (checkC<M, K, N>(h_C))
        printf("\n RESULT CORRECT.");
    else
        printf("\n INCORRECT RESULT.");

    // reset both h_C and d_C
    initMat<M, N>(h_C, 0);
    CUDA_CALL(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    cublasDestroy(handle);
}


int main()
{
    const int M = 4096;
    const int K = 4096;
    const int N = 4096;

    // host arrays 
    float *h_A = (float *)malloc(M * K * sizeof(float));
    initMat<M, K>(h_A, A_VAL);

    float *h_B = (float *)malloc(K * N * sizeof(float));
    initMat<K, N>(h_B, B_VAL);

    float *h_C = (float *)malloc(M * N * sizeof(float));
    initMat<M, N>(h_C, 0);

    // device arrays 
    float *d_A, *d_B, *d_C;
    CUDA_CALL(cudaMalloc((void **)&d_A, M * K * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&d_B, K * N * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&d_C, M * N * sizeof(float)));

    CUDA_CALL(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // kernels calls
    callMatMul0<M, K, N>(d_A, d_B, d_C, h_C);
    callMatMul1<M, K, N>(d_A, d_B, d_C, h_C);
    callMatMul2<M, K, N>(d_A, d_B, d_C, h_C);
    callMatMul3<M, K, N>(d_A, d_B, d_C, h_C);
    callMatMul4<M, K, N>(d_A, d_B, d_C, h_C);
    callMatMul5<M, K, N>(d_A, d_B, d_C, h_C);
    callMatMul6<M, K, N>(d_A, d_B, d_C, h_C);
    callMatMul7<M, K, N>(d_A, d_B, d_C, h_C);
    callMatMul8<M, K, N>(d_A, d_B, d_C, h_C);
    callMatMul9(d_A, d_B, d_C, h_C);
    callCuBlas<M, K, N>(d_A, d_B, d_C, h_C);

    return 0;
}
