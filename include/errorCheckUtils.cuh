#pragma once

#include <cuda_runtime.h>

// cuda error checking macro
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    }} while(0)