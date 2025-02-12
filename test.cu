
#include <cstdio>

__global__ void helloCUDA() {
    printf("Hello from CUDA kernel cuda!\n");
}

int main() {
    printf("Hello from CUDA kernel cuda!\n");
    helloCUDA<<<1,1>>>();
    cudaDeviceSynchronize();
    printf("Hello from CUDA kernel cuda!\n");
    return 0;
}
