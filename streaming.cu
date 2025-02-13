#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElems) {
    // conditional in place when num threads exceed needed operations. 
    // E.g 256 threads/block * 4 blocks = 1024 threads, but only 1000 needed
    // Extra 24 threads will be ignored
    if (threadIdx.x >= numElems) {
        return;
    }
    for (int i = 0; i < numElems; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // PROBLEM: optimize a large vector addition process using streaming and paralellism

    // currently in realms of host (CPU) so can use non-cuda operations
    int numRows = 50000;
    size_t size = numRows * sizeof(float);
    float *h_A, *h_B, *h_C;

    // Allocate host memory
    h_A = (float *) malloc(size);
    h_B = (float *) malloc(size);
    h_C = (float *) malloc(size);

    // Initialize host arrays
    for (int i = 0; i < numRows; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate memory for device arrays
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // create streams to async-ly copy data to device
    cudaStream_t stream1, stream2;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C, numRows);
    int threadsPerBlock = 256;
    int blocksPerGrid = (numRows + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(h_A, h_B, h_C, size);
    

    // copy data back to host
    cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream1);

    // clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

};
