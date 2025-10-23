#include <stdio.h>

__global__ void simple_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    const int N = 1024;
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    simple_kernel<<<4, 256>>>(d_data, N);
    cudaDeviceSynchronize();
    
    cudaFree(d_data);
    printf("Kernel executed successfully\n");
    return 0;
}
