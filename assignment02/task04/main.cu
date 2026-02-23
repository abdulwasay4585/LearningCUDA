#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>

void cpuMatrixMultiply(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void gpuMatrixMultiplyNaive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048};
    std::ofstream outFile("metrics.csv");
    outFile << "Size,CPU_Time_ms,GPU_Naive_Time_ms" << std::endl;

    for (int N : sizes) {
        std::cout << "Testing matrix size: " << N << "x" << N << std::endl;
        size_t sizeBytes = N * N * sizeof(float);

        std::vector<float> h_A(N * N, 1.0f);
        std::vector<float> h_B(N * N, 2.0f);
        std::vector<float> h_CCPU(N * N, 0.0f);
        std::vector<float> h_CGPU(N * N, 0.0f);

        // CPU Timing
        auto startCPU = std::chrono::high_resolution_clock::now();
        cpuMatrixMultiply(h_A, h_B, h_CCPU, N);
        auto endCPU = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> durationCPU = endCPU - startCPU;

        // GPU Timing (Including data transfer)
        cudaEvent_t startGPU, stopGPU;
        cudaEventCreate(&startGPU);
        cudaEventCreate(&stopGPU);

        cudaEventRecord(startGPU);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, sizeBytes);
        cudaMalloc(&d_B, sizeBytes);
        cudaMalloc(&d_C, sizeBytes);

        cudaMemcpy(d_A, h_A.data(), sizeBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), sizeBytes, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

        gpuMatrixMultiplyNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        cudaMemcpy(h_CGPU.data(), d_C, sizeBytes, cudaMemcpyDeviceToHost);

        cudaEventRecord(stopGPU);
        cudaEventSynchronize(stopGPU);

        float durationGPU = 0;
        cudaEventElapsedTime(&durationGPU, startGPU, stopGPU);

        outFile << N << "," << durationCPU.count() << "," << durationGPU << std::endl;

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(startGPU);
        cudaEventDestroy(stopGPU);
    }
    
    outFile.close();
    std::cout << "Benchmarking complete." << std::endl;

    return 0;
}
