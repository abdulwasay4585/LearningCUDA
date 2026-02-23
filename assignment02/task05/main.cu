#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

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

__global__ void gpuMatrixMultiplyTiled(const float* A, const float* B, float* C, int N) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        if (row < N && m * TILE_WIDTH + threadIdx.x < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + m * TILE_WIDTH + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (m * TILE_WIDTH + threadIdx.y < N && col < N)
            sB[threadIdx.y][threadIdx.x] = B[(m * TILE_WIDTH + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}


int main() {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048};
    std::ofstream outFile("metrics_task5.csv");
    outFile << "Size,CPU_Time_ms,GPU_Naive_Time_ms,GPU_Tiled_Time_ms" << std::endl;

    for (int N : sizes) {
        std::cout << "Testing matrix size: " << N << "x" << N << std::endl;
        size_t sizeBytes = N * N * sizeof(float);

        std::vector<float> h_A(N * N, 1.0f);
        std::vector<float> h_B(N * N, 2.0f);
        std::vector<float> h_CCPU(N * N, 0.0f);
        std::vector<float> h_CGPUNaive(N * N, 0.0f);
        std::vector<float> h_CGPUTiled(N * N, 0.0f);

        // CPU Timing
        auto startCPU = std::chrono::high_resolution_clock::now();
        cpuMatrixMultiply(h_A, h_B, h_CCPU, N);
        auto endCPU = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> durationCPU = endCPU - startCPU;

        // GPU Native Setup
        cudaEvent_t startGPU, stopGPU;
        cudaEventCreate(&startGPU);
        cudaEventCreate(&stopGPU);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, sizeBytes);
        cudaMalloc(&d_B, sizeBytes);
        cudaMalloc(&d_C, sizeBytes);
        
        // --- GPU Naive Timing --- 
        cudaEventRecord(startGPU);
        cudaMemcpy(d_A, h_A.data(), sizeBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), sizeBytes, cudaMemcpyHostToDevice);

        dim3 threadsPerBlockNaive(16, 16);
        dim3 blocksPerGridNaive((N + threadsPerBlockNaive.x - 1) / threadsPerBlockNaive.x,
                           (N + threadsPerBlockNaive.y - 1) / threadsPerBlockNaive.y);

        gpuMatrixMultiplyNaive<<<blocksPerGridNaive, threadsPerBlockNaive>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        cudaMemcpy(h_CGPUNaive.data(), d_C, sizeBytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stopGPU);
        cudaEventSynchronize(stopGPU);

        float durationGPUNaive = 0;
        cudaEventElapsedTime(&durationGPUNaive, startGPU, stopGPU);
        
        // --- GPU Tiled Timing --- 
        cudaEventRecord(startGPU);
        cudaMemcpy(d_A, h_A.data(), sizeBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), sizeBytes, cudaMemcpyHostToDevice);

        dim3 threadsPerBlockTiled(TILE_WIDTH, TILE_WIDTH);
        dim3 blocksPerGridTiled((N + threadsPerBlockTiled.x - 1) / threadsPerBlockTiled.x,
                           (N + threadsPerBlockTiled.y - 1) / threadsPerBlockTiled.y);

        gpuMatrixMultiplyTiled<<<blocksPerGridTiled, threadsPerBlockTiled>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        cudaMemcpy(h_CGPUTiled.data(), d_C, sizeBytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stopGPU);
        cudaEventSynchronize(stopGPU);

        float durationGPUTiled = 0;
        cudaEventElapsedTime(&durationGPUTiled, startGPU, stopGPU);


        outFile << N << "," << durationCPU.count() << "," << durationGPUNaive << "," << durationGPUTiled << std::endl;

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
