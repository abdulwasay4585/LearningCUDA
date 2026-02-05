#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>

// CUDA Kernel
__global__ void matrixAddKernel(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

void checkCuda(cudaError_t result, const char *msg)
{
    if (result != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(result) << "\n";
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> [output_file]\n";
        return 1;
    }

    try
    {
        std::ifstream inFile(argv[1]);
        if (!inFile)
            throw std::runtime_error("Cannot open input file.");

        int rows, cols;
        if (!(inFile >> rows >> cols))
            throw std::runtime_error("Invalid header.");
        int N = rows * cols;
        size_t size = N * sizeof(float);

        std::vector<float> h_A(N), h_B(N), h_C(N);
        for (int i = 0; i < N; i++)
            inFile >> h_A[i];
        for (int i = 0; i < N; i++)
            inFile >> h_B[i];

        float *d_A, *d_B, *d_C;

        // --- GPU Timing (Data Transfer + Compute) ---
        auto start = std::chrono::high_resolution_clock::now();

        checkCuda(cudaMalloc(&d_A, size), "Alloc A");
        checkCuda(cudaMalloc(&d_B, size), "Alloc B");
        checkCuda(cudaMalloc(&d_C, size), "Alloc C");

        checkCuda(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice), "Copy A");
        checkCuda(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice), "Copy B");

        // Kernel Launch (256 threads per block is standard)
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        matrixAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

        checkCuda(cudaGetLastError(), "Kernel Launch");
        checkCuda(cudaDeviceSynchronize(), "Kernel Sync");

        checkCuda(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost), "Copy C");

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        // Print time to cerr for benchmark script
        std::cerr << duration.count() << std::endl;

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // Output handling
        if (argc >= 3)
        {
            std::ofstream outFile(argv[2]);
            outFile << rows << " " << cols << "\n";
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    outFile << std::fixed << std::setprecision(2) << h_C[i * cols + j] << (j == cols - 1 ? "" : " ");
                }
                outFile << "\n";
            }
        }
        else
        {
            std::cout << rows << " " << cols << "\n";
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    std::cout << std::fixed << std::setprecision(2) << h_C[i * cols + j] << (j == cols - 1 ? "" : " ");
                }
                std::cout << "\n";
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}