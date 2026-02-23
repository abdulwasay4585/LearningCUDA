#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

__global__ void matrixMultiplyNaive(const float* A, const float* B, float* C, int rA, int cA, int cB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rA && col < cB) {
        float sum = 0.0f;
        for (int k = 0; k < cA; ++k) {
            sum += A[row * cA + k] * B[k * cB + col];
        }
        C[row * cB + col] = sum;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [output_file]" << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = (argc > 2) ? argv[2] : "";

    std::ifstream inFile(inputFile);
    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open input file " << inputFile << std::endl;
        return 1;
    }

    int rA, cA, rB, cB;
    inFile >> rA >> cA >> rB >> cB;

    if (cA != rB) {
        std::cerr << "Error: Incompatible matrix dimensions (" << rA << "x" << cA << " and " << rB << "x" << cB << ")." << std::endl;
        return 1;
    }

    size_t sizeA = rA * cA * sizeof(float);
    size_t sizeB = rB * cB * sizeof(float);
    size_t sizeC = rA * cB * sizeof(float);

    std::vector<float> h_A(rA * cA);
    std::vector<float> h_B(rB * cB);

    for (int i = 0; i < rA * cA; ++i) inFile >> h_A[i];
    for (int i = 0; i < rB * cB; ++i) inFile >> h_B[i];
    inFile.close();

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cB + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rA + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMultiplyNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rA, cA, cB);
    cudaDeviceSynchronize();

    std::vector<float> h_C(rA * cB);
    cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    if (!outputFile.empty()) {
        std::ofstream outFile(outputFile);
        outFile << rA << " " << cB << std::endl;
        for (int i = 0; i < rA; ++i) {
            for (int j = 0; j < cB; ++j) {
                outFile << h_C[i * cB + j] << " ";
            }
            outFile << std::endl;
        }
        outFile.close();
    } else {
        std::cout << rA << " " << cB << std::endl;
        for (int i = 0; i < rA; ++i) {
            for (int j = 0; j < cB; ++j) {
                std::cout << h_C[i * cB + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
