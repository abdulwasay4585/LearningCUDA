#include <iostream>
#include <fstream>
#include <vector>

void multiplyMatrices(const std::vector<float>& A, int rA, int cA,
                      const std::vector<float>& B, int cB,
                      std::vector<float>& C) {
    for (int i = 0; i < rA; ++i) {
        for (int j = 0; j < cB; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < cA; ++k) {
                sum += A[i * cA + k] * B[k * cB + j];
            }
            C[i * cB + j] = sum;
        }
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
        std::cerr << "Error: Incompatible matrix dimensions for multiplication ("
                  << rA << "x" << cA << " and " << rB << "x" << cB << ")." << std::endl;
        return 1;
    }

    std::vector<float> A(rA * cA);
    std::vector<float> B(rB * cB);

    for (int i = 0; i < rA * cA; ++i) inFile >> A[i];
    for (int i = 0; i < rB * cB; ++i) inFile >> B[i];

    inFile.close();

    std::vector<float> C(rA * cB, 0.0f);
    multiplyMatrices(A, rA, cA, B, cB, C);

    if (!outputFile.empty()) {
        std::ofstream outFile(outputFile);
        if (!outFile.is_open()) {
            std::cerr << "Error: Could not open output file " << outputFile << std::endl;
            return 1;
        }
        outFile << rA << " " << cB << std::endl;
        for (int i = 0; i < rA; ++i) {
            for (int j = 0; j < cB; ++j) {
                outFile << C[i * cB + j] << " ";
            }
            outFile << std::endl;
        }
        outFile.close();
    } else {
        std::cout << rA << " " << cB << std::endl;
        for (int i = 0; i < rA; ++i) {
            for (int j = 0; j < cB; ++j) {
                std::cout << C[i * cB + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
