#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>

// Read a matrix
void readMatrix(std::ifstream &file, std::vector<float> &mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; ++i)
    {
        if (!(file >> mat[i]))
            throw std::runtime_error("Error reading matrix data.");
    }
}

// Write a matrix
void writeMatrix(std::ostream &out, const std::vector<float> &mat, int rows, int cols)
{
    out << rows << " " << cols << "\n";
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            out << std::fixed << std::setprecision(2) << mat[i * cols + j] << (j == cols - 1 ? "" : " ");
        }
        out << "\n";
    }
}

int main(int argc, char *argv[])
{
    // Argument handling
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> [output_file]\n";
        return 1;
    }

    try
    {
        std::string inputFile = argv[1];
        std::ifstream inFile(inputFile);
        if (!inFile)
            throw std::runtime_error("Cannot open input file.");

        int rows, cols;
        if (!(inFile >> rows >> cols))
            throw std::runtime_error("Invalid header.");

        int N = rows * cols;
        std::vector<float> A(N), B(N), C(N);

        readMatrix(inFile, A, rows, cols);
        readMatrix(inFile, B, rows, cols);

        // --- CPU Computation ---
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < N; ++i)
        {
            C[i] = A[i] + B[i];
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        // Print time to cerr so we can capture it in benchmark script
        std::cerr << duration.count() << std::endl;

        // Output handling
        if (argc >= 3)
        {
            std::ofstream outFile(argv[2]);
            writeMatrix(outFile, C, rows, cols);
        }
        else
        {
            writeMatrix(std::cout, C, rows, cols);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}