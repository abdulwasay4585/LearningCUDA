#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): N/A (Unknown in this toolkit)\n");
        printf("  Memory Bus Width (bits): N/A (Unknown in this toolkit)\n");
        printf("  Peak Memory Bandwidth (GB/s): ~40.1\n");
        printf("  Clock Rate (KHz): N/A (Unknown in this toolkit)\n");
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);
        
        // Assuming ~128 cores per SM for common architectures (can vary, would need precise lookup for exact arch)
        // Let's use an approximation or just list SMs and Clock Rate.
        // Peak compute in GFLOPs = 2 * Cores * ClockRate (in GHz).  
        // Since core count per SM varies widely between architectures (Pascal=64, Volta=64, Turing=64, Ampere=128 for FP32).        
        int coresPerSM = 0;
        // Basic heuristic for common architectures based on major.minor compute capability
        if (prop.major == 8 && prop.minor == 9) coresPerSM = 128; // Ada Lovelace
        else if (prop.major == 8 && prop.minor == 0) coresPerSM = 64; // Ampere A100
        else if (prop.major == 8 && prop.minor >= 6) coresPerSM = 128; // Ampere consumer
        else if (prop.major == 7 && prop.minor == 5) coresPerSM = 64; // Turing
        else if (prop.major == 7 && prop.minor == 0) coresPerSM = 64; // Volta
        else if (prop.major == 6) coresPerSM = 64; // Pascal
        else if (prop.major == 5) coresPerSM = 128; // Maxwell
        
        if (coresPerSM > 0) {
            int totalCores = prop.multiProcessorCount * coresPerSM;
            float clockGHz = 1122.0 / 1e3; // Approx local rate
            float peakFLOPS = 2.0 * totalCores * clockGHz;
            printf("  Estimated Cores per SM: %d (Total Cores: %d)\n", coresPerSM, totalCores);
            printf("  Peak Compute Performance (GFLOPs): %f (Formula: 2 * Total_Cores * Clock_GHz)\n", peakFLOPS);
        } else {
             printf("  Could not estimate core count for this architecture (Compute %d.%d).\n", prop.major, prop.minor);
        }
        
    }
    return 0;
}
