# Compiler Settings
CXX = g++
CXXFLAGS = -Wall -Wextra -Werror -O3 -std=c++17

NVCC = nvcc
# Link against standard C++ library and CUDA Runtime
NVCCFLAGS = -O3 -Wno-deprecated-gpu-targets -I$(CONDA_PREFIX)/include -L$(CONDA_PREFIX)/lib -lcudart

# Targets
all: task02/matrix_add_cpu task03/matrix_add_gpu

task02/matrix_add_cpu: task02/main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

task03/matrix_add_gpu: task03/main.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f task02/matrix_add_cpu task03/matrix_add_gpu