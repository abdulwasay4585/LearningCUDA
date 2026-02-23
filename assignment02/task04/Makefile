NVCC=nvcc
CFLAGS=-O3
TARGET=task04

all: $(TARGET)

$(TARGET): main.cu
	$(NVCC) $(CFLAGS) -o $(TARGET) main.cu

clean:
	rm -f $(TARGET) metrics.csv performance_plot.png
