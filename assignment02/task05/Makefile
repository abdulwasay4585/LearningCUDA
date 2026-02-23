NVCC=nvcc
CFLAGS=-O3
TARGET=task05

all: $(TARGET)

$(TARGET): main.cu
	$(NVCC) $(CFLAGS) -o $(TARGET) main.cu

clean:
	rm -f $(TARGET) metrics_task5.csv performance_plot_task5.png
