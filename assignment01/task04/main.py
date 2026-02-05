import os
import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np

# Define Paths
CPU_EXEC = "../task02/matrix_add_cpu"
GPU_EXEC = "../task03/matrix_add_gpu"
INPUT_FILE = "input.txt"
OUTPUT_FILE = "output.txt"

sizes = [128, 512, 1024, 2048, 4096, 8192]

cpu_times = []
gpu_times = []


def generate_input(n, filename):
    total = n * n
    print(f"  - Generating {n}x{n} data ({total/1e6:.1f} M elements)...")
    data_a = np.random.rand(total).astype(np.float32)
    data_b = np.random.rand(total).astype(np.float32)

    with open(filename, 'w') as f:
        f.write(f"{n} {n}\n")
        np.savetxt(f, data_a.reshape(1, -1), fmt='%.2f')
        np.savetxt(f, data_b.reshape(1, -1), fmt='%.2f')


print("Starting Benchmark...")

for n in sizes:
    print(f"Benchmarking Size: {n}x{n} ...")
    generate_input(n, INPUT_FILE)

    # --- Run CPU ---
    print("  - Running CPU...")
    result_cpu = subprocess.run(
        [CPU_EXEC, INPUT_FILE, OUTPUT_FILE], capture_output=True, text=True)
    try:
        t_cpu = float(result_cpu.stderr.strip())
        cpu_times.append(t_cpu)
    except ValueError:
        cpu_times.append(0)

    # --- Run GPU ---
    print("  - Running GPU...")
    result_gpu = subprocess.run(
        [GPU_EXEC, INPUT_FILE, OUTPUT_FILE], capture_output=True, text=True)
    try:
        t_gpu = float(result_gpu.stderr.strip())
        gpu_times.append(t_gpu)
    except ValueError:
        gpu_times.append(0)

# Clean up
if os.path.exists(INPUT_FILE):
    os.remove(INPUT_FILE)
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(sizes, cpu_times, label='CPU Time', marker='o')
plt.plot(sizes, gpu_times, label='GPU Time (incl. transfer)', marker='x')
plt.xlabel('Matrix Dimension (N)')
plt.ylabel('Time (ms)')
plt.title('CPU vs GPU Performance')
plt.legend()
plt.grid(True)
plt.savefig('benchmark_plot.png')
