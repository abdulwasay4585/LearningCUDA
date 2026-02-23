import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('metrics_task5.csv')

plt.figure(figsize=(10, 6))
plt.plot(df['Size'], df['CPU_Time_ms'], marker='o', label='CPU')
plt.plot(df['Size'], df['GPU_Naive_Time_ms'], marker='s', label='GPU (Naive)')
plt.plot(df['Size'], df['GPU_Tiled_Time_ms'], marker='^', label='GPU (Tiled)')

plt.title('Matrix Multiplication Performance: CPU vs GPU (Naive) vs GPU (Tiled)')
plt.xlabel('Matrix Size (N x N)')
plt.ylabel('Time (ms)')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig('performance_plot_task5.png')
print("Graph saved as performance_plot_task5.png")
