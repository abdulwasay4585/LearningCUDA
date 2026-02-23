import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('metrics.csv')

plt.figure(figsize=(10, 6))
plt.plot(df['Size'], df['CPU_Time_ms'], marker='o', label='CPU')
plt.plot(df['Size'], df['GPU_Naive_Time_ms'], marker='s', label='GPU (Naive)')

plt.title('Matrix Multiplication Performance: CPU vs GPU (Naive)')
plt.xlabel('Matrix Size (N x N)')
plt.ylabel('Time (ms)')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig('performance_plot.png')
print("Graph saved as performance_plot.png")
