# plot.py

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("results.csv")

cpu = data[data["type"] == "cpu"]
gpu = data[data["type"] == "gpu"]

plt.figure()

plt.plot(cpu["iteration"], cpu["latency_ns"], label="CPU Only")
plt.plot(gpu["iteration"], gpu["latency_ns"], label="GPU -> CPU (Migration)")

plt.xlabel("Iteration")
plt.ylabel("Latency (ns)")
plt.title("Unified Memory Page Migration Latency")
plt.legend()

plt.grid()

plt.savefig("latency_plot.png")
plt.show()