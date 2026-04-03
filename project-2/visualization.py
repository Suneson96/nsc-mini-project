import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_mandelbrot_set(mandelbrot_set, x_min, x_max, y_min, y_max, title="Mandelbrot Set"):
    plt.imshow(mandelbrot_set, extent=(x_min, x_max, y_min, y_max))
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Real Axis")
    plt.ylabel("Imaginary Axis")

    # Save the plot to a file.
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{title.replace(' ', '_').replace('-', '_').lower()}.png")
    plt.close()

def visualize_parallel_measurements(measurements: pd.DataFrame):
    # Filter measurements for the parallel approach.
    parallel_measurements = measurements[measurements["approach"] == "parallel"]

    plt.figure(figsize=(10, 6))
    for processes, group in parallel_measurements.groupby("processes"):
        plt.plot(group["chunk_size"], group["time_seconds"], marker="o", label=f"{processes} processes")
    plt.title("Time taken for Parallel Mandelbrot Set Generation")
    plt.xlabel("Chunk Size")
    plt.ylabel("Time (seconds)")
    plt.legend()

    # Save the plot to a file.
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/parallel_measurements.png")
    plt.close()
