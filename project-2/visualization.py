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

def visualize_dask_local_measurements(measurements: pd.DataFrame):
    # Filter measurements for the Dask local approach.
    dask_local_measurements = measurements[measurements["approach"] == "dask_local"]

    plt.figure(figsize=(10, 6))
    plt.plot(dask_local_measurements["chunk_size"], dask_local_measurements["time_seconds"], marker="o")
    plt.title("Time taken for Dask Local Mandelbrot Set Generation")
    plt.xlabel("Chunk Size")
    plt.ylabel("Time (seconds)")

    # Save the plot to a file.
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/dask_local_measurements.png")
    plt.close()

def visualize_parallel_speedup(measurements: pd.DataFrame):
    # Filter measurements for the parallel and vectorized approaches.
    parallel_measurements = measurements[measurements["approach"] == "parallel"]
    vectorized_measurements = measurements[measurements["approach"] == "vectorized"]

    plt.figure(figsize=(10, 6))
    for processes, group in parallel_measurements.groupby("processes"):
        speedup = vectorized_measurements["time_seconds"].values[0] / group["time_seconds"]
        plt.plot(group["chunk_size"], speedup, marker="o", label=f"{processes} processes")
    plt.title("Speedup of Parallel Approach Compared to Vectorized Approach")
    plt.xlabel("Chunk Size")
    plt.ylabel("Speedup")
    plt.legend()

    # Save the plot to a file.
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/speedup_parallel_vs_vectorized.png")
    plt.close()

def visualize_dask_local_speedup(measurements: pd.DataFrame):
    # Filter measurements for the Dask local and vectorized approaches.
    dask_local_measurements = measurements[measurements["approach"] == "dask_local"]
    vectorized_measurements = measurements[measurements["approach"] == "vectorized"]

    plt.figure(figsize=(10, 6))
    speedup = vectorized_measurements["time_seconds"].values[0] / dask_local_measurements["time_seconds"]
    plt.plot(dask_local_measurements["chunk_size"], speedup, marker="o", label="Dask Local")
    plt.title("Speedup of Dask Local Approach Compared to Vectorized Approach")
    plt.xlabel("Chunk Size")
    plt.ylabel("Speedup")
    plt.legend()

    # Save the plot to a file.
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/speedup_dask_local_vs_vectorized.png")
    plt.close()
