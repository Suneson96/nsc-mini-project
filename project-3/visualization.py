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


def plot_per_scale_comparisons(csv_path="csv/measurements.csv"):
    df = pd.read_csv(csv_path)
    df["scale"] = df["scale"].astype(int)
    df["time_seconds"] = df["time_seconds"].astype(float)

    scales = sorted(df["scale"].unique())
    approaches = sorted(df["approach"].unique())
    avgs = df.groupby(["approach", "scale"])["time_seconds"].mean()

    fig, ax = plt.subplots(figsize=(9, 5))

    for approach in approaches:
        times = [avgs[approach, s] for s in scales]
        ax.plot([str(s) for s in scales], times, marker="o", label=approach)

    ax.set_title("Average execution time by scale")
    ax.set_xlabel("Scale")
    ax.set_ylabel("Average time (seconds)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig("plots/scaling_comparison.png")
    plt.close(fig)
    print("Saved plots/scaling_comparison.png")


def plot_speedup(csv_path="csv/measurements.csv"):
    df = pd.read_csv(csv_path)
    df["scale"] = df["scale"].astype(int)
    df["time_seconds"] = df["time_seconds"].astype(float)

    avgs = df.groupby(["approach", "scale"])["time_seconds"].mean()
    scales = sorted(df["scale"].unique())
    approaches = sorted(a for a in df["approach"].unique() if a != "naive")

    fig, ax = plt.subplots(figsize=(9, 5))

    for approach in approaches:
        speedups = [avgs["naive", s] / avgs[approach, s] for s in scales]
        ax.plot([str(s) for s in scales], speedups, marker="o", label=approach)

    ax.axhline(1, color="gray", linestyle="--", linewidth=0.8, label="naive (reference)")
    ax.set_title("Speedup relative to naive approach")
    ax.set_xlabel("Scale")
    ax.set_ylabel("Speedup (×)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig("plots/speedup.png")
    plt.close(fig)
    print("Saved plots/speedup.png")


def print_stats_table(csv_path="csv/measurements.csv"):
    df = pd.read_csv(csv_path)
    df["scale"] = df["scale"].astype(int)
    df["time_seconds"] = df["time_seconds"].astype(float)

    table = (
        df.groupby(["approach", "scale"])["time_seconds"]
        .agg(fastest="min", slowest="max", average="mean")
        .round(6)
    )
    print("\nExecution time statistics (seconds):")
    print(table.to_string())


if __name__ == "__main__":
    plot_per_scale_comparisons()
    plot_speedup()
    print_stats_table()

