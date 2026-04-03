import matplotlib.pyplot as plt
import os

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