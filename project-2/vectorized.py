import numpy as np


def generate_mandelbrot_set_vectorized(x_min: float, x_max: float, y_min: float, y_max: float,
                                       width: int, height: int, max_iterations: int) -> np.ndarray:

    # Create a 2D array of complex numbers representing the points in the complex plane.
    real = np.linspace(x_min, x_max, width)
    imag = np.linspace(y_min, y_max, height)

    # Define all the complex numbers in the grid.
    c = np.array([[complex(r, i) for r in real] for i in imag])

    # Initialize an array to hold the number of iterations for each point.
    mandelbrot_set = np.zeros(c.shape, dtype=int)

    # Create an array to hold the current values of z for each point.
    z = np.zeros(c.shape, dtype=complex)

    # Iterate up to max_iterations times, updating all points simultaneously using vectorized operations.
    for n in range(max_iterations):
        mask = np.abs(z) <= 2.0  # Only update points that haven't diverged (magnitude <= 2).
        z[mask] = z[mask] * z[mask] + c[mask]  # Update z for those points.
        mandelbrot_set[mask] = n  # Update the iteration count for those points.

    # Return iteration counts for each complex number.
    return mandelbrot_set


if __name__ == "__main__":
    from visualization import plot_mandelbrot_set
    from measurements import save_measurements
    import timeit

    # Define parameters for the Mandelbrot set generation.
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    width, height = 1024, 1024
    max_iterations = 100

    # Measure the time taken to generate the Mandelbrot set using the vectorized approach.
    # Averaging over 5 runs for better accuracy.
    elapsed = timeit.timeit(lambda: generate_mandelbrot_set_vectorized(x_min, x_max, y_min, y_max,
                                                      width, height, max_iterations), number=5) / 5
    print(f"Average time taken to generate Mandelbrot set (vectorized): {elapsed:.2f} seconds")

    # Save the measurement to a CSV file for later analysis.
    save_measurements("vectorized", elapsed)

    # Visualize the Mandelbrot set.
    mandelbrot_set = generate_mandelbrot_set_vectorized(x_min, x_max, y_min, y_max,
                                                      width, height, max_iterations)
    plot_mandelbrot_set(mandelbrot_set, x_min, x_max, y_min, y_max, title="Mandelbrot Set - Vectorized Approach")