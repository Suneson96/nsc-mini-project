import numpy as np
from dask import array as da


def _mandelbrot_block(c_block: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    z = np.zeros_like(c_block)
    mandelbrot = np.zeros(c_block.shape, dtype=int)
    for n in range(max_iterations):
        mask = np.abs(z) <= 2.0
        z[mask] = z[mask] * z[mask] + c_block[mask]
        mandelbrot[mask] = n
    return mandelbrot


def generate_mandelbrot_set_dask_local(x_min: float, x_max: float, y_min: float, y_max: float,
                                width: int, height: int, max_iterations: int, chunk_size: int) -> np.ndarray:

    # Build the complex number grid using NumPy meshgrid.
    real = np.linspace(x_min, x_max, width)
    imag = np.linspace(y_min, y_max, height)
    R, I = np.meshgrid(real, imag)
    c = R + 1j * I

    # Convert to a chunked dask array and apply the per-block computation via map_blocks.
    c_dask = da.from_array(c, chunks=(chunk_size, chunk_size))
    result = c_dask.map_blocks(_mandelbrot_block, dtype=int, max_iterations=max_iterations)

    return result.compute()

if __name__ == "__main__":
    from visualization import plot_mandelbrot_set, visualize_dask_local_measurements
    from measurements import save_measurements, load_measurements
    import timeit

    # Define parameters for the Mandelbrot set generation.
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    width, height = 1024, 1024
    max_iterations = 100

    # Measure the time taken to generate the Mandelbrot set using the Dask local approach.
    chunk_sizes = [32 ,64, 128, 256, 512, 1024]  # Test with different chunk sizes.
    for chunk_size in chunk_sizes:
        elapsed = timeit.timeit(lambda: generate_mandelbrot_set_dask_local(x_min, x_max, y_min, y_max,
                                                          width, height, max_iterations, chunk_size), number=5) / 5
        print(f"Average time taken to generate Mandelbrot set (Dask local) with chunk size {chunk_size}: {elapsed:.2f} seconds")

        # Save the measurement to a CSV file for later analysis.
        save_measurements("dask_local", elapsed, chunk_size=chunk_size, processes=None)

    # Load the measurements and visualize the results.
    measurements = load_measurements()
    visualize_dask_local_measurements(measurements)

    # Visualize the Mandelbrot set.
    mandelbrot_set = generate_mandelbrot_set_dask_local(x_min, x_max, y_min, y_max,
                                                      width, height, max_iterations, chunk_size)
    plot_mandelbrot_set(mandelbrot_set, x_min, x_max, y_min, y_max, title="Mandelbrot Set - Dask Local Approach")