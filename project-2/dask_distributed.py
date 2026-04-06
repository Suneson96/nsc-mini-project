from dask.distributed import Client
from dask import array as da
import numpy as np

def _mandelbrot_block(c_block: np.ndarray, max_iterations: int = 100) -> np.ndarray:

    # The maps_block function always passes a NumPy array to the function.
    z = np.zeros_like(c_block)
    mandelbrot = np.zeros(c_block.shape, dtype=int)
    for n in range(max_iterations):
        mask = np.abs(z) <= 2.0
        z[mask] = z[mask] * z[mask] + c_block[mask]
        mandelbrot[mask] = n
    return mandelbrot

def generate_mandelbrot_set_dask_distributed(x_min: float, x_max: float, y_min: float, y_max: float,
                                             width: int, height: int, max_iterations: int,
                                             ip: str, chunk_size: int) -> np.ndarray:
    
    # Connect to the Dask cluster using the provided IP address.
    client = Client(ip)

    # Build the complex number grid as a native Dask array.
    da_real = da.linspace(x_min, x_max, width, chunks=chunk_size)
    da_imag = da.linspace(y_min, y_max, height, chunks=chunk_size)
    R, I = da.meshgrid(da_real, da_imag)
    c_dask = R + 1j * I
    result = c_dask.map_blocks(_mandelbrot_block, dtype=int, max_iterations=max_iterations)

    # Compute the result, which will execute the tasks on the Dask cluster.
    result = result.compute()

    # Close the Dask client connection after computation is done.
    client.close()
    return result

if __name__ == "__main__":
    from visualization import plot_mandelbrot_set, visualize_dask_distributed_measurements, visualize_dask_distributed_speedup
    from measurements import save_measurements, load_measurements
    import timeit
    import os
    from dotenv import load_dotenv

    # Load the HEAD_NODE_IP and the HEAD_NODE PORT from the .env file.
    load_dotenv()
    HEAD_NODE_IP = os.getenv("HEAD_NODE_IP")
    HEAD_NODE_PORT = os.getenv("HEAD_NODE_PORT", "8786")  # Default is 8786.
    if not HEAD_NODE_IP:
        raise ValueError("HEAD_NODE_IP environment variable not set. Please set it in the .env file.")

    HEAD_NODE_ADDRESS = f"{HEAD_NODE_IP}:{HEAD_NODE_PORT}"

    # Define parameters for the Mandelbrot set generation.
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    width, height = 1024, 1024
    max_iterations = 100
    chunk_sizes = [32, 64, 128, 256, 512, 1024]  # Test with different chunk sizes.

    for chunk_size in chunk_sizes:
        elapsed = timeit.timeit(lambda: generate_mandelbrot_set_dask_distributed(x_min, x_max, y_min, y_max,
                                                          width, height, max_iterations, HEAD_NODE_ADDRESS, chunk_size), number=5) / 5
        print(f"Average time taken to generate Mandelbrot set (Dask distributed) with chunk size {chunk_size}: {elapsed:.2f} seconds")

        # Save the measurement to a CSV file for later analysis.
        save_measurements("dask_distributed", elapsed, chunk_size=chunk_size, processes=None)

    # Load the measurements and visualize the results.
    measurements = load_measurements()
    visualize_dask_distributed_measurements(measurements)
    visualize_dask_distributed_speedup(measurements)

    # Visualize the Mandelbrot set.
    mandelbrot_set = generate_mandelbrot_set_dask_distributed(x_min, x_max, y_min, y_max,
                                                      width, height, max_iterations, HEAD_NODE_ADDRESS, chunk_size)
    plot_mandelbrot_set(mandelbrot_set, x_min, x_max, y_min, y_max, title="Mandelbrot Set - Dask Distributed Approach")
