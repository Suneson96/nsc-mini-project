import multiprocessing as mp
import numpy as np

from vectorized import generate_mandelbrot_set_vectorized

def generate_mandelbrot_set_parallel(x_min: float, x_max: float, y_min: float, y_max: float,
                                     width: int, height: int, max_iterations:int,
                                     processes: int, chunk_size: int) -> np.ndarray:
    
    # Create a 2D array to hold the results.
    mandelbrot_set = np.zeros((height, width), dtype=int)

    # Create a list of tasks for each chunk.
    # Each task will compute a portion of the Mandelbrot set.
    tasks = []
    chunks = []
    for y_start in range(0, height, chunk_size):
        y_end = min(y_start + chunk_size, height)
        for x_start in range(0, width, chunk_size):
            x_end = min(x_start + chunk_size, width)
            chunk_x_min = x_min + (x_start / width) * (x_max - x_min)
            chunk_x_max = x_min + (x_end / width) * (x_max - x_min)
            chunk_y_min = y_min + (y_start / height) * (y_max - y_min)
            chunk_y_max = y_min + (y_end / height) * (y_max - y_min)
            tasks.append((chunk_x_min, chunk_x_max, chunk_y_min, chunk_y_max,
                          x_end - x_start, y_end - y_start, max_iterations))
            chunks.append((y_start, y_end, x_start, x_end))

    # Use multiprocessing to compute the Mandelbrot set in parallel.
    with mp.Pool(processes=processes) as pool:
        results = [pool.apply_async(generate_mandelbrot_set_vectorized, task) for task in tasks]

        # Gather results and combine the corresponding chunks in the final Mandelbrot set array.
        for result, (y_start, y_end, x_start, x_end) in zip(results, chunks):
            mandelbrot_set[y_start:y_end, x_start:x_end] = result.get()
    return mandelbrot_set

if __name__ == "__main__":

    from visualization import plot_mandelbrot_set, visualize_parallel_measurements, visualize_parallel_speedup
    from measurements import save_measurements, load_measurements
    import timeit

    # Define parameters for the Mandelbrot set generation.
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    width, height = 1024, 1024
    max_iterations = 100
    processeses = [2, 3, 4]  # Test with different numbers of processes. :)
    chunk_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]  # Test with different chunk sizes.

    for processes in processeses:
        for chunk_size in chunk_sizes:

            average = timeit.timeit(lambda: generate_mandelbrot_set_parallel(x_min, x_max, y_min, y_max,
                                                                width, height, max_iterations,
                                                                processes, chunk_size), number=5) / 5
            print(f"Average time taken to generate Mandelbrot set (parallel with {processes} processes and chunk size {chunk_size}): {average:.2f} seconds")

            # Save the measurement to a CSV file for later analysis.
            save_measurements("parallel", average, chunk_size, processes)

    # Visualize the Mandelbrot set.
    mandelbrot_set = generate_mandelbrot_set_parallel(x_min, x_max, y_min, y_max,
                                                      width, height, max_iterations,
                                                      processes, chunk_size)
    plot_mandelbrot_set(mandelbrot_set, x_min, x_max, y_min, y_max, title="Mandelbrot Set - Parallel Approach")

    # Load the measurements from the CSV file.
    measurements = load_measurements()

    # Visualize the measurements for the parallel approach.
    visualize_parallel_measurements(measurements)
    visualize_parallel_speedup(measurements)