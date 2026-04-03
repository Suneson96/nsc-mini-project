import multiprocessing as mp
import numpy as np

from vectorized import generate_mandelbrot_set_vectorized

def generate_mandelbrot_set_parallel(x_min: float, x_max: float, y_min: float, y_max: float,
                                     width: int, height: int, max_iterations:int,
                                     processes: int, chunk_size: int) -> np.ndarray:
    
    # Create a 2D array to hold the results.
    mandelbrot_set = np.zeros((height, width), dtype=int)

    # Create a list of tasks for each chunk of rows.
    # Each task will compute a portion of the Mandelbrot set for a specific range of y values.
    tasks = []
    for y_start in range(0, height, chunk_size):
        y_end = min(y_start + chunk_size, height)
        tasks.append((x_min, x_max, y_min + (y_start / height) * (y_max - y_min),
                      y_min + (y_end / height) * (y_max - y_min), width, y_end - y_start, max_iterations))

    # Create a multiprocessing pool and execute the tasks in parallel.
    with mp.Pool(processes=processes) as pool:
        results = [pool.apply_async(generate_mandelbrot_set_vectorized, args=task) for task in tasks]
        for i, result in enumerate(results):
            y_start = i * chunk_size
            y_end = min(y_start + chunk_size, height)
            mandelbrot_set[y_start:y_end, :] = result.get()

    return mandelbrot_set

if __name__ == "__main__":

    from visualization import plot_mandelbrot_set, visualize_parallel_measurements
    from measurements import save_measurements, load_measurements
    import timeit

    # Define parameters for the Mandelbrot set generation.
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    width, height = 1024, 1024
    max_iterations = 100
    processeses = [2, 3, 4]  # Test with different numbers of processes. :)
    chunk_sizes = range(1, 51)  # Test with different chunk sizes.

    for processes in processeses:
        for chunk_size in chunk_sizes:
            measurements = []  # List to hold measurements.
            for _ in range(10):  # Run each configuration 10 times for better accuracy.

                measurements.append(timeit.timeit(lambda: generate_mandelbrot_set_parallel(x_min, x_max, y_min, y_max,
                                                                width, height, max_iterations,
                                                                processes, chunk_size), number=1))
            # Remove outliers (e.g., times that are more than 2 standard deviations from the mean).
            mean_time = np.mean(measurements)
            std_time = np.std(measurements)
            filtered_measurements = [t for t in measurements if abs(t - mean_time) <= 2 * std_time]
            average = np.mean(filtered_measurements)  # Average time after removing outliers.
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