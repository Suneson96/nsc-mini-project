from numba import cuda
import numpy as np
import time


@cuda.jit
def mandelbrot_kernel(x_min: float, x_max: float, y_min: float, y_max: float,
                      width: int, height: int, max_iterations: int,
                      output: np.ndarray):

    # Get the 2D grid position for the current thread.
    x, y = cuda.grid(2)

    # Check if the thread is within the bounds of the output array.
    if x < width and y < height:
        real = x_min + (x / width) * (x_max - x_min)
        imag = y_min + (y / height) * (y_max - y_min)
        c = complex(real, imag)
        z = complex(0.0, 0.0)
        n = 0
        while abs(z) <= 2.0 and n < max_iterations:
            z = z*z + c
            n += 1
        output[y, x] = n

def generate_mandelbrot_set_cuda(x_min: float, x_max: float, y_min: float, y_max: float,
                                 width: int, height: int, max_iterations: int):

    threadsperblock = (16, 16)
    blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    host_output = np.zeros((height, width), dtype=np.int32)

    # Start time before transferring data to the GPU (host to device).
    t0 = time.perf_counter()
    d_output = cuda.to_device(host_output)
    mandelbrot_kernel[blockspergrid, threadsperblock](
        x_min, x_max, y_min, y_max, width, height, max_iterations, d_output
    )
    output = d_output.copy_to_host()

    # End time after transferring data back to the CPU (device to host).
    elapsed = time.perf_counter() - t0

    return output, elapsed

if __name__ == "__main__":

    # Define parameters for the Mandelbrot set generation.
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    width, height = 1024, 1024
    max_iterations = 100
    num_runs = 5

    # Warm-up run (Not part of the timed runs).
    generate_mandelbrot_set_cuda(x_min, x_max, y_min, y_max, width, height, max_iterations)

    times = []
    for _ in range(num_runs):
        result, elapsed = generate_mandelbrot_set_cuda(
            x_min, x_max, y_min, y_max, width, height, max_iterations
        )
        times.append(elapsed)

    avg_total = sum(times) / num_runs

    print(f" Average time: {avg_total:.4f} s")
