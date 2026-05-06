import numpy as np

def generate_mandelbrot_set_naive(x_min: float, x_max: float, y_min: float, y_max: float,
                                  width: int, height: int, max_iterations: int) -> list:
    """
    Generate the Mandelbrot set using a naive approach with nested loops.
    
    Args:
        x_min (float): The minimum x-coordinate of the complex plane.
        x_max (float): The maximum x-coordinate of the complex plane.
        y_min (float): The minimum y-coordinate of the complex plane.
        y_max (float): The maximum y-coordinate of the complex plane.
        width (int): The width of the output array (image).
        height (int): The height of the output array (image).
        max_iterations (int): The maximum number of iterations to perform.

    Returns:
        list: A python native 2D list representing the Mandelbrot set, where each element is the number of iterations before divergence.
    """

    def mandelbrot(c: complex, max_iterations: int) -> int:
        """
        Calculate the number of iterations for a given complex number to determine if it belongs to the Mandelbrot set.

        Args:
            c (complex): The complex number to test.
            max_iterations (int): The maximum number of iterations to perform.

        Returns:
            int: The number of iterations before the magnitude of z exceeds 2, or max_iterations if it doesn't.
        """
        z = 0
        n = 0

        # As long as the magnitude of z is less than or equal to 2 (hasn't grown too large) and we haven't reached the maximum number of iterations.
        while abs(z) <= 2.0 and n < max_iterations:
            z = z*z + c
            n += 1
        return n

    # Create a 2D list to hold the Mandelbrot set values.
    mandelbrot_set = [[0 for _ in range(width)] for _ in range(height)]

    for y in range(height):
        for x in range(width):
            real = x_min + (x / width) * (x_max - x_min)
            imag = y_min + (y / height) * (y_max - y_min)
            c = complex(real, imag)
            mandelbrot_set[y][x] = mandelbrot(c, max_iterations)

    return mandelbrot_set


def generate_mandelbrot_set_vectorized(x_min: float, x_max: float, y_min: float, y_max: float,
                                       width: int, height: int, max_iterations: int) -> np.ndarray:

    """
    Generate the Mandelbrot set using a vectorized approach with NumPy.

    Args:
        x_min (float): The minimum x-coordinate of the complex plane.
        x_max (float): The maximum x-coordinate of the complex plane.
        y_min (float): The minimum y-coordinate of the complex plane.
        y_max (float): The maximum y-coordinate of the complex plane.
        width (int): The width of the output array (image).
        height (int): The height of the output array (image).
        max_iterations (int): The maximum number of iterations to perform.

    Returns:
        np.ndarray: A 2D NumPy array representing the Mandelbrot set, where each element is the number of iterations before divergence.
    """

    # Create a 2D array of complex numbers representing the points in the complex plane.
    real = np.linspace(x_min, x_max, width, endpoint=False)
    imag = np.linspace(y_min, y_max, height, endpoint=False)

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
        mandelbrot_set[mask] = n + 1  # Update the iteration count for those points.

    # Return iteration counts for each complex number.
    return mandelbrot_set


if __name__ == "__main__":
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