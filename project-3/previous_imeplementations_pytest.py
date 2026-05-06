from previous_implementations import generate_mandelbrot_set_naive, generate_mandelbrot_set_vectorized
import numpy as np
import pytest

def test_consistency_between_implementations():

    # Test that the outputs from both implementations are consistent for the same parameters.
    x_min, x_max, y_min, y_max = -2.0, 1.0, -1.5, 1.5
    width, height = 100, 100
    max_iterations = 100

    result_vectorized = generate_mandelbrot_set_vectorized(x_min, x_max, y_min, y_max, width, height, max_iterations)
    result_naive = generate_mandelbrot_set_naive(x_min, x_max, y_min, y_max, width, height, max_iterations)

    assert np.array_equal(result_vectorized, np.array(result_naive)), "The outputs from the vectorized and naive implementations do not match."

@pytest.mark.parametrize("width, height", [(10, 10), (20, 20), (50, 50), (100, 100)])
def test_width_and_height_consistency(width, height):

    # Test that the output shape matches the specified width and height.
    x_min, x_max, y_min, y_max = -2.0, 1.0, -1.5, 1.5
    max_iterations = 100

    result = generate_mandelbrot_set_vectorized(x_min, x_max, y_min, y_max, width, height, max_iterations)
    assert result.shape == (height, width), f"Vectorized output shape {result.shape} does not match expected shape ({height}, {width})."

    result_naive = generate_mandelbrot_set_naive(x_min, x_max, y_min, y_max, width, height, max_iterations)
    assert len(result_naive) == height, f"Naive output height {len(result_naive)} does not match expected height {height}."
    assert all(len(row) == width for row in result_naive), f"Naive output width does not match expected width {width}."

    # Test that the outputs from both implementations are consistent for the same parameters.
    assert np.array_equal(result, np.array(result_naive)), "The outputs from the vectorized and naive implementations do not match for the given width and height."
    
@pytest.mark.parametrize("max_iterations", [10, 50, 100, 200])
def test_max_iterations_consistency(max_iterations):

    # Test that the output values are consistent with the specified max_iterations.
    x_min, x_max, y_min, y_max = -2.0, 1.0, -1.5, 1.5
    width, height = 100, 100

    result_vectorized = generate_mandelbrot_set_vectorized(x_min, x_max, y_min, y_max, width, height, max_iterations)
    assert np.all(result_vectorized <= max_iterations), f"Vectorized output contains values greater than max_iterations {max_iterations}."

    result_naive = generate_mandelbrot_set_naive(x_min, x_max, y_min, y_max, width, height, max_iterations)
    assert all(all(value <= max_iterations for value in row) for row in result_naive), f"Naive output contains values greater than max_iterations {max_iterations}."

    # Test that the outputs from both implementations are consistent for the same parameters.
    assert np.array_equal(result_vectorized, np.array(result_naive)), "The outputs from the vectorized and naive implementations do not match for the given max_iterations."
