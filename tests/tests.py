import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import pairwise_kernels
from src.kernelpopper import get_quadratic_weights, get_cubic_weights


def get_quadratic_test_data() -> (np.array, np.array):
    """Create x and y matrices for testing the quadratic method.

    Randomly generate 2D data for x, and then combine the features to create a target variable y.

    :return: (x, y), where:
        x is a 10000 x 10 array of random numbers between 0 and 1,
        y is a 10000 element array calculated from the features of x.
    """
    # synthetic data
    np.random.seed(0)
    x = np.random.rand(10000, 10)
    # generate data w/ non-linear relationship between x and y
    y = x[:, 0] + (2 * x[:, 1] ** 2) + 5 * x[:, 4] * x[:, 6]
    return x, y


def test_quadratic_kernel() -> None:
    """Tests the feature mapping in the get_quadratic_weights function.

    Create synthetic data, train a model with polynomial kernel (degree 2) and compare the
    output of the kernel function with the manually calculated inner products in feature space.

    :return: nothing.
    """
    x, y = get_quadratic_test_data()
    # fit a model with polynomial kernel
    degree = 2
    coef0 = 0.4
    gamma = 1.3
    model = KernelRidge(gamma=gamma, alpha=0.1, kernel="poly", degree=degree, coef0=coef0)
    model.fit(x, y)
    kernel_output = pairwise_kernels(x, x, model.kernel, degree=degree, coef0=coef0, gamma=gamma)
    # compute the primary weight vector
    _, _, phi_x = get_quadratic_weights(model)
    inner_products = np.matmul(phi_x, np.transpose(phi_x))
    tolerance = 1e-9
    assert np.max(np.abs(kernel_output - inner_products)) <tolerance


def test_quadratic_model() -> None:
    """Tests the feature weight calculation in the get_quadratic_weights function.

    Create synthetic data, train a model with polynomial kernel (degree 2) and check that the
    enumerated polynomial features with calculated weights give the same predictions (within small tolerance)
    as the fit model.

    :return: nothing.
    """
    x, y = get_quadratic_test_data()
    # fit a model with polynomial kernel
    model = KernelRidge(gamma=0.1, alpha=0.1, kernel="poly", degree=2, coef0=2)
    model.fit(x, y)
    original_predictions = model.predict(x)
    # compute the primary weight vector
    _, new_predictions, _ = get_quadratic_weights(model)
    # compare with original predictions from the model, tolerating a small difference from calculations
    tolerance = 1e-9
    assert np.max(np.abs(original_predictions - new_predictions)) < tolerance


def test_cubic_model() -> None:
    """Tests the feature weight calculation in the get_cubic_weights function.

    Create synthetic data, train a model with polynomial kernel (degree 3) and check that the
    enumerated polynomial features with calculated weights give the same predictions (within small tolerance)
    as the fit model.

    :return: nothing.
    """
    gamma = 0.5
    coef0 = 2
    # create data with non-linear relationship between x and y
    x = np.random.rand(10000, 10)
    y = x[:, 0] + (2 * x[:, 1] ** 2 - x[:, 2]) + (x[:, 0] * -3 * x[:, 1]) - x[:, 3] ** 2 * x[:, 8] * x[:, 9]
    model = KernelRidge(kernel="poly", degree=3, gamma=gamma, coef0=coef0)
    model.fit(x, y)
    original_predictions = model.predict(x)
    # compute the primary weight vector
    _, new_predictions, _ = get_cubic_weights(model)
    # compare with original predictions from the model, tolerating a small difference from calculations
    tolerance=1e-9
    assert np.max(np.abs(original_predictions - new_predictions)) < tolerance
