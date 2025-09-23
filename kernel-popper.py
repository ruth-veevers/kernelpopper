import numpy as np
from tqdm import tqdm
from math import sqrt
from sklearn.kernel_ridge import KernelRidge
from typing import Optional, List


def get_quadratic_weights(model: KernelRidge, training_data: np.array,
                          feature_names: Optional[List[str]] = None) -> (dict, np.array):
    """
        Takes a scikit-learn KernelRidge regression model (fit with a polynomial kernel with degree 2)
        and computes primary (feature)
        weights from the dual (sample) weights.

        :param model: a trained KernelRidge regression model with a polynomial kernel (degree 2)
        :param training_data: an array containing the data on which the model was trained (the x argument
            passed to model.fit())
        :param feature_names: Optional parameter - if blank, the original features (columns in training_data) will be
            referred to as 'f0', 'f1', 'f2', etc... Otherwise, will use apply names from this list to the columns in order
        :return: (weight_values, new_predictions), where:
            weight_values is a dictionary of weights where the keys are descriptions of the expanded polynomial features
            and the values are the calculated weights
            new_predictions calculates a prediction for each sample from the fully enumerated polynomial features
            and the calculated feature weights (for testing)
        """
    # check input
    if feature_names is not None:
        assert len(feature_names) == training_data.shape[1], f'Length of feature name list ({len(feature_names)})' \
                                                             f' does not match columns in training' \
                                                             f' data ({training_data.shape[1]})'
    assert model.kernel == 'poly' and model.degree == 2, "Model does not have a polynomial kernel with degree 2"
    # enumerate polynomial features
    # store some square roots so we don't have to keep recalculating
    gamma = model.gamma
    coef0 = model.coef0
    # calculate quadratic expansion of training_data
    columns = [[coef0] * training_data.shape[0]]
    polynomial_feature_names = ['coef0']
    n_features = training_data.shape[1]
    if feature_names is None:
        feature_names = [f'f{i}' for i in range(n_features)]
    for i in tqdm(range(n_features)):
        columns.append(training_data[:, i] ** 2 * gamma)
        polynomial_feature_names.append(f'{feature_names[i]}**2 * gamma')
        columns.append(sqrt(2 * coef0 * gamma) * training_data[:, i])
        polynomial_feature_names.append(f'sqrt(2 * coef0 * gamma) * {feature_names[i]}')
        for j in range(i + 1, n_features):
            columns.append(sqrt(2) * training_data[:, i] * training_data[:, j] * gamma)
            polynomial_feature_names.append(f'sqrt(2) * {feature_names[i]} * {feature_names[j]} * gamma')
    polynomial_x = np.transpose(columns)
    # . product with dual weights
    poly_weights = np.matmul(model.dual_coef_, polynomial_x)
    weight_values = {polynomial_feature_names[i]: poly_weights[i] for i in range(n_features)}
    new_predictions = np.matmul(polynomial_x, poly_weights)
    return weight_values, new_predictions


def get_cubic_weights(model: KernelRidge, training_data: np.array,
                      feature_names: Optional[List[str]] = None) -> (dict, np.array):
    """
        Takes a scikit-learn KernelRidge regression model (fit with a polynomial kernel with degree 3)
        and computes primary (feature)
        weights from the dual (sample) weights.

        :param model: a trained KernelRidge regression model with a polynomial kernel (degree 3)
        :param training_data: an array containing the data on which the model was trained (the x argument
            passed to model.fit())
        :param feature_names: Optional parameter - if blank, the original features (columns in training_data) will be
            referred to as 'f0', 'f1', 'f2', etc... Otherwise, will use apply names from this list to the columns in order
        :return: (weight_values, new_predictions), where:
            weight_values is a dictionary of weights where the keys are descriptions of the expanded polynomial features
            and the values are the calculated weights
            new_predictions calculates a prediction for each sample from the fully enumerated polynomial features
            and the calculated feature weights (for testing)
        """
    # enumerate polynomial features
    # store some square roots, so we don't have to keep recalculating
    gamma = model.gamma
    coef0 = model.coef0
    # calculate quadratic expansion of training_data
    columns = [[sqrt(coef0**3)] * training_data.shape[0]]
    polynomial_feature_names = ['sqrt(coef0**3)']
    n_features = training_data.shape[1]
    assert model.kernel == 'poly' and model.degree == 3, "Model does not have a polynomial kernel with degree 3"
    if feature_names is None:
        feature_names = [f'f{i}' for i in range(n_features)]
    else:
        assert len(feature_names) == training_data.shape[1], f'Length of feature name list ({len(feature_names)})' \
                                                      f' does not match columns in training' \
                                                      f' data ({training_data.shape[1]})'
    for i in tqdm(range(n_features)):
        columns.append(training_data[:, i] ** 3 * sqrt(gamma**3))
        polynomial_feature_names.append(f'{feature_names[i]}**3 * sqrt(gamma**3)')
        columns.append(sqrt(3) * training_data[:, i] ** 2 * gamma * sqrt(coef0))
        polynomial_feature_names.append(f'sqrt(3) * {feature_names[i]}**2 * gamma * sqrt(coef0)')
        columns.append(sqrt(3) * training_data[:, i] * sqrt(gamma) * coef0)
        polynomial_feature_names.append(f'sqrt(3) * {feature_names[i]} * sqrt(gamma) * coef0')
        for j in range(i + 1, n_features):
            columns.append(sqrt(6) * training_data[:, i] * training_data[:, j] * gamma * sqrt(coef0))
            polynomial_feature_names.append(f'sqrt(6) * {feature_names[i]} * {feature_names[j]} * gamma * sqrt(coef0)')
            columns.append(sqrt(3) * training_data[:, i] ** 2 * training_data[:, j] * sqrt(gamma**3))
            polynomial_feature_names.append(f'sqrt(3) * {feature_names[i]}**2 * '
                                            f'{feature_names[j]} * sqrt(gamma**3)')
            columns.append(sqrt(3) * training_data[:, i] * training_data[:, j] ** 2 * sqrt(gamma**3))
            polynomial_feature_names.append(f'sqrt(3) * {feature_names[i]} * {feature_names[j]}**2 '
                                            f'* sqrt(gamma**3)')
            for k in range(j + 1, n_features):
                columns.append(sqrt(6) * training_data[:, i] * training_data[:, j]
                               * training_data[:, k] * sqrt(gamma**3))
                polynomial_feature_names.append(f'sqrt(6) * {feature_names[i]} * {feature_names[j]}'
                                                f' * {feature_names[k]} * sqrt(gamma**3)')
    polynomial_x = np.transpose(columns)
    # . product with dual weights
    poly_weights = np.matmul(model.dual_coef_, polynomial_x)
    weight_values = {polynomial_feature_names[i]: poly_weights[i] for i in range(n_features)}
    new_predictions = np.matmul(polynomial_x, poly_weights)
    return weight_values, new_predictions


def test_quadratic_model() -> None:
    """
    Create synthetic data, train a model with polynomial kernel (degree 2) and check that the
    enumerated polynomial features with calculated weights give the same predictions (within small tolerance)
    as the fit model

    :return: nothing
    """
    # synthetic data
    x = np.random.rand(10000, 10)
    gamma = 0.1
    coef0 = 2
    # generate data w/ non-linear relationship between x and y
    y = x[:, 0] + (2 * x[:, 1] ** 2) + 5 * x[:, 4] * x[:, 6]
    # fit a model with polynomial kernel
    model = KernelRidge(gamma=gamma, alpha=0.1, kernel='poly', degree=2, coef0=coef0)
    model.fit(x, y)
    original_predictions = model.predict(x)
    # compute the primary weight vector
    weight_values, new_predictions = get_quadratic_weights(model, x)
    # compare with original predictions from the model, tolerating a small difference from calculations
    assert np.max(np.abs(original_predictions - new_predictions)) < 1e-9


def test_cubic_model() -> None:
    """
    Create synthetic data, train a model with polynomial kernel (degree 3) and check that the
    enumerated polynomial features with calculated weights give the same predictions (within small tolerance)
    as the fit model

    :return: nothing
    """
    gamma = 0.5
    coef0 = 2
    # create data with non-linear relationship between x and y
    x = np.random.rand(10000, 10)
    y = x[:, 0] + (2 * x[:, 1] ** 2 - x[:, 2]) + (x[:, 0] * -3 * x[:, 1]) - x[:, 3] ** 2 * x[:, 8] * x[:, 9]
    model = KernelRidge(kernel='poly', degree=3, gamma=gamma, coef0=coef0)
    model.fit(x, y)
    original_predictions = model.predict(x)
    # compute the primary weight vector
    _, new_predictions = get_cubic_weights(model, x)
    # compare with original predictions from the model, tolerating a small difference from calculations
    assert np.max(np.abs(original_predictions - new_predictions)) < 1e-9


def run_tests():

    test_quadratic_model()
    test_cubic_model()
