from math import sqrt

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from tqdm import tqdm


def get_quadratic_weights(model: KernelRidge, feature_names: list[str] | None = None) -> (dict, np.array, np.array):
    """Calculate feature weights from a polynomial kernel with degree 2.

    Takes a scikit-learn KernelRidge regression model (fit with a polynomial kernel with degree 2)
    and computes primary (feature) weights from the dual (sample) weights.

    :param model: a trained KernelRidge regression model with a polynomial kernel (degree 2)
    :param feature_names: Optional parameter - if blank, the original features (columns in training_data) will be
        referred to as 'f0', 'f1', 'f2', etc... Otherwise, will use apply names from this list to the columns in order
    :return: (weight_values, new_predictions), where:
        weight_values is a dictionary of weights where the keys are descriptions of the expanded polynomial features
        and the values are the calculated weights
        new_predictions calculates a prediction for each sample from the fully enumerated polynomial features
        and the calculated feature weights (for testing)
        polynomial_x is the samples from the original x matrix mapped to the implicit feature space.
    """
    # check input
    training_data = model.X_fit_
    if feature_names is not None:
        assert len(feature_names) == training_data.shape[1], (f"Length of feature name list ({len(feature_names)})",
                                                             " does not match columns in training",
                                                             f" data ({training_data.shape[1]})")
    assert model.kernel == "poly", "Model does not have a polynomial kernel."
    assert model.degree == 2, "Model degree is not 2."
    # enumerate polynomial features
    # store some square roots, so we don't have to keep recalculating
    gamma = model.gamma
    coef0 = model.coef0
    # calculate quadratic expansion of training_data
    columns = [[coef0] * training_data.shape[0]]
    polynomial_feature_names = ["coef0"]
    n_features = training_data.shape[1]
    if gamma is None:
        gamma = 1 / n_features
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]
    for i in tqdm(range(n_features)):
        columns.append(training_data[:, i] ** 2 * gamma)
        polynomial_feature_names.append(f"{feature_names[i]}**2 * gamma")
        columns.append(sqrt(2 * coef0 * gamma) * training_data[:, i])
        polynomial_feature_names.append(f"sqrt(2 * coef0 * gamma) * {feature_names[i]}")
        for j in range(i + 1, n_features):
            columns.append(sqrt(2) * training_data[:, i] * training_data[:, j] * gamma)
            polynomial_feature_names.append(f"sqrt(2) * {feature_names[i]} * {feature_names[j]} * gamma")
    polynomial_x = np.transpose(columns)
    # . product with dual weights
    poly_weights = np.matmul(model.dual_coef_, polynomial_x)
    weight_values = {polynomial_feature_names[i]: poly_weights[i] for i in range(len(polynomial_feature_names))}
    new_predictions = np.matmul(polynomial_x, poly_weights)
    return weight_values, new_predictions, polynomial_x


def get_cubic_weights(model: KernelRidge, feature_names: list[str] | None = None) -> (dict, np.array, np.array):
    """Calculate feature weights from a polynomial kernel with degree 3.

    Takes a scikit-learn KernelRidge regression model (fit with a polynomial kernel with degree 3)
    and computes primary (feature)
    weights from the dual (sample) weights.

    :param model: a trained KernelRidge regression model with a polynomial kernel (degree 3)
    :param feature_names: Optional parameter - if blank, the original features (columns in training_data) will be
        referred to as 'f0', 'f1', 'f2', etc... Otherwise, will use apply names from this list to the columns in order
    :return: (weight_values, new_predictions, polynomial_x), where:
        weight_values is a dictionary of weights where the keys are descriptions of the expanded polynomial features
        and the values are the calculated weights
        new_predictions calculates a prediction for each sample from the fully enumerated polynomial features
        and the calculated feature weights (for testing)
        polynomial_x is the samples from the original x matrix mapped to the implicit feature space.
    """
    # enumerate polynomial features
    # store some square roots, so we don't have to keep recalculating
    gamma = model.gamma
    coef0 = model.coef0
    training_data = model.X_fit_
    # calculate quadratic expansion of training_data
    columns = [[sqrt(coef0 ** 3)] * training_data.shape[0]]
    polynomial_feature_names = ["sqrt(coef0**3)"]
    n_features = training_data.shape[1]
    if gamma is None:
        gamma = 1 / n_features
    assert model.kernel == "poly", "Model does not have a polynomial kernel"
    assert model.degree == 3, "Model does not have a kernel with degree 3"
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]
    else:
        assert len(feature_names) == training_data.shape[1], (f"Length of feature name list ({len(feature_names)})",
                                                             " does not match columns in training",
                                                             f" data ({training_data.shape[1]})")
    for i in tqdm(range(n_features)):
        columns.append(training_data[:, i] ** 3 * sqrt(gamma ** 3))
        polynomial_feature_names.append(f"{feature_names[i]}**3 * sqrt(gamma**3)")
        columns.append(sqrt(3) * training_data[:, i] ** 2 * gamma * sqrt(coef0))
        polynomial_feature_names.append(f"sqrt(3) * {feature_names[i]}**2 * gamma * sqrt(coef0)")
        columns.append(sqrt(3) * training_data[:, i] * sqrt(gamma) * coef0)
        polynomial_feature_names.append(f"sqrt(3) * {feature_names[i]} * sqrt(gamma) * coef0")
        for j in range(i + 1, n_features):
            columns.append(sqrt(6) * training_data[:, i] * training_data[:, j] * gamma * sqrt(coef0))
            polynomial_feature_names.append(f"sqrt(6) * {feature_names[i]} * {feature_names[j]} * gamma * sqrt(coef0)")
            columns.append(sqrt(3) * training_data[:, i] ** 2 * training_data[:, j] * sqrt(gamma ** 3))
            polynomial_feature_names.append(f"sqrt(3) * {feature_names[i]}**2 * "
                                            f"{feature_names[j]} * sqrt(gamma**3)")
            columns.append(sqrt(3) * training_data[:, i] * training_data[:, j] ** 2 * sqrt(gamma ** 3))
            polynomial_feature_names.append(f"sqrt(3) * {feature_names[i]} * {feature_names[j]}**2 "
                                            f"* sqrt(gamma**3)")
            for k in range(j + 1, n_features):
                columns.append(sqrt(6) * training_data[:, i] * training_data[:, j]
                               * training_data[:, k] * sqrt(gamma ** 3))
                polynomial_feature_names.append(f"sqrt(6) * {feature_names[i]} * {feature_names[j]}"
                                                f" * {feature_names[k]} * sqrt(gamma**3)")
    polynomial_x = np.transpose(columns)
    # . product with dual weights
    poly_weights = np.matmul(model.dual_coef_, polynomial_x)
    weight_values = {polynomial_feature_names[i]: poly_weights[i] for i in range(len(polynomial_feature_names))}
    new_predictions = np.matmul(polynomial_x, poly_weights)
    return weight_values, new_predictions, polynomial_x


