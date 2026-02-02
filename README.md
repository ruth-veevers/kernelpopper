<img width="240" height="211" alt="kernel-popper logo" src="https://github.com/user-attachments/assets/ba44d0f2-e6a2-4e64-9f38-3d45ed58a01d"/> 

# KERNELPOPPER

## Background

KernelPopper is a package designed to be compatible with ```scikit-learn```'s ```KernelRidge``` regression models; it calculates primary (feature) weights from the dual (sample) weights learnt by the model. This allows analysis and interpretation of the relationships that the model is using to make predictions. 

A manuscript describing how KernelPopper works is in preparation and will be linked here upon publication.

## Install

The package will shortly be available via `pip`. We will update this section with instructions.

### Requirements

The following dependencies are required to use the package:

```
  - numpy=2.3.3
  - python=3.12
  - scikit-learn=1.7.2
  - tqdm=4.67.1
```

## Usage

First, ```scikit-learn``` should be used to build and train and ```RidgeRegression``` model object. KernelPopper is only compatible with polynomial kernels of degree 2 ("quadratic") and 3 ("cubic").

Example:

```
import kernelpopper
from sklearn.kernel_ridge import KernelRidge
from sklearn.datasets import make_friedman1

# Make a dataset
X, y = makefriedman1(random_state = 0)
# Either build a quadratic model
model = KernelRidge(kernel = 'poly', degree = 2)
# Or, build a cubic model
model = KernelRidge(kernel = 'poly', degree = 3)
# Train the model
model.fit(X, y)
```

### Popping a quadratic kernel

Use the ```get_quadratic_weights``` function. This takes the trained model as a mandatory argument. An optional argument, ```feature_names```, is a list of strings to use as the names of the features in the original training data. If not provided, the features will be named 'f0', 'f1', 'f2', etc. The function returns three objects: a dictionary where the keys describe the expanded polynomial features and the values are the corresponding calculated weights; the predictions as calculated from the expanded polynomial features to compare with the model's original predictions (for testing); the training data mapped into its feature space representation. The feature names will include references to the model's coef0 and gamma hyperparameters, and uses ```sqrt()``` to indicate the square root.

Example:

```
weight_values, new_predictions, _ = kernelpopper.get_quadratic_weights(model)
```

### Popping a cubic kernel

Use the ```get_cubic_weights``` function. This operates the same as the ```get_quadratic_weights``` function, but accepting a model with a cubic kernel.

### Running tests

There are three test functions provided to confirm that KernelPopper is running as expected. All three can be run in series:

```
kernelpopper.run_tests()
```

or separately:

- The ```test_quadratic_kernel``` function creates a synthetic dataset and uses it to train a KRR model with a quadratic kernel. The ```get_quadratic_weights``` function is called to obtain KernelPopper's calcuation of the feature space representation of each sample, from which a matrix of inner products is calculated. This is checked against the output of the kernel function to confirm that ```get_quadratic_weights``` is finding the correct feature space representation.
- The ```test_quadratic_kernel``` function creates a synthetic dataset and uses it to train a KRR model with a quadratic kernel. The ```get_quadratic_weights``` function is called, which returns new predictions for each sample in the synthetic dataset calculated using the expanded polynomial weights. The test function then confirms that these are the same as the original predictions.
- The ```test_cubic_kernel``` function creates a synthetic dataset and uses it to train a KRR model with a cubic kernel. The ```get_cubic_weights``` function is called, which returns new predictions for each sample in the synthetic dataset calculated using the expanded polynomial weights. The test function then confirms that these are the same as the original predictions.
