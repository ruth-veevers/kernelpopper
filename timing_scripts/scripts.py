import kernelpopper
import numpy as np
from time import time, perf_counter
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sys import argv
from sklearn.datasets import make_regression, make_friedman1

def time_quadratic(n_features: int, seed: int) -> None:
	n_samples=1000
	np.random.seed(seed)
	x,y=make_friedman1(n_samples, n_features, random_state=seed, noise=0.5)
	# quadratic (krr)
	model=KernelRidge(kernel='poly', degree=2)
	start=perf_counter()
	model.fit(x,y)
	time_quadratic=perf_counter()-start
	# linear (quadratic features)
	_, _, poly_x= kernelpopper.get_quadratic_weights(model)
	model_linear=LinearRegression()
	start=perf_counter()
	model_linear.fit(poly_x, y)
	time_linear_quadratic=perf_counter()-start
	with open(f'result_quadratic_{n_features}_{seed}.txt','w') as resultfile:
		resultfile.write(f'n_features:{n_features}\n')
		resultfile.write(f'seed:{seed}\n')
		resultfile.write(f'time_quadratic:{time_quadratic}\n')
		resultfile.write(f'time_linear_quadratic:{time_linear_quadratic}')
	
def time_cubic(n_features: int, seed: int) -> None:
	n_samples=1000
	np.random.seed(seed)
	x,y=make_friedman1(n_samples, n_features, random_state=seed, noise=0.5)
	# cubic (krr)
	model=KernelRidge(kernel='poly', degree=3)
	start=perf_counter()
	model.fit(x,y)
	time_cubic=perf_counter()-start
	# linear (cubic features)
	_, _, poly_x= kernelpopper.get_cubic_weights(model)
	model_linear=LinearRegression()
	start=perf_counter()
	model_linear.fit(poly_x, y)
	time_linear_cubic=perf_counter()-start
	with open(f'result_cubic_{n_features}_{seed}.txt','w') as resultfile:
		resultfile.write(f'n_features:{n_features}\n')
		resultfile.write(f'seed:{seed}\n')
		resultfile.write(f'time_cubic:{time_cubic}\n')
		resultfile.write(f'time_linear_cubic:{time_linear_cubic}')
	
def build_report():
	pass
	
if __name__ == '__main__':
	if len(argv) < 4:
		raise ValueError('Not enough arguments')
	if argv[2].isdigit():
		n_features=int(argv[2])
	else:
		raise ValueError('Second argument should be an integer (number of features)')
	if argv[3].isdigit():
		seed=int(argv[3])
	else:
		raise ValueError('Third argument should be an integer (seed)')
	if argv[1].lower()=='q':
		time_quadratic(n_features, seed)
	elif argv[1].lower()=='c':
		time_cubic(n_features, seed)
	else:
		raise ValueError('First argument should either be "q" (quadratic) or "c" (cubic)')