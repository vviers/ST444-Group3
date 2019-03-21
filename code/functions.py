# Test Functions for the PSO algorithm
import numpy as np

def quad_function(x):
    assert len(x) == 2, "This function expects a two-dimensional input."
    return (x[0] + 2*x[1] - 3)**2 + (x[0] - 2)**2

def sphere(x):
    return sum([xi**2 for xi in x])

def high_dim_rosenbrock(x):
    return sum([(1-x[i])**2 + 100*(x[i+1] - x[i]**2)**2 for i in range(len(x) - 1)])

def griewank(x):
    return 1 + (1/4000) * sum([xi**2 for xi in x]) - np.product([np.cos(x[i]/np.sqrt(i+1))
                                                                     for i in range(len(x))])

def schaffer_f6(x):
    assert len(x) == 2, "This function expects a two-dimensional input."
    return .5 + ((np.sin(np.sqrt(x[0]**2 + x[1]**2))**2) - .5)/((1 + 0.001*(x[0]**2 + x[1]**2))**2)

def rastrigin(x):
    return 10*len(x) + sum([xi**2 - 10 * np.cos(2*np.pi*xi) for xi in x])