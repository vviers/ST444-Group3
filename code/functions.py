# Test Functions for the PSO algorithm
import numpy as np

def quad_function(li):
    assert type(li) == list or type(li) == np.ndarray, "Argument must be a list of coordinates."
    return [(x[0] + 2*x[1] - 3)**2 + (x[0] - 2)**2 for x in li]

def high_dim_rosenbrock(x):
    return sum([(1-x[i])**2 + 100*(x[i+1] - x[i]**2)**2 for i in range(len(x) - 1)])

def griewank(li):
    assert type(li) == list or type(li) == np.ndarray, "Argument must be a list of coordinates."
    def f(x):
        return 1 + (1/4000) * sum([xi**2 for xi in x]) - np.product([np.cos(x[i]/np.sqrt(i+1))
                                                                     for i in range(len(x))])
    return [f(x) for x in li]

def schaffer_f6(li):
    assert type(li) == list or type(li) == np.ndarray, "Argument must be a list of coordinates."
    def f(x):
        return .5 + ((np.sin(np.sqrt(x[0]**2 + x[1]**2))**2) - .5)/((1 + 0.001*(x[0]**2 + x[1]**2))**2)
    return [f(x) for x in li]

def rastrigin(li):
    assert type(li) == list or type(li) == np.ndarray, "Argument must be a list of coordinates."
    def f(x):
        return 10*len(x) + sum([xi**2 - 10 * np.cos(2*np.pi*xi) for xi in x])
    return [f(x) for x in li]