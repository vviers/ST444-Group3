if __name__ == "__main__":
    
    # seed the script
    from random import seed
    seed(20190323)
    # import test function
    from functions import *
    # import PSO solvers
    from PSO import PSO, PSO_synchron, PSO_asynchron
    
    
    print("A simple quadratic function.\n__________________\n")

    solver_basic = PSO(num_particles = 30, function = quad_function, n_iter = 500, ndim = 2,
                 lower = -10, upper = 10, epsilon = 10e-10)

    solver_basic.run()
    
    print("\n\nThe sphere function.\n__________________\n")

    solver_basic = PSO(num_particles = 30, function = sphere, n_iter = 500, ndim = 30,
                 lower = -100, upper = 100, epsilon = 10e-10)

    solver_basic.run()
    
    print("\n\nThe Rosenbrock function.\n__________________\n")

    solver_basic = PSO(num_particles = 30, function = high_dim_rosenbrock, n_iter = 500, ndim = 30,
                 lower = -30, upper = 30, epsilon = 10e-10)
    
    solver_basic.run()
    
    print("\n\nThe Griewank function.\n__________________\n")

    solver_basic = PSO(num_particles = 30, function = griewank, n_iter = 500, ndim = 30,
                 lower = -600, upper = 600, epsilon = 10e-10)
    
    solver_basic.run()
    
    print("\n\nThe Rastrigrin function.\n__________________\n")

    solver_basic = PSO(num_particles = 30, function = rastrigin, n_iter = 500, ndim = 30,
                 lower = -5.12, upper = 5.12, epsilon = 10e-10)
    
    solver_basic.run()
    
    print("\n\nThe Schaffer F6 function.\n__________________\n")

    solver_basic = PSO(num_particles = 30, function = schaffer_f6, n_iter = 500, ndim = 2,
                 lower = -100, upper = 100, epsilon = 10e-10)
    
    solver_basic.run()