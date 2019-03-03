# Parallel PSO (Synchronous)

import multiprocessing

import numpy as np
import random

# -- global parameters
c1 = 1 # cognitive parameter
c2 = 2 # social parameter
w = .5 # inertia

# -- Particle Class

class Particle:
    """A Python Class for a simple particle."""
    
    def __init__(self, upper, lower, ndim):
        '''Initiate a Particle with an upper and lower bound, and a number of dimensions.'''
        # Initiate position and velocity randomly
        self.position = np.array([random.uniform(lower, upper) for _ in range(ndim)])
        self.velocity = np.array([random.uniform(-1, 1) for _ in range(ndim)])
        # These attributes are here to store the "memory" of the function
        self.personal_best_position = self.position # initial position
        self.personal_best_error = np.inf # infinity
    
    def __str__(self):
        '''what will be called when you print a Particle instance, could be useful for debugging.'''
        return("I am a particle with best position {},\ncurrent position {},\nand velocity {}.\n"
               .format(self.personal_best_position, self.position, self.velocity))       
    
    def update_velocity(self, global_best):
        '''update a particle's velocity'''
        # two random coefficients in uniform([0, 1])
        r1 = random.random()
        r2 = random.random()
        self.velocity = (w*self.velocity + # inertia
                         c1*r1*(self.personal_best_position - self.position) + # cognitive term
                         c2*r2*(global_best - self.position)) # social term
            
    def move(self):
        '''Moves a Particle at a new iteration. Deals with out-of-bound cases (are they ok or not?)'''
        self.position = self.position + self.velocity
        # need to deal with when the particle goes out of bound...
        
        
# --- Solver Class
class PSO_parallel:    
    """Parallel Solver instance for Particle Swarm Optimizer. (Basically the mastermind which commands the Particles.)
    Initiated with:
    - num_particles: how many particles to use?
    - function: function to optimize.
    - n_iter: number of iterations to be performed.
    - lower, upper: lower and upper bounds for the search space (for now, a box i.e. they are the same accross all dimensions.
    - ndim: dimensionality of the search space.)"""
    
    def __init__(self, num_particles, function, n_iter, lower = -10, upper = 10, ndim = 3):
        '''Initiate the solver'''
        # create all the Particles, stored in a list.
        self.particles = [Particle(lower, upper, ndim) for _ in range(num_particles)]
        self.fitnesses = np.array([])
        # store global best and associated error
        self.global_best = np.array([])
        self.global_best_error = np.inf # infinity
        self.function = function # function to be optimised
        self.n_iter = n_iter # num of iterations
        
    def get_fitnesses(self):
        with Pool(2) as p:
            fitnesses = p.apply(self.function, [[part.position for part in self.particles]])
        self.fitnesses = np.array(fitnesses)
    
    def update_particles(self):
        '''update particle best known personal position'''
        for i in range(len(self.fitnesses)):
            if self.fitnesses[i] < self.particles[i].personal_best_error:
                self.particles[i].personal_best_error = self.fitnesses[i]
                self.particles[i].personal_best_position = self.particles[i].position          
        
    def update_best(self):
        '''Find the new best global position and update it.'''
        if np.any(self.fitnesses < self.global_best_error):
            self.global_best_error = np.min(self.fitnesses)
            self.global_best = self.particles[np.argmin(self.fitnesses)].position
                      
    def move_particles(self):
        '''Run one iteration of the algorithm. Update particles velocity and move them.'''
        for particle in self.particles:
            particle.update_velocity(self.global_best)
            particle.move()
    
    def __str__(self):
        '''Print best global position when calling `print(pso instance)`'''
        return """Current best position: {}
        With error: {}""".format(self.global_best, self.global_best_error)
    
    def go(self):
        '''Run the algorithm and print the result. By default, update us every 50 iterations.'''
        print("Running the PSO algorithm in parallel with {} particles, for {} iterations.\n".
              format(len(self.particles), self.n_iter))
        
        for iteration in range(self.n_iter):
            # this happens in parallel (synchronous only rn)
            pso.get_fitnesses()
            # this doesn't
            pso.update_particles()
            pso.update_best()
            pso.move_particles()
            
            if iteration % 50 == 0:
                print("Iteration number " + str(iteration))
                print("Current best error: " + str(self.global_best_error))
                print("\n")
                
        print("Found minimum at {} with value {}.".format(self.global_best, self.global_best_error))