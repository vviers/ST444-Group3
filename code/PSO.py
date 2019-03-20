# import useful libraries/modules
import numpy as np
import random
import multiprocessing
from multiprocessing import Pool, Process, Queue, Value, Array

# ----- Define a useful Particle Class
class Particle:
    """A Python Class for a simple particle."""
    
    def __init__(self, upper, lower, ndim,
                 c1 = 1.49618, c2 = 1.49618, w = 0.7298):
        
        # c1 = c2 = 1.49618 and w = 0.7298, based on Van den Bergh, Engelbrecht (2006)
        
        '''Initiate a Particle with an upper and lower bound, and a number of dimensions.'''
        # Initiate position and velocity randomly
        self.position = np.array([random.uniform(lower, upper) for _ in range(ndim)])
        self.velocity = np.array([random.uniform(-1, 1) for _ in range(ndim)])
        # These attributes are here to store the "memory" of the function
        self.personal_best_position = self.position # initial position
        self.personal_best_fitness = np.inf # infinity
        self.c1 = c1
        self.c2 = c2
        self.w = w
    
    
    def __str__(self):
        '''what will be called when you print a Particle instance, could be useful for debugging.'''
        return("I am a particle with best position {},\ncurrent position {},\nand velocity {}.\n"
               .format(self.personal_best_position, self.position, self.velocity))
    
    
    def update_velocity(self, global_best):
        '''update a particle's velocity'''
        # two random coefficients in uniform([0, 1])
        r1 = random.random()
        r2 = random.random()
        self.velocity = (self.w*self.velocity + # inertia
                         self.c1*r1*(self.personal_best_position - self.position) + # cognitive term
                         self.c2*r2*(global_best - self.position)) # social term
                   
    def move(self):
        '''Moves a Particle at a new iteration. Deals with out-of-bound cases (We assume they are ok.)'''
        self.position = self.position + self.velocity
        # no need to deal with out of bounds cases, just adapt the function itself.
        
        
        


# ------- PSO Base Class -------------------------------------------------------------------------
class PSO:
    """
    +----------------------------------------------------------+
    + Parallel Solver instance for Particle Swarm Optimizer.   +
    + (Basically the mastermind which commands the Particles.) +
    +----------------------------------------------------------+
    
    Instantiated with:
    -----------------
    - num_particles: how many particles to use?
    
    - function: function to minimize. IT MUST BE of the form:
                function np.array[coordinates] --- f() ---> [list] (or np.array)
                typically, the function returns a list comprehension.
                
    - n_iter: number of iterations to be performed. To be replaced/completed with convergence criterions.
    
    - ndim: dimensionality of the search space.
    
    - lower, upper: lower and upper bounds for the search space
    (for now, it's a box i.e. they are the same accross all dimensions.)
    
    - c1, c2, w: cognitive, social, and inertia parameters. To be tuned to the specific problem.
    
    - parallel: whether to evaluate the fitness of particles in parallel. `False` by default as a speed-boost is
    unlikely in most simple settings.
    
    - threadpool: should the parallelization in fact be multi-threading (shared memory)?
    
    - epsilon: defines convergence. If an update to the global best is smaller than epsilon, then the algorithm
               has converged.
    
    """
    
    def __init__(self, num_particles, function, n_iter, ndim, lower = -10, upper = 10,
                 c1 = 1.49618, c2 = 1.49618, w = 0.7298, epsilon = 10e-7):
        '''Instantiate a simple PSO solver'''
        # create all the Particles, stored in a list.
        self.particles = [Particle(lower, upper, ndim, c1, c2, w) for _ in range(num_particles)]
        self.fitnesses = np.array([])
        # store global best and associated fitness
        self.global_best = np.array([])
        self.global_best_fitness = np.inf # infinity
        self.function = function # function to be optimised
        self.n_iter = n_iter # num of iterations
        self.epsilon = epsilon
        self.hasConverged = False
                
    def __str__(self):
        '''Print best global position when calling `print(pso instance)`'''
        return """Current best position: {}
        With fitness: {}""".format(self.global_best, self.global_best_fitness)
    
    def get_fitnesses(self):
        """Evaluate all fitnesses"""
        fitnesses = [self.function(part.position) for part in self.particles]
        self.fitnesses = np.array(fitnesses)
            
    def update_particles(self):
        '''update particle best known personal position'''
        for i in range(len(self.fitnesses)):
            if self.fitnesses[i] < self.particles[i].personal_best_fitness:
                self.particles[i].personal_best_fitness = self.fitnesses[i]
                self.particles[i].personal_best_position = self.particles[i].position
        
    def update_best(self):
        '''Find the new best global position and update it.
        Additionally check for convergence.'''
        if np.any(self.fitnesses < self.global_best_fitness):
            fit_before = self.global_best_fitness
            self.global_best_fitness = np.min(self.fitnesses)
            self.global_best = self.particles[np.argmin(self.fitnesses)].position
            
            #check convergence
            if abs(fit_before - self.global_best_fitness) < self.epsilon:
                self.hasConverged = True
                      
    def move_particles(self):
        '''Run one iteration of the algorithm. Update particles velocity and move them.'''
        for particle in self.particles:
            particle.update_velocity(self.global_best)
            particle.move()
     
    def run(self, verbose = True):
        '''Run the algorithm and print the result.'''
        
        if verbose:
            print("Running the PSO algorithm with {} particles, for at most {} iterations.\n".
                          format(len(self.particles), self.n_iter))

        c = 0 # loop counter
        for iteration in range(self.n_iter):
            # this could happen in parallel
            self.get_fitnesses()
            # this doesn't
            self.update_particles()
            self.update_best()
            self.move_particles()

            c += 1
            if self.hasConverged == True:
                break

        if verbose:
            print("After {} iterations,\nFound minimum at {} with value {}.".format(c, self.global_best,
                                                                                        self.global_best_fitness))
        return(self.global_best)
        
        
# ------- PSO Synchronous ------------------------------------------------------------------------      
class PSO_synchron(PSO):
    '''Derived from the base PSO class, evaluate particle fitness in parallel (synchronously)'''
    def __init__(self, num_particles, function, n_iter, ndim, lower = -10, upper = 10,
                         c1 = 1.49618, c2 = 1.49618, w = 0.7298, epsilon = 10e-7):
        
        # Init base PSO class
        super().__init__(num_particles, function, n_iter, ndim, lower, upper,
                         c1, c2, w, epsilon)
        
        # Create Multiprocessing Pooler
        self.pooler = Pool(multiprocessing.cpu_count() - 1)
    
    # overload get_fitness so that this happens in parallel
    def get_fitnesses(self):
        """Evaluate all fitnesses in parallel (synchronously)"""
        fitnesses = self.pooler.map(self.function, [p.position for p in self.particles])
        self.fitnesses = np.array(fitnesses)
        
        

        
# ------- PSO ASynchronous -----------------------------------------------------------------------
class PSO_asynchron(PSO):
    '''Derived from the base PSO class, evaluate particle fitness in parallel (asynchronously)'''
    def __init__(self, num_particles, function, n_iter, ndim, lower = -10, upper = 10,
                         c1 = 1.49618, c2 = 1.49618, w = 0.7298, epsilon = 10e-7):
        
        # Init base PSO class
        super().__init__(num_particles, function, n_iter, ndim, lower, upper,
                         c1, c2, w, epsilon)
        
        # Add asynchron attributes (overrides those from base class when they exist)
        #--
        # Create shared memory array for storing global best position
        self.global_best = Array('d', ndim)
        # Create shared memory Value for global best fitness
        self.global_best_fitness = Value('d', np.inf)
        # Shared counter (to count the number of function evaluations)
        self.count = Value('i', 0)
        # How many time should we evaluate the function?
        self.n_func_eva = n_iter * num_particles
        self.stop_queuing = self.n_func_eva - num_particles
    
    # define methods for asynchronous parallel PSO
    def worker(self, queue):
        '''A worker used for asynchronous parallelisation.'''
        for part in iter(queue.get, 'STOP'):
            
            # get fitness of particle and update if better
            fitness = self.function(part.position)
            
            if fitness < part.personal_best_fitness:
                part.personal_best_fitness = fitness
                part.personal_best_position = part.position
                
            # update global fitness if needed
            # best_fitness_before = self.global_best_fitness.value #(for convergence check)
            if fitness < self.global_best_fitness.value:
                self.global_best_fitness.value = fitness
                self.global_best[:] = part.position
                # check convergence!
                # self.hasConverged = True if best_fitness_before - fitness < self.epsilon else False
            
            # upgrade and move particle
            part.update_velocity(self.global_best[:])
            part.move()
            
            # update the update count by one, check if should line up for another update
            self.count.value +=1
            
            # make sure that we only evaluate the function `n_func_eva` time in total
            # also convergence criterion check!
            if self.count.value <= self.stop_queuing:# and not self.hasConverged:
                queue.put(part)
            else:
                queue.put('STOP')
                
    # override the run method
    def run(self, verbose=True):
        if verbose:
            print("Running the PSO algorithm asynchronously with {} particles, for at most {} function evaluations.\n".
                    format(len(self.particles), self.n_func_eva))
                
        # run 3 processes in parallel
        n_processes = 3
        # create a Queue
        task_queue = Queue()

        # asynchron PSO
        for part in self.particles:
            task_queue.put(part)
            processes = []
        for i in range(n_processes):
            p=Process(target=self.worker, args=(task_queue,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Return value of the optimisation procedure
        if verbose:
            print("After {} function evaluations,\nFound minimum at {} with value {}.".format(self.count.value, 
                                                                                              self.global_best[:],
                                                                                              self.global_best_fitness.value))
        return self.global_best[:]