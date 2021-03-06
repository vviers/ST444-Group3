# import two useful libraries/modules
import numpy as np
import random
import multiprocessing
from multiprocessing import Pool, Process, Queue, Value, Array
import matplotlib
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------------------
class Particle:
    """A Python Class for a simple particle."""
    
    def __init__(self, upper, lower, ndim, c1 = 1, c2 = 2, w = .5):
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
        '''Moves a Particle at a new iteration. Deals with out-of-bound cases (are they ok or not?)'''
        self.position = self.position + self.velocity
        # need to deal with when the particle goes out of bound...
    
# ----------------------------------------------------------------------------------------------------------------
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
    
    """
    
    def __init__(self, num_particles, function, n_iter, ndim, n_func_eva=5000, lower = -10, upper = 10,
                 c1 = 1, c2 = 2, w = .5, parallel = False, asy_parallel = False, visual = False):
        '''Initiate the solver'''
        # create all the Particles, stored in a list.
        self.particles = [Particle(lower, upper, ndim, c1, c2, w) for _ in range(num_particles)]
        self.fitnesses = np.array([])
        # store global best and associated fitness
        self.global_best = np.array([])
        self.global_best_fitness = np.inf # infinity
        self.function = function # function to be optimised
        self.n_iter = n_iter # num of iterations
        self.parallel = parallel
        self.visual = visual

        if parallel:
            self.pooler = Pool(multiprocessing.cpu_count()-1)
                
        if asy_parallel:
            self.global_best = Array('d',ndim)
            self.global_best_fitness = Value('d', np.inf)
            self.count = Value('i', 0)
            self.n_func_eva = n_func_eva
            self.num_particles = num_particles
        
    def get_fitnesses(self):
        """Evaluate all fitnesses (in parallel if self.parallel is True.)"""
        if self.parallel:            
            fitnesses = self.pooler.map(self.function, [p.position for p in self.particles])
            self.fitnesses = np.array(fitnesses)
        else :
            fitnesses = [self.function(part.position) for part in self.particles]
            self.fitnesses = np.array(fitnesses)
    
    def update_particles(self):
        '''update particle best known personal position'''
        for i in range(len(self.fitnesses)):
            if self.fitnesses[i] < self.particles[i].personal_best_fitness:
                self.particles[i].personal_best_fitness = self.fitnesses[i]
                self.particles[i].personal_best_position = self.particles[i].position          
        
    def update_best(self):
        '''Find the new best global position and update it.'''
        if np.any(self.fitnesses < self.global_best_fitness):
            self.global_best_fitness = np.min(self.fitnesses)
            self.global_best = self.particles[np.argmin(self.fitnesses)].position
                      
    def move_particles(self):
        '''Run one iteration of the algorithm. Update particles velocity and move them.'''
        for particle in self.particles:
            particle.update_velocity(self.global_best)
            particle.move()
    
    def __str__(self):
        '''Print best global position when calling `print(pso instance)`'''
        return """Current best position: {}
        With fitness: {}""".format(self.global_best, self.global_best_fitness)
    
    def visualise(self, iteration):
        '''visualisation for PSO on Schaffer f6'''
        matplotlib.rcParams['xtick.direction'] = 'out'
        matplotlib.rcParams['ytick.direction'] = 'out'        
        plt.rcParams["figure.figsize"] = [12,9]
        x = np.arange(-50, 50, 1.25)
        y = np.arange(-50, 50, 1.25)
        X, Y = np.meshgrid(x, y)
        Z = .5 + ((np.sin(np.sqrt(X**2 + Y**2)))**2 - .5)/((1 + 0.001*(X**2 + Y**2))**2)
        CS = plt.contour(X, Y, Z, 6)
        plt.colorbar(CS, shrink = 0.8, extend = 'both')
        for part in self.particles:
            circle = plt.Circle((part.position[0], part.position[1]), .5, facecolor='r', edgecolor='r', alpha=0.8)
            plt.gca().add_patch(circle)
            
        plt.savefig('figure_' + str(iteration) + '.png')
        plt.close()
    
    def run(self, verbose = True):
        '''Run the algorithm and print the result. By default, update us every 50 iterations.'''
        
        if verbose:
            print("Running the PSO algorithm in parallel with {} particles, for {} iterations.\n".
                  format(len(self.particles), self.n_iter))
        
        for iteration in range(self.n_iter):
            # this happens in parallel (synchronous only rn)
            if self.visual:
                self.visualise(iteration)
            self.get_fitnesses()
            # this doesn't
            self.update_particles()
            self.update_best()
            self.move_particles()
            
            if (iteration % 50 == 0) & verbose==True:
                print("Iteration number " + str(iteration))
                print("Current best fitness: " + str(self.global_best_fitness))
                print("\n")
                
        if verbose:
            print("Found minimum at {} with value {}.".format(self.global_best, self.global_best_fitness))
            
        return(self.global_best)
    
    
    # ----- Parallel Asynchronous PSO stuff
    def worker(self, queue):
        '''A worker used for asynchronous parallelisation.'''
        for part in iter(queue.get, 'STOP'):
            
            # get fitness of particle and update if better
            fitness = self.function(part.position)
            
            if fitness < part.personal_best_fitness:
                part.personal_best_fitness = fitness
                part.personal_best_position = part.position
                
            # update global fitness if needed
            if fitness < self.global_best_fitness.value:
                self.global_best_fitness.value = fitness
                self.global_best[:] = part.position
            
            # upgrade and move particle
            part.update_velocity(self.global_best[:])
            part.move()
            
            # update the update count by one, check if should line up for another update
            self.count.value +=1
            
            # make sure that we only evaluate the function `n_func_eva` time in total
            # TODO: add convergence criterion check!
            if self.count.value <= self.n_func_eva - self.num_particles:
                queue.put(part)
                
            else:
                queue.put('STOP')
                
    def asynch_run(self):
        '''Run the algorithm in Asynchronous Parallel.'''
        # run 3 processes in parallel
        n_processes = 3
        # create a Queue
        task_queue = Queue()
        
        #
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
        print("Found minimum at {} with value {}.".format(self.global_best[:], self.global_best_fitness.value))



    
#def schaffer_f6(x): 
    
#    return .5 + ((np.sin(np.sqrt(x[0]**2 + x[1]**2))**2) - .5)/((1 + 0.001*(x[0]**2 + x[1]**2))**2)



 
#pso = PSO(30, schaffer_f6, 50, 2, lower=-100, upper=100, asy_parallel=True)
#pso.asy_run()

