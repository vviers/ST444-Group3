#working directory to save figures
import os
os.chdir("C:/Users/User/Documents/Academics/MSc Statistics/ST444/Project")

#run with visual=True
pso = PSO(50, schaffer_f6, 100, 2, lower=-50, upper=50, visual=True)
pso.run()
