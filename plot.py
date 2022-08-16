import matplotlib.pyplot as plt
from HPSOGWO import HPSOGWO
from GWO import GWO
import benchmarks
from PSO import PSO

function_to_be_used = benchmarks.F5

s = GWO(function_to_be_used)
s.opt()
x1 = s.return_result()

d = HPSOGWO(function_to_be_used)
d.opt()
x2 = d.return_result()


pso = PSO(function_to_be_used)
pso.random_init()
pso.start()
x3 = pso.return_result()

# PLOT
plt.figure()
plt.plot(x1)
plt.plot(x2)
plt.plot(x3)
plt.grid()
plt.legend(["GWO", "HPSOGWO", "PSO"], loc="upper right")
plt.title("Comparision of HPSOGWO with PSO and GWO")
plt.xlabel("Number of Iterations")
plt.ylabel("Best Score")
plt.show()
