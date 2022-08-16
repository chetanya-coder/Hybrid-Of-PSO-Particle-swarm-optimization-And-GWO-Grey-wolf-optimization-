# Hybridization-of-PSO-and-GWO

This Project is based on the research paper by Narinder Singh and S. B. Singh, " Hybrid Algorithm of Particle Swarm Optimization and Grey Wolf Optimizer for Improving Convergence Performance", Journal of Applied Mathematics, vol. 2017, pp. 1-15, 2017. "

The main idea is to improve the ability of exploitation in Particle Swarm Optimization with the ability of exploration in Grey Wolf Optimizer to produce both variants’ strength.

We have used 9 standard benchmark functions to evaluate the performance of this BPSOGWO. The results show that BPSOGWO is significantly better than the original version of GWO and PSO. The BPSOGWO improved the global optimum value in some of the functions, while in all the others, it reached the global optimum value in fewer iterations.


<br/>
<h2>Components</h2>
plot.py file plots the graph of all three optimization algorithms' performance with " optimum value vs iterations "<br/>
benchmark.py file consist of 9 benchmark's function which can be used to evaluate/compare all three algorithms.

<h3>How to Use:</h3>
open plot.py file and set the value of variable " function_to_be_used " from F1 to F9 representing the 9 benchmark function.<br/>
run plot.py file.
