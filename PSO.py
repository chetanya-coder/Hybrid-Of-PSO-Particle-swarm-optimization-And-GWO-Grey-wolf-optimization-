import numpy as np


class PSO:
    def __init__(self, fitness_func, c1=0.5, c2=0.5, dimension=30, pop_size=20, max_iter=100):
        self.dimension = dimension
        self.pop_size = pop_size
        self.max_iter = max_iter

        # Accelerations constants
        self.c1 = c1  # cognitive constant (p_best)
        self.c2 = c2  # social constant (g_best)
        self.w = 0.9  # Inertia

        self.cost_func = fitness_func  # Cost function
        self.X = None  # Positions
        self.V = None  # Velocities
        self.p_best = None  # Particles best positions so far
        self.g_best = None  # Global best position

        # Keeps a list of g_best at each iteration to plot the evolution
        self.evolution = []

    def random_init(self):
        self.X = np.random.rand(self.pop_size, self.dimension)
        self.V = np.random.rand(self.pop_size, self.dimension)
        # At first p_best is equal to first positions
        self.p_best = np.copy(self.X)
        # At first we consider that the X_1 is the global best
        self.g_best = self.X[0]

    def update_p_best(self):
        for index, values in enumerate(self.X):
            cost = self.cost_func(values)
            if cost < self.cost_func(self.p_best[index]):
                self.p_best[index] = values

    def update_g_best(self):
        for index, values in enumerate(self.p_best):
            cost = self.cost_func(values)
            if cost < self.cost_func(self.g_best):
                self.g_best = values

    def update_velocity(self):
        r1 = np.random.random_sample()
        r2 = np.random.random_sample()
        self.V = (self.w * self.V +
                  self.c1 * r1 * (self.p_best - self.X) +
                  self.c2 * r2 * (self.g_best - self.X))

    def update_positions(self):
        self.X = self.X + self.V

    def start(self):

        i = 0
        while i < self.max_iter:
            self.update_p_best()
            self.update_g_best()
            self.update_velocity()
            self.update_positions()

            self.evolution.append(self.cost_func(self.g_best))

            i += 1

    def return_result(self):

        return np.array(self.evolution)
