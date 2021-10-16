import unittest
import mip 

class EnvTest(unittest.TestCase): 
    def test_gurobi_solver(self):
        model = mip.Model('knapsack',sense=mip.MAXIMIZE, solver_name='gurobi')
        p = [10, 13, 18, 31, 7, 15]
        w = [11, 15, 20, 35, 10, 33]
        c, I = 47, range(len(w))

        x = [model.add_var(var_type=mip.BINARY) for i in I]

        model.objective = mip.xsum(p[i] * x[i] for i in I)

        model += mip.xsum(w[i] * x[i] for i in I) <= c

        model.optimize()

        selected = [i for i in I if x[i].x >= 0.99]
        print("selected items: {}".format(selected))
    