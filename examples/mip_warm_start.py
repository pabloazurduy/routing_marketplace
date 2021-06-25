from mip import Model, xsum, maximize, BINARY
import numpy as np
import itertools as it 
import logging
logging.getLogger(__name__).setLevel(logging.DEBUG)

np.random.seed(42)

p = np.random.uniform(size=1000)
w = np.random.uniform(size=1000)
c, I = 47, range(len(w))

m = Model("knapsack")

x = [m.add_var(var_type=BINARY) for i in I]

m.objective = maximize(xsum(p[i] * x[i] for i in I))

m += xsum(w[i] * x[i] for i in I) <= c

pairs = []
for iter in range(1000):
    i = np.random.randint(len(w))
    j = np.random.randint(len(w))
    # m += x[i] >= x[j]
    if i==0 or j ==0:
        pairs.append((i,j))

print('validating mip start')
m.start = [(x[0],1.0)]+[(var,0) for var in x[1:]]
m.validate_mip_start()
print('optimizing')
m.optimize()

selected = [i for i in I if x[i].x >= 0.99]
print("selected items: {}".format(selected))