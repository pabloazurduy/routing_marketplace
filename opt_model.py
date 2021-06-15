from data_models import OptInstance
from datetime import date
import pandas as pd 
import numpy as np
import mip
import itertools as it 

# create instance 
# load data 
instance_df = pd.read_csv('instance_simulator/real_instances/instance_2021-05-24.csv', 
                          sep=';')
# add req_date 
instance_df['req_date'] = np.where(~instance_df['is_warehouse'],  
                                   date(2021,5,24), 
                                   None)

opt_instance = OptInstance.load_instance(instance_df)


# optimization model 

model = mip.Model(name = 'clustering')

"""
# var declaration
y = {} # cluster variables
for i,j in it.product(opt_isntance.nodes):
    # declarate path variables 
    if i != j : 
        x[(i,j,k)] = model.add_var(var_type = mip.BINARY , 
                                name = f'path_{i}_{j}_{k}')

y = {}    
for i,k in itertools.product(nodes,trucks):
    # declarate path variables 
    y[(i,k)] = model.add_var(var_type = mip.CONTINUOUS , 
                            name = f'asignation_{i}_{k}')

u = {} # path length variable 
for i,k in itertools.product(nodes,trucks):
    u[(i,k)] = model.add_var(var_type = mip.INTEGER , 
                            name = f'order_{i}_{k}')


# ======================== #
# ===== constraints ====== #
# ======================== #

# 0. end node codification  
for k in trucks:
    model.add_constr(mip.xsum([x[(origin[k],j,k)] for j in nodes if j!= origin[k]]) <= 1, name=f'origin_out_cod_{k}' ) 

# 1. flow conservation
for i,k in itertools.product(nodes,trucks):
    if i != origin[k]:
        model.add_constr(mip.xsum([x[(j,i,k)] for j in nodes if j!=i ]) == # lo que entra
                        mip.xsum([x[(i,j,k)] for j in nodes if j!=i ]) , # lo que sale
                        name = f'flow_conservation_{i}_{k}' ) 

# 2. y codification 
for i,k in itertools.product(nodes,trucks):
    model.add_constr(y[(i,k)] == mip.xsum([x[(j,i,k)] for j in nodes if j!=i]) , name=f'y[{i}{k}]_cod') 

# 3. demand fulfillment
for i in nodes:
    if i not in origin.values(): # is not an origin node
        model.add_constr(mip.xsum([ y[(i,k)] for k in trucks]) == 1 , name=f'y[{i}{k}]_cod') 


# 4. subtour elimination 
graph_len = len(nodes)
for k in trucks:
    for i,j in itertools.product(nodes,nodes):
        if i != j and (i != origin[k] and j!= origin[k]): # remove origin 
            model.add_constr(u[(i,k)] - u[(j,k)] + 1  <= graph_len*(1- x[(i,j,k)]) , name=f'subtour_constraint_{i}_{j}_{k}')
    
    model.add_constr(u[(origin[k],k)] == 1 , name=f'subtour_constraint_origin_{k}')
    
    for i in nodes:
        if i != origin[k] :
            model.add_constr(u[(i,k)] >=2  , name=f'subtour_constraint_lowerbound_{i}')
            model.add_constr(u[(i,k)] <= graph_len -1, name=f'subtour_constraint_upperbound_{i}')
            

# ============================ #
# ==== model declaration ===== #
# ============================ #

# objective function
if objective_function == 'min_distance':
    model.objective = mip.xsum([x[key]*vrp_instance.cost(key[0],key[1],key[2]) for key in x.keys()])

elif objective_function == 'lowest_pos':
    model.objective = mip.xsum([u[key] for key in u.keys()])

if objective_function == 'min_dist_max_len':
    for key in u.keys():
        model.add_constr(u[key] <= int(graph_len/len(trucks) *1.15) +1  , name='max_len')
    model.objective = mip.xsum([x[key]*vrp_instance.cost(key[0],key[1],key[2]) for key in x.keys()])

if objective_function == 'min_last_attended':
    M = 10e6
    t_len = {}
    bin_l = {}
    max_route = model.add_var(name='max_route', var_type = mip.CONTINUOUS)
    
    for k in trucks:
        t_len[k] = model.add_var(name=f'route_len_{k}', var_type = mip.CONTINUOUS)
        bin_l[k] = model.add_var(name=f'cod_route_max_{k}', var_type = mip.BINARY)

        model.add_constr(t_len[k] == mip.xsum([ x[key] * vrp_instance.cost(key[0],key[1],k)  for key in x.keys() if key[2]==k ]), 
                                name = f'route_len_{k}')

        model.add_constr(max_route >= t_len[k], name=f'max_route_cod_lb_{k}')
        model.add_constr(max_route <= t_len[k] + M * (1-bin_l[k]), name=f'max_route_cod_ub_{k}')

    model.objective = max_route

model.sens = mip.MINIMIZE

# model tunning
# cuts
# -1  automatic, 0 disables completely, 
# 1 (default) generates cutting planes in a moderate way,
# 2 generates cutting planes aggressively  
# 3 generates even more cutting planes
model.cuts = 2 
model.max_mip_gap = 0.005 # 0.5%
model.max_seconds = 25*60 
model.optimize()
"""