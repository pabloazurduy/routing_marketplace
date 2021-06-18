from data_models import OptInstance, Solution
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

# =========================== #
# ===  optimization model === #
# =========================== #
model = mip.Model(name = 'clustering', sense = mip.MINIMIZE)

# Instance Parameters 
n_clusters = 25 # max number of clusters ? TODO: there should be a Z* equivalent way of modeling this problem 
clusters = range(n_clusters)

# var declaration
y = {} # cluster variables 
for node,c in it.product(opt_instance.nodes, clusters): # in cluster var
    y[(node.sid,c)] = model.add_var(var_type = mip.BINARY , name = f'cluster_{node.sid}_{c}')
    
    #closest node var  
    """TODO: remove 
    # b = {} # closest node var
    for node_j in opt_instance.nodes:
        if node_j.sid != node.sid:
            b[(c, node.sid, node_j.sid)] = model.add_var(var_type = mip.BINARY , name = f'closest_node_c{c}_osid{node.sid}_dsid{node_j.sid}')
    """

# features 
ft_size = {}
ft_size_drops = {}
ft_size_pickups = {}
ft_has_geo = {}
ft_size_geo = {}
# ft_dist_appx ={}

for c in clusters:
    ft_size[c] =         model.add_var(var_type = mip.CONTINUOUS , name = f'ft_size_c{c}', lb=0)
    ft_size_drops[c] =   model.add_var(var_type = mip.CONTINUOUS , name = f'ft_size_drops_c{c}', lb=0)
    ft_size_pickups[c] = model.add_var(var_type = mip.CONTINUOUS , name = f'ft_size_pickups_c{c}', lb=0)
    ft_size_geo[c] =     model.add_var(var_type = mip.CONTINUOUS , name = f'ft_size_geos_c{c}', lb=0)
    # ft_distance_appx[c] =model.add_var(var_type = mip.CONTINUOUS , name = f'ft_distance_appx_c{c}', lb=0)
    
    for geo in opt_instance.comunas:
        ft_has_geo[(c,geo.id)] = model.add_var(var_type = mip.BINARY , name = f'ft_has_geo_c{c}_{geo.name}')
    
# ======================== #
# ===== constraints ====== #
# ======================== #

# Cluster Constraint
for node in opt_instance.drops:
    # 0. demand satisfy
    model.add_constr(mip.xsum([y[(node.sid, c)] for c in clusters]) ==  1, name=f'cluster_fill_{node.sid}') # TODO SOS ? 
    # 1. pair drop,warehouse 
    model.add_constr(y[(node.sid, c)] == y[(node.warehouse_sid, c)], name=f'pair_drop_warehouse_{node.sid}_{node.warehouse_sid}') 

# Size Features
for c in clusters:
    # 2. cod ft_size
    model.add_constr(ft_size[c] == mip.xsum([y[(node.sid, c)] for node in opt_instance.nodes]), name=f'cod_ft_size_c{c}') 
    # 3. cod ft_size_drops
    model.add_constr(ft_size_drops[c] == mip.xsum([y[(node.sid, c)] for node in opt_instance.drops]), name=f'cod_ft_size_drops_c{c}') 
    # 4. cod ft_size_pickups
    model.add_constr(ft_size_pickups[c] == mip.xsum([y[(node.sid, c)] for node in opt_instance.warehouses]), name=f'cod_ft_size_pickups_c{c}') 

# Geo Codifications
M1 =  len(opt_instance.nodes)
for c,geo in it.product(clusters,opt_instance.comunas):
    # 5. cod min ft_has_geo 
    model.add_constr(M1 * ft_has_geo[(c,geo.id)] >= mip.xsum([y[node.sid,c] for node in opt_instance.drops if node.comuna_id == geo.id]), name=f'cod_ft_has_geo_min_{c}_{geo.id}') 
    # 6. cod max ft_has_geo 
    model.add_constr(ft_has_geo[(c,geo.id)] <= mip.xsum([y[node.sid,c] for node in opt_instance.drops if node.comuna_id == geo.id]), name=f'cod_ft_has_geo_max_{c}_{geo.id}') 

for c in clusters:
    # 7. cod ft_size_geos 
    model.add_constr(ft_size_geo[c] == mip.xsum([ft_has_geo[(c,geo.id)] for geo in opt_instance.comunas]), name=f'cod_ft_size_geos_{c}_{geo.id}') 

# Cluster route distance proxy 
"""TODO:remove 
M2 = 100 # longest route km 
for c, node_i, node_j in it.product(clusters, opt_instance.nodes, opt_instance.nodes):
    if node_i.sid != node_j.sid:
        d_ij = opt_instance.distance(node_i.sid, node_j.sid)
        # 8. cod_max_distance_same_cluster_c
        model.add_constr(ft_distance_appx[c] <= y[(node_i.sid, c)]*d_ij + M2*(1-y[(node_i.sid, c)]), name= f'cod_max_distance_same_cluster_c{c}_{node_i.sid}_{node_j.sid}') 
        # 9 cod_min_distance_same_cluster_c
        model.add_constr(ft_distance_appx[c] >= y[(node_i.sid, c)]*d_ij + M2*(1-y[(node_i.sid, c)]) - 2*M2*(1-b[(c, node_i.sid, node_j.sid)]), 
                         name= f'cod_min_distance_same_cluster_c{c}_{node_i.sid}_{node_j.sid}') 

for c, node_i in it.product(clusters, opt_instance.nodes): 
    model.add_constr(mip.xsum([b[(c, node_i.sid, node_j.sid)] ==1 for node_j in opt_instance.nodes if node_i.sid != node_j.sid]), name=f'cod_closest_node_var_c{c}_{node_i.sid}') 
"""

# objective function
beta_size = -1
beta_size_drops = -1
beta_size_pickups = -1
beta_size_geo = -400

model.objective = mip.xsum([beta_size*ft_size[c] + 
                            beta_size_drops*ft_size_drops[c] + 
                            beta_size_pickups*ft_size_pickups[c] +
                            beta_size_geo*ft_size_geo[c]
                            for c in clusters])

model.max_seconds = 60 * 25 # min 
model.optimize()

solution_dict = { 'y':  y,  
                  'ft_size' :  ft_size,
                  'ft_size_drops' :  ft_size_drops,
                  'ft_size_pickups' :  ft_size_pickups,
                  'ft_has_geo' :  ft_has_geo,
                  'ft_size_geo' :  ft_size_geo,
                }

opt_instance.solution = Solution(y = y)
opt_instance.plot()
