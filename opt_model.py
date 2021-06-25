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
model = mip.Model(name = 'clustering', sense = mip.MAXIMIZE)

# Instance Parameters 
n_clusters = 25 # max number of clusters ? TODO: there should be a Z* equivalent way of modeling this problem 
clusters = range(n_clusters)

# var declaration
print('var declaration')
y = {} # cluster variables 
for node,c in it.product(opt_instance.nodes, clusters): # in cluster var
    y[(node.sid,c)] = model.add_var(var_type = mip.BINARY , name = f'y_{c}_{node.sid}')


z = {} # distance variables 
for c,(g1,g2) in it.product(clusters, it.combinations(opt_instance.geos, 2)): # unique combinations  
    z[(c,g1.id,g2.id)] = model.add_var(var_type = mip.BINARY , name = f'z_{c}_{g1.id}_{g2.id}')


# features 
ft_size = {}
ft_size_drops = {}
ft_size_pickups = {}
ft_has_geo = {}
ft_size_geo = {}
ft_inter_geo_dist = {}

for c in clusters:
    ft_size[c] =           model.add_var(var_type = mip.CONTINUOUS , name = f'ft_size_c{c}', lb=0)
    ft_size_drops[c] =     model.add_var(var_type = mip.CONTINUOUS , name = f'ft_size_drops_c{c}', lb=0)
    ft_size_pickups[c] =   model.add_var(var_type = mip.CONTINUOUS , name = f'ft_size_pickups_c{c}', lb=0)
    ft_size_geo[c] =       model.add_var(var_type = mip.CONTINUOUS , name = f'ft_size_geos_c{c}', lb=0)
    ft_inter_geo_dist[c] = model.add_var(var_type = mip.CONTINUOUS , name = f'ft_inter_geo_dist_c{c}', lb=0)
    
    for geo in opt_instance.geos:
        ft_has_geo[(c,geo.id)] = model.add_var(var_type = mip.BINARY , name = f'has_geo_{c}_{geo.id}')
    
# ======================== #
# ===== constraints ====== #
# ======================== #
print('adding cluster constraints')
# Cluster Constraint
for node in opt_instance.drops:
    # 0. demand satisfy
    model.add_constr(mip.xsum([y[(node.sid, c)] for c in clusters]) ==  1, name=f'cluster_fill_{node.sid}') # TODO SOS ? 

for node,c in it.product(opt_instance.drops, clusters):
    # 1. pair drop,warehouse 
    model.add_constr(y[(node.sid, c)] <= y[(node.warehouse_sid, c)], name=f'pair_drop_warehouse_{node.sid}_{node.warehouse_sid}') 

for c, wh in it.product(clusters, opt_instance.warehouses):
    # 2. remove unused nodes 
    model.add_constr(mip.xsum([y[(drop.sid, c)] for drop in opt_instance.drops if drop.warehouse_id == wh.id]) >= y[(wh.sid, c)], 
                     name=f'no_wh_if_no_need_to_c{c}_{wh.sid}') 

print('adding size features constraints')
# Size Features
for c in clusters:
    # 2. cod ft_size
    model.add_constr(ft_size[c] == mip.xsum([y[(node.sid, c)] for node in opt_instance.nodes]), name=f'cod_ft_size_c{c}') 
    # 3. cod ft_size_drops
    model.add_constr(ft_size_drops[c] == mip.xsum([y[(node.sid, c)] for node in opt_instance.drops]), name=f'cod_ft_size_drops_c{c}') 
    # 4. cod ft_size_pickups
    model.add_constr(ft_size_pickups[c] == mip.xsum([y[(node.sid, c)] for node in opt_instance.warehouses]), name=f'cod_ft_size_pickups_c{c}') 

# Geo Codifications
print('adding geo cod constraints')
M1 =  len(opt_instance.nodes)+1
for c,geo in it.product(clusters,opt_instance.geos):
    # 5. cod min ft_has_geo 
    model.add_constr(M1 * ft_has_geo[(c,geo.id)] >= mip.xsum([y[node.sid,c] for node in opt_instance.nodes if node.geo_id == geo.id]), name=f'cod_ft_has_geo_min_{c}_{geo.id}') 
    # 6. cod max ft_has_geo 
    model.add_constr(     ft_has_geo[(c,geo.id)] <= mip.xsum([y[node.sid,c] for node in opt_instance.nodes if node.geo_id == geo.id]), name=f'cod_ft_has_geo_max_{c}_{geo.id}') 

for c in clusters:
    # 7. cod ft_size_geos 
    model.add_constr(ft_size_geo[c] == mip.xsum([ft_has_geo[(c,geo.id)] for geo in opt_instance.geos]), name=f'cod_ft_size_geos_{c}_{geo.id}') 

# Inter Geo Codification
print('adding inter geo cod constraints')
for (c,g1,g2) in z.keys():
    # 8. codification z min has_geo_g1 
    model.add_constr(z[(c,g1,g2)] <= ft_has_geo[(c,g1)], name=f'cod_z_min_bound_g1_{c}_{g1}_{g2}') # this formulation has more constraints than the sum_g2 <= ..
    # 9. codification z min has_geo_g2 
    model.add_constr(z[(c,g1,g2)] <= ft_has_geo[(c,g2)], name=f'cod_z_min_bound_g2_{c}_{g1}_{g2}')
    # 9. codification z up bound  
    model.add_constr(z[(c,g1,g2)] >= ft_has_geo[(c,g1)] + ft_has_geo[(c,g2)] -1  , name=f'cod_z_max_bound_{c}_{g1}_{g2}') 

for c in clusters:
    # 7. cod ft_inter_geo_dist 
    model.add_constr(ft_inter_geo_dist[c] == mip.xsum([z[(c,g1.id,g2.id)] * float(g1.distance(g2)) for g1,g2 in it.combinations(opt_instance.geos,2)]),
                     name=f'cod_ft_inter_geo_dist_{c}') 

print('adding objective function')
# objective function
beta_size = 4
beta_size_drops = -1
beta_size_pickups = -1
beta_size_geo = -1
beta_ft_inter_geo_dist = -2

model.objective = mip.xsum([beta_size*ft_size[c] 
                            + beta_size_drops*ft_size_drops[c]
                            + beta_size_pickups*ft_size_pickups[c]
                            + beta_size_geo*ft_size_geo[c] 
                            + beta_ft_inter_geo_dist*ft_inter_geo_dist[c] 
                            for c in clusters])

model.max_seconds = 60 * 20 # min 
# get a warm start 
warm_start = opt_instance.get_warm_start(n_clusters)
start_list = [(model.var_by_name(var_name), value_start) for (var_name, value_start) in warm_start.items()]

print('validating start')
model.start = start_list 
# model.validate_mip_start()

print('optimization starting')
model.optimize()

solution_dict = { 'y':  y,  
                  'ft_size' :  ft_size,
                  'ft_size_drops' :  ft_size_drops,
                  'ft_size_pickups' :  ft_size_pickups,
                  'ft_has_geo' :  ft_has_geo,
                  'ft_size_geo' :  ft_size_geo,
                  'ft_inter_geo_dist' : ft_inter_geo_dist,
                }

for c in clusters:
    print(f'{c = }, {ft_size_geo[c].x = }, {ft_size_drops[c].x = }, {ft_size_drops[c].x = }, {ft_size_pickups[c].x = }, {ft_inter_geo_dist[c].x = }')

opt_instance.solution = Solution(y = y, z = z)

opt_instance.plot()