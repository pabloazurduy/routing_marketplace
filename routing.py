# Data Models + Utilities

import itertools as it
import json
import random
from copy import deepcopy
from datetime import date
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union

import mip
import pandas as pd
from keplergl import KeplerGl
from pydantic import BaseModel
from shapely.geometry import LineString, Point, Polygon, shape
from sklearn import cluster  # used in eval
from sklearn import linear_model

from constants import KEPLER_CONFIG


# city meta classes 
class Geo(BaseModel):
    id: int
    name: Optional[str]
    polygon: Polygon
    area: Optional[float]
    class Config:
        arbitrary_types_allowed = True

    @property
    def centroid(self):
        return self.polygon.centroid

    def contains(self, lat:float, lng:float) -> bool:
        point = Point(lng,lat)
        return self.polygon.contains(point)
    
    def distance(self, other) -> float:
        # hausdorff_distance:  longest distance        
        # distance: shortest distance 
        # representative_point: a point inside the object 
        # centroid: centroid of the polygon    
        # return self.polygon.hausdorff_distance(other.polygon)
        self_centroid = self.polygon.centroid
        other_centroid = other.polygon.centroid
        return self_centroid.distance(other_centroid )

    def get_random_point(self)-> Point:
        xmin, ymin, xmax, ymax = self.polygon.bounds
        while True:
            x = random.uniform(xmin, xmax)
            y = random.uniform(ymin, ymax)
            point = Point(x, y)
            if point.within(self.polygon):
                return point 

class City(BaseModel):
    id: Optional[int]
    name_city: Optional[str]
    geos: Dict[int, Geo]
    
    # cache_distance 
    dist_geos: Optional[Dict[frozenset, float]]

    @property
    def geos_list(self) -> List[Geo]:
        return List(self.geos.values())
    @classmethod
    def from_geojson(cls, geojson_file_path:str, name_city:Optional[str]=None, id:Optional[int]=None):

        with open(geojson_file_path) as f:
            geojson = json.load(f)
        
        geos = {}
        for geo_id, feature in enumerate(geojson['features']):
            polygon = shape(feature['geometry'])
            geos[geo_id] =  Geo(id = geo_id,
                                name = feature['properties']['NOM_COM'],
                                area = feature['properties']['SHAPE_Area'],
                                polygon = polygon)

        return cls(id=id, name_city = name_city, geos=geos)
    
    def get_geo(self, lat:float, lng:float) -> Optional[Geo]: 
        for geo in self.geos.values():
            if geo.contains(lat, lng):
                return geo # pointer 
        return None
    
    def get_geo_id(self, lat:float, lng:float) -> Optional[int]:
        geo_loc = self.get_geo(lat,lng)  
        if geo_loc is None:
            return None 
        else:
            return geo_loc.id
    
    def remove_geo(self, geo_id:int):
        del self.geos[geo_id]
    
    def to_gpd(self):
        data = [geo.dict() for geo in self.geos.values()]
        data_mod = []
        for geo_dict in data:
            geo_dict['polygon'] = geo_dict['polygon'].wkt
            data_mod.append(geo_dict)
        geos_df = pd.DataFrame.from_records(data = data_mod)
        geos_df.rename(columns={'polygon':'geometry'}, inplace=True)
        return  geos_df
    
    def distance_geos(self, g1_id:int, g2_id:int):
        key = frozenset([g1_id, g2_id])
        if self.dist_geos is None:
            self.dist_geos = {}
        if key not in self.dist_geos: # try cache
            self.dist_geos[key] = self.geos[g1_id].distance(self.geos[g2_id])
        
        return self.dist_geos[key]

class Node(BaseModel):
    id: int 
    point: Point
    geo_id: Optional[int]
    node_type:str # in ['warehouse','drop']  

    # only valid for drops 
    warehouse_id : Optional[int]
    store_id : Optional[int]
    req_date: Optional[date]
    schedule_date: Optional[date]

    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)

    def __init__(self, lat:float, lng:float ,**data) -> None:
        super().__init__(point = Point(lng, lat), **data)

    @cached_property
    def sid(self) -> str:
        return self.node_type[:1] + str(self.id)

    @property
    def is_drop(self)-> bool:
        return self.node_type == 'drop'
    
    @property
    def is_warehouse(self)-> bool :
        return self.node_type == 'warehouse'

    @property
    def lng(self)->float:
        return self.point.x

    @property
    def lat(self)->float:
        return self.point.y

    @property
    def warehouse_sid(self)-> str:
        if self.is_warehouse:
            raise ValueError('This node is a Warehouse')
        return 'w' + str(self.warehouse_id)
            
class Route(BaseModel):
    id: Optional[int]
    nodes: List[Node]
    city: City
    price: Optional[float]
    
    @property
    def ft_size(self) -> float:
        return len(self.nodes)    

    @property
    def ft_size_drops(self) -> float:
        return len([node for node in self.nodes if node.is_drop])
    
    @property
    def ft_size_pickups(self) -> float:
        return len([node for node in self.nodes if node.is_warehouse])

    @property
    def ft_size_geo(self) -> float:
        return len(self.geos)
    
    @property
    def ft_inter_geo_dist(self) -> float:
        return sum([self.city.distance_geos(g1,g2) for  g1,g2 in it.combinations(self.geos, 2)])

    @property
    def geos(self)-> List[int]:
        return list(set([node.geo_id for node in self.nodes]))

    @cached_property
    def centroid(self) -> Point():
        return Polygon(self.nodes).centroid
    
    def ft_has_geo(self, geo_id:int) -> float:
        return 1.0 if geo_id in self.geos else 0

    def make_fake(self):
        raise NotImplemented('Not Implemented')
    
    def centroid_distance(self, point:Point) -> float:
        return self.centroid.distance(point)

    @classmethod
    def make_fake_best(cls, beta_features:Dict[str,float], ideal_route_len:int, 
                       beta_price:float, sim_beta_origin:float, geo_origin:Geo, 
                       price_by_node:float, city:City):
        
        nodes = []   
        whs_point = geo_origin.get_random_point()
        nodes.append((Node(id = 0,
                           lat =whs_point.y,
                           lng =whs_point.x,
                           node_type = 'warehouse'
                    )))

        for drop_id in range(ideal_route_len-1):
            drop_point = geo_origin.get_random_point()
            nodes.append(Node(id = drop_id,
                              lat =drop_point.y,
                              lng =drop_point.x,
                              node_type = 'drop',
                              warehouse_id = 0,
                        ))  
        price_total_route =  len(nodes)*price_by_node
        return cls(nodes = nodes,city=city, price= price_total_route )

    @classmethod
    def make_fake_worst(cls, beta_features:Dict[str,float], ideal_route_len:int, 
                       beta_price:float, sim_beta_origin:float, geo_origin:Geo, 
                       price_by_node:float, city:City):
        pass

INSTANCE_DF_COLUMNS = ['store_id', 'lon', 'is_warehouse', 
                       'lat', 'pickup_warehouse_id', 'req_date']
class RoutingInstance(BaseModel):
    nodes : List[Node]
    city: City
    
    # solution variables
    sol_cluster:   Optional[List[Dict[str,Union[int,object]]]] # [{node_sid:cluster(int)}]
    sol_arcs_list: Optional[List[Dict[str,Union[int,str]]]] # list arcs {gi,gj,cluster}
    sol_features:  Optional[Dict[str,Dict[int,float]]] # list dict{cluster: value_feature}
    sol_y : Optional[Dict[Tuple[Any, Any], int]] # [{node_sid:cluster(int)}]
    sol_z : Optional[Dict[Tuple[Any, Any, Any], int]] # [{():cluster(int)}]

    # markeplace_performance
    sol_time_performance: Optional[List[Dict[str,float]]]
    beta_dict: Optional[Dict[str,float]] # {'feat_name': beta_coef} 

    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)
    
    @cached_property
    def drops(self):
        return [node for node in self.nodes if node.is_drop]

    @cached_property
    def warehouses(self):
        return [node for node in self.nodes if node.is_warehouse]

    @property
    def drops_df(self):
        columns =['id', 'sid', 'lat', 'lng','warehouse_id' ,'store_id',
                  'req_date','schedule_date','geo_id']
        rows = []
        for node in self.drops:
            rows.append({key: getattr(node,key) for key in columns})
        return pd.DataFrame(rows)
    
    @property
    def warehouses_df(self):
        columns =[ 'id','sid' ,'lat','lng','geo_id']
        rows = []
        for node in self.warehouses:
            rows.append({key: getattr(node,key) for key in columns})
        return pd.DataFrame(rows)
    
    @cached_property
    def geos(self) -> List[Geo]:
        return list(self.city.geos.values())
    
    @cached_property
    def nodes_dict(self) -> Dict[str,Node]:
        return {node.sid:node for node in self.nodes}
    
    @property
    def mip_has_geo(self) -> Dict[str, float]:
        return {f'has_geo_{k[0]}_{k[1]}':v for k,v in self.sol_features['ft_has_geo'].items()}
    
    @property
    def mip_y(self) -> Dict[str, float]:
        return {f'y_{k[1]}_{k[0]}':v for (k,v) in self.sol_y.items()}
    
    @property
    def mip_z(self) -> Dict[str, float]:
        return {f'z_{k[0]}_{k[1]}_{k[2]}':v for (k,v) in self.sol_z.items()}

    @classmethod
    def load_instance(cls, instance_df = pd.DataFrame):
        """create an RoutingInstance based on a pandas dataframe with all the request and warehouse points

        Args:
            instance_df ([type], optional): [description]. Defaults to pd.DataFrame.

        Returns:
            RoutingInstance: [description]
        """    

        assert set(INSTANCE_DF_COLUMNS).issubset(set(instance_df.columns) ), 'instance dataframe is missing some columns'

        city_inst = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')

        warehouses_list = instance_df[instance_df.is_warehouse].to_dict(orient='records')
        drops_list      = instance_df[~instance_df.is_warehouse].to_dict(orient='records')
        
        nodes = []
        for wh_inst in warehouses_list:
            wh_id = wh_inst['pickup_warehouse_id']
            nodes.append(Node(id = wh_id,
                              lat = wh_inst['lat'],
                              lng = wh_inst['lon'],
                              geo_id = city_inst.get_geo_id(wh_inst['lat'], wh_inst['lon']),
                              node_type = 'warehouse'
                            ))

        for d_id, drop_inst in enumerate(drops_list):
            nodes.append(Node(id = d_id,
                              lat =drop_inst['lat'],
                              lng =drop_inst['lon'],
                              node_type = 'drop',
                              warehouse_id = drop_inst['pickup_warehouse_id'],
                              store_id = drop_inst['store_id'],
                              req_date = drop_inst['req_date'],
                              geo_id = city_inst.get_geo_id(drop_inst['lat'],drop_inst['lon'])
                        ))        
        # remove unused geos 
        used_geos = set()
        for node in nodes:
            used_geos.add(node.geo_id)
        
        for geo_id in list(city_inst.geos.keys()):
            if geo_id not in used_geos:
                city_inst.remove_geo(geo_id)
        
        sol_cluster = None
        # if solution in df then load
        if 'id_route' in instance_df.columns:
            sol_cluster = []
            for d_id, drop_inst in enumerate(drops_list):
                sol_cluster.append({'node_sid':f'd{d_id}',
                                    'cluster':int(drop_inst['id_route'])
                })
                sol_cluster.append({'node_sid':f'w{drop_inst["pickup_warehouse_id"]}',
                                    'cluster':     int(drop_inst['id_route'])
                })
            # remove duplicates
            sol_cluster = [dict(node_t) for node_t in {tuple(node.items()) for node in sol_cluster}]

        return cls(nodes = nodes, city = city_inst, sol_cluster = sol_cluster)
    
    def get_geo_by_id(self, geo_id: int) -> Geo:
        return self.city.geos[geo_id]

    def get_node_by_sid(self, sid:str):
        if sid not in self.nodes_dict:
            raise ValueError(f'{sid = } not in RoutingInstance.nodes')
        return self.nodes_dict[sid]

    def distance_geos(self, g1_id:int, g2_id:int):
        return self.city.distance_geos(g1_id, g2_id)

    def build_features(self, sol_cluster:Optional[List[Dict[str,int]]]=None):
        """infer features based on a solution `self.sol_cluster` : [{node_sid:cluster}]

        Raises:
            ValueError: [description]
        """
        #check if solution 
        if not self.sol_cluster and not sol_cluster: 
            raise ValueError('No Solution added yet, try using "routing_instance.get_warm_start()" method before')
        
        if self.sol_cluster and not sol_cluster:
            sol_cluster = deepcopy(self.sol_cluster)

        clusters_df = pd.DataFrame(sol_cluster)
        clusters_node_dict = clusters_df.groupby('cluster')['node_sid'].apply(list).to_dict()
        clusters_geo_dict = {c:set([self.get_node_by_sid(n).geo_id for n in nodes]) for (c,nodes) in clusters_node_dict.items()}
        clusters = list(clusters_node_dict.keys())

        # features
        y = {} 
        z = {}
        ft_size = {}
        ft_size_drops = {}
        ft_size_pickups = {}
        ft_has_geo = {}
        ft_size_geo = {}
        ft_inter_geo_dist = {}
        
        for node,c in it.product(self.nodes, clusters): 
            y[(node.sid,c)] = 1 if node.sid in clusters_node_dict[c] else 0

        for c,(g1,g2) in it.product(clusters, it.combinations(self.geos, 2)):
            z[(c,g1.id,g2.id)] = 1 if g1.id in clusters_geo_dict[c] and g2.id in clusters_geo_dict[c] else 0

        for c in clusters:
            route = Route(id = c, 
                          nodes = [self.get_node_by_sid(nsid) for nsid in clusters_node_dict[c]], 
                          city = self.city)
            ft_size[c]           = route.ft_size
            ft_size_drops[c]     = route.ft_size_drops
            ft_size_pickups[c]   = route.ft_size_pickups
            ft_size_geo[c]       = route.ft_size_geo
            ft_inter_geo_dist[c] = route.ft_inter_geo_dist
        
            for geo in self.geos:
                ft_has_geo[(c,geo.id)] = route.ft_has_geo(geo.id)

            self.sol_features = {'ft_size':ft_size, 
                                 'ft_size_drops':ft_size_drops,
                                 'ft_size_pickups':ft_size_pickups,
                                 'ft_has_geo':ft_has_geo,
                                 'ft_size_geo':ft_size_geo,
                                 'ft_inter_geo_dist':ft_inter_geo_dist }
            self.sol_y = y
            self.sol_z = z

            arcs_list = []
            for tuple_key in z.keys():
                if z[tuple_key] == 1:
                    arcs_list.append({'cluster':tuple_key[0],
                                     'geo_i': tuple_key[1],
                                     'geo_j':tuple_key[2]
                    })                
            self.sol_arcs_list = arcs_list # Optional[List[Dict[str,Union[int,str]]]]
            

    def build_warm_start(self, n_clusters:int, algorithm:str = 'KMeans'):
        
        # clustering
        allowed_algo = ['KMeans','SpectralClustering', 'AgglomerativeClustering']
        clusters = range(n_clusters)
        drops_df = self.drops_df.copy()
        
        if algorithm not in allowed_algo:
            raise ValueError(f'{algorithm} not in allowed values {allowed_algo}')

        cluster_model = eval(f'cluster.{algorithm}(n_clusters={n_clusters})')    
        drops_df['cluster'] = cluster_model.fit_predict(drops_df[['lat', 'lng']])

        warehouses_df = self.warehouses_df[['id','geo_id']]\
                            .rename(columns={'geo_id':'wh_geo_id', 
                                             'id':'wh_id'})
        drops_df = pd.merge(left=drops_df,right= warehouses_df, 
                            left_on='warehouse_id', right_on='wh_id', how='left')
        sol_cluster = []
        for drop in drops_df.itertuples():
            sol_cluster.append({'node_sid':f'd{drop.id}',
                                'cluster' :int(drop.cluster)
            })
            sol_cluster.append({'node_sid':f'w{drop.warehouse_id}',
                                'cluster' :int(drop.cluster)
            })
        # remove duplicates
        sol_cluster = [dict(node_t) for node_t in {tuple(node.items()) for node in sol_cluster}]
        self.sol_cluster = sol_cluster
        

    def load_solution_mip_vars(self, y:Dict[Tuple[Any, Any],mip.Var], z:Dict[Tuple[Any,Any,Any],mip.Var]) -> None:
        sol_cluster = []
        for tuple_key in y.keys():
            if y[tuple_key].x == 1:
                # if node in cluster
                sol_cluster.append({'node_sid': tuple_key[0],
                                    'cluster':int(tuple_key[1])
                })  
        self.sol_cluster = sol_cluster

        arcs_list = []
        for tuple_key in z.keys():
            if z[tuple_key].x == 1:
                arcs_list.append({'cluster':tuple_key[0],
                                  'geo_i': tuple_key[1],
                                  'geo_j':tuple_key[2]
                })                
        self.sol_arcs_list = arcs_list
    
    def load_markeplace_data(self, mkp_instance_df:pd.DataFrame) -> None:
        mkp_instance_df['acceptance_time_min'] =(   pd.to_datetime(mkp_instance_df['route_acceptance_timestamp'])
                                                  - pd.to_datetime(mkp_instance_df['route_creation_timestamp'])
                                                ).dt.total_seconds()/60 
        
        self.sol_time_performance = mkp_instance_df[['id_route','acceptance_time_min']].to_dict(orient='records')

    def fit_betas_time_based(self):
        sol_df = pd.DataFrame(self.sol_time_performance)
        features_df = pd.DataFrame({k:v for k,v in self.sol_features.items() if k != 'ft_has_geo'} )
        
        columns_ft_has_geo = {}    
        for geo in set([ geo for (clt, geo) in self.sol_features['ft_has_geo'].keys()]):
            columns_ft_has_geo[f'ft_has_geo_{geo}']={clt:val for (clt,geoi),val in self.sol_features['ft_has_geo'].items() if geoi == geo}
        ft_has_geo_df = pd.DataFrame(columns_ft_has_geo)
        features_df = pd.merge(left=features_df, right=ft_has_geo_df, left_index=True, right_index=True)

        features_df['id_route'] = features_df.index
        train_df = pd.merge(left = sol_df, right = features_df, how='left', on ='id_route')

        model = linear_model.Lasso(alpha=0.1)
        #linear_model.LassoLars(alpha=.1)
        #linear_model.Ridge(alpha=.5)
        x_df = train_df[train_df.columns.difference(['acceptance_time_min', 'id_route'])]
        model.fit(X = x_df , y = train_df['acceptance_time_min'] )
        
        # To print OLS summary  
        # from statsmodels.api import OLS
        # result = OLS(train_df['acceptance_time_min'],x_df).fit_regularized('sqrt_lasso')
        # with open('summary.txt', 'w') as fh:
        #     fh.write(OLS(train_df['acceptance_time_min'],x_df).fit().summary().as_text())
        # print(result.params)

        beta_dict = {col:model.coef_[i] for i,col in enumerate(x_df.columns)}
        self.beta_dict = beta_dict

    def get_cluster_df(self):
        node_list = []
        for node_c in self.sol_cluster:
            node = self.get_node_by_sid(node_c['node_sid'])
            node_list.append({ 
              'node_sid': node.sid,
              'lat': node.lat,
              'lng': node.lng,
              'geo_id': node.geo_id,
              'node_type': node.node_type,
              'cluster': node_c['cluster']
            })
        cluster_df = pd.DataFrame(node_list)       
        return cluster_df
    
    def get_inter_geo_df(self):
        arcs_list = []
        for arc in self.sol_arcs_list:
            geo_i = self.city.geos[arc['geo_i']]
            geo_j = self.city.geos[arc['geo_j']]
            line = LineString([geo_i.centroid, geo_j.centroid]).wkt
            arcs_list.append({'geo_i': arc['geo_i'],
                              'geo_j': arc['geo_j'],
                              'cluster': arc['cluster'],
                              'shape': line
            })
        return pd.DataFrame(arcs_list)
    
    def plot(self, file_name='plot_map.html'):
        # get data 
        geos_layer_df    = self.city.to_gpd()
        cluster_layer_df = self.get_cluster_df()
        inter_geo_df     = self.get_inter_geo_df()

        # build map
        out_map = KeplerGl(height=400, config=KEPLER_CONFIG)
        # load data
        out_map.add_data(data=geos_layer_df, name='geos')
        out_map.add_data(data=cluster_layer_df, name='cluster')
        out_map.add_data(data=inter_geo_df, name='inter_geo')

        out_map.save_to_html(file_name=file_name)


class Geodude(BaseModel):
    
    @staticmethod
    def run_opt_model(routing_instance:RoutingInstance, beta_dict:dict, max_time:int = 30 ):
        # ============================ # 
        # ==== optimization model ==== #
        # ============================ # 

        model = mip.Model(name = 'clustering')
        # Instance Parameters 
        n_clusters = 25 # max number of clusters ? TODO: there should be a Z* equivalent way of modeling this problem 
        clusters = range(n_clusters)

        # var declaration
        print('var declaration')
        y = {} # cluster variables 
        for node,c in it.product(routing_instance.nodes, clusters): # in cluster var
            y[(node.sid,c)] = model.add_var(var_type = mip.BINARY , name = f'y_{c}_{node.sid}')


        z = {} # distance variables 
        for c,(g1,g2) in it.product(clusters, it.combinations(routing_instance.geos, 2)): # unique combinations  
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
            
            for geo in routing_instance.geos:
                ft_has_geo[(c,geo.id)] = model.add_var(var_type = mip.BINARY , name = f'has_geo_{c}_{geo.id}')
            
        # ======================== #
        # ===== constraints ====== #
        # ======================== #
        print('adding cluster constraints')
        # Cluster Constraint
        for node in routing_instance.drops:
            # 0. demand satisfy
            model.add_constr(mip.xsum([y[(node.sid, c)] for c in clusters]) ==  1, name=f'cluster_fill_{node.sid}') # TODO SOS ? 

        for node,c in it.product(routing_instance.drops, clusters):
            # 1. pair drop,warehouse 
            model.add_constr(y[(node.sid, c)] <= y[(node.warehouse_sid, c)], name=f'pair_drop_warehouse_{node.sid}_{node.warehouse_sid}') 

        for c, wh in it.product(clusters, routing_instance.warehouses):
            # 2. remove unused nodes 
            model.add_constr(mip.xsum([y[(drop.sid, c)] for drop in routing_instance.drops if drop.warehouse_id == wh.id]) >= y[(wh.sid, c)], 
                            name=f'no_wh_if_no_need_to_c{c}_{wh.sid}') 

        print('adding size features constraints')
        # Size Features
        for c in clusters:
            # 2. cod ft_size
            model.add_constr(ft_size[c] == mip.xsum([y[(node.sid, c)] for node in routing_instance.nodes]), name=f'cod_ft_size_c{c}') 
            # 3. cod ft_size_drops
            model.add_constr(ft_size_drops[c] == mip.xsum([y[(node.sid, c)] for node in routing_instance.drops]), name=f'cod_ft_size_drops_c{c}') 
            # 4. cod ft_size_pickups
            model.add_constr(ft_size_pickups[c] == mip.xsum([y[(node.sid, c)] for node in routing_instance.warehouses]), name=f'cod_ft_size_pickups_c{c}') 

        # Geo Codifications
        print('adding geo cod constraints')
        M1 =  len(routing_instance.nodes)+1
        for c,geo in it.product(clusters,routing_instance.geos):
            # 5. cod min ft_has_geo 
            model.add_constr(M1 * ft_has_geo[(c,geo.id)] >= mip.xsum([y[node.sid,c] for node in routing_instance.nodes if node.geo_id == geo.id]), name=f'cod_ft_has_geo_min_{c}_{geo.id}') 
            # 6. cod max ft_has_geo 
            model.add_constr(     ft_has_geo[(c,geo.id)] <= mip.xsum([y[node.sid,c] for node in routing_instance.nodes if node.geo_id == geo.id]), name=f'cod_ft_has_geo_max_{c}_{geo.id}') 

        for c in clusters:
            # 7. cod ft_size_geos 
            model.add_constr(ft_size_geo[c] == mip.xsum([ft_has_geo[(c,geo.id)] for geo in routing_instance.geos]), name=f'cod_ft_size_geos_{c}_{geo.id}') 

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
            model.add_constr(ft_inter_geo_dist[c] == mip.xsum([z[(c,g1.id,g2.id)] * routing_instance.distance_geos(g1.id, g2.id) for g1,g2 in it.combinations(routing_instance.geos,2)]),
                            name=f'cod_ft_inter_geo_dist_{c}') 

        print('adding objective function')
        # objective function

        model.sense = mip.MINIMIZE
        model.objective = mip.xsum([  beta_dict['ft_size']          *ft_size[c] 
                                    + beta_dict['ft_size_drops']    *ft_size_drops[c]
                                    + beta_dict['ft_size_pickups']  *ft_size_pickups[c]
                                    + beta_dict['ft_size_geo']      *ft_size_geo[c] 
                                    + beta_dict['ft_inter_geo_dist']*ft_inter_geo_dist[c] 
                                    for c in clusters] + 
                                    [beta_dict.get(f'ft_has_geo_{geo.id}',0) * ft_has_geo[c,geo.id]
                                    for geo, c in it.product(routing_instance.geos,clusters)]                            
                                    )

        model.max_seconds = 60 * max_time # min 
        # get a warm start 
        routing_instance.build_warm_start(n_clusters)
        routing_instance.build_features()
        warm_start = routing_instance.mip_y | routing_instance.mip_z | routing_instance.mip_has_geo
        start_list = [(model.var_by_name(var_name), value_start) for (var_name, value_start) in warm_start.items()]

        print('validating start')
        model.start = start_list 
        model.emphasis = 2
        # model.validate_mip_start()

        print('optimization starting')
        model.optimize()

        solution_dict = {'y':  y,  
                        'ft_size' :  ft_size,
                        'ft_size_drops' :  ft_size_drops,
                        'ft_size_pickups' :  ft_size_pickups,
                        'ft_has_geo' :  ft_has_geo,
                        'ft_size_geo' :  ft_size_geo,
                        'ft_inter_geo_dist' : ft_inter_geo_dist,
                        }

        for c in clusters:
            print(f'{c = }, {ft_size_geo[c].x = }, {ft_size_drops[c].x = }, {ft_size_drops[c].x = }, {ft_size_pickups[c].x = }, {ft_inter_geo_dist[c].x = }')

        routing_instance.load_solution_mip_vars(y = y, z = z)
