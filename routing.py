from __future__ import annotations
# Data Models + Utilities

import itertools as it
import json
import random
from copy import deepcopy
from datetime import date
from functools import cached_property
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import mip
import pandas as pd
from keplergl import KeplerGl
from pandas.core.frame import DataFrame
from pydantic import BaseModel
from shapely.geometry import LineString, Point, Polygon, shape
from sklearn import cluster  # used in eval

from constants import KEPLER_CONFIG, BETA_INIT, ROUTE_FEATURES


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
    
    def __eq__(self, other: Any) -> bool:
        if type(other) == Geo:
            return self.id == other.id
        else: 
            return False
class City(BaseModel):
    id: Optional[int]
    name_city: Optional[str]
    geos: Dict[int, Geo]
    
    # cache_distance 
    dist_geos: Optional[Dict[frozenset, float]]

    @property
    def geos_list(self) -> List[Geo]:
        return list(self.geos.values())
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
    
    def get_geo_from_name(self, geo_name:str) -> Optional[Geo]:
        for geo in self.geos.values():
            if geo.name is not None and geo.name.casefold() == geo_name.casefold():
                return geo 
        return None 

    def get_geo_from_latlong(self, lat:float, lng:float) -> Optional[Geo]: 
        for geo in self.geos.values():
            if geo.contains(lat, lng):
                return geo # pointer 
        return None
    
    def get_geo_id_from_latlong(self, lat:float, lng:float) -> Optional[int]:
        geo_loc = self.get_geo_from_latlong(lat,lng)  
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
    geo_id: int
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
    price: Optional[float] # total price for all the route
    
    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)
    
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

    def ft_has_geo(self, geo_id:int) -> float:
        return 1.0 if geo_id in self.geos else 0
    
    @cached_property
    def nodes_dict(self):
        return {node.sid:node for node in self.nodes}

    @cached_property
    def geos(self)-> List[int]:
        return list(set([node.geo_id for node in self.nodes]))

    @cached_property
    def arc_list(self) -> List[Tuple[Geo,Geo]]:
        return [(g1,g2) for g1,g2 in it.combinations(self.city.geos_list, 2) if g1.id in self.geos and g2.id in self.geos]

    @cached_property
    def centroid(self) -> Point:
        return Polygon([n.point for n in self.nodes]).centroid
    
    def has_node_sid(self, sid:str)-> bool:
        return sid in self.nodes_dict.keys()

    def make_fake(self):
        raise NotImplemented('Not Implemented')
    
    def centroid_distance(self, point:Point) -> float:
        return self.centroid.distance(point)
    
    def get_features_dict(self, features:List[str] = ROUTE_FEATURES) -> Dict[str, float]:
        feat_dict = { feat:getattr(self,feat) for feat in features if 'has_geo' not in feat }
        feat_dict.update({feat:self.ft_has_geo(int(feat.split('_')[-1])) for feat in features if 'has_geo' in feat})
        return feat_dict 

    @classmethod
    def make_fake_best(cls, beta_features:Dict[str,float], ideal_route_len:int, 
                       beta_price:float, sim_beta_origin:float, geo_origin:Geo, 
                       price_by_node:float, city:City) -> Route:
        
        nodes = []   
        whs_point = geo_origin.get_random_point()
        nodes.append((Node(id = 0,
                           lat =whs_point.y,
                           lng =whs_point.x,
                           node_type = 'warehouse',
                           geo_id = city.get_geo_id_from_latlong(whs_point.y, whs_point.x),
                    )))

        for drop_id in range(ideal_route_len-1):
            drop_point = geo_origin.get_random_point()
            nodes.append(Node(id = drop_id,
                              lat =drop_point.y,
                              lng =drop_point.x,
                              node_type = 'drop',
                              warehouse_id = 0,
                              geo_id = city.get_geo_id_from_latlong(drop_point.y, drop_point.x),
                        ))  
        price_total_route =  len(nodes)*price_by_node
        return cls(nodes = nodes,city=city, price= price_total_route )

    @classmethod
    def make_fake_worst(cls, beta_features:Dict[str,float], ideal_route_len:int, 
                         beta_price:float, sim_beta_origin:float, geo_origin:Geo, 
                         price_by_node:float, city:City) -> Route:

        num_nodes =  2 * ideal_route_len # this is guaranteed to be <0 
        num_warehouses =  5 # realistic highest number of warehouses 

        negative_betas = {k:v for k,v in sorted(beta_features.items(),key=lambda item:item[1]) if v<0 and 'ft_has_geo' in k}
        negative_betas_cycle = it.cycle(negative_betas.items())
        
        nodes = []
        for drop_id in range(num_nodes):
            beta_name,_ = next(negative_betas_cycle)
            geo_id = int(beta_name.split('_')[-1])
            drop_point = city.geos[geo_id].get_random_point()
            nodes.append(Node(id = drop_id,
                              lat =drop_point.y,
                              lng =drop_point.x,
                              node_type = 'drop',
                              warehouse_id = 0,
                              geo_id = city.get_geo_id_from_latlong(drop_point.y, drop_point.x),                              
                        ))

        for wh_id in range(num_warehouses):
            beta_name,_ = next(negative_betas_cycle)
            geo_id = int(beta_name.split('_')[-1])
            whs_point = city.geos[geo_id].get_random_point()
            nodes.append(Node(id = drop_id,
                              lat = whs_point.y,
                              lng = whs_point.x,
                              node_type = 'warehouse',
                              warehouse_id = 0,
                              geo_id = city.get_geo_id_from_latlong(whs_point.y, whs_point.x),
                        ))
        price_total_route =  len(nodes)*price_by_node
        return cls(nodes = nodes,city=city, price= price_total_route )

class RoutingSolution(BaseModel):
    routes: List[Route]
    city: City 

    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)
    
    @classmethod
    def from_df(cls, solved_instance_df:DataFrame, city:City):

        warehouses_list = solved_instance_df[solved_instance_df['is_warehouse']].to_dict(orient='records')
         
        warehouses_dict:Dict[int, Node] = {}
        for wh_inst in warehouses_list:
            wh_id = wh_inst['pickup_warehouse_id']
            warehouses_dict[wh_id] = Node(id = wh_id,
                                        lat = wh_inst['lat'],
                                        lng = wh_inst['lon'],
                                        geo_id = city.get_geo_id_from_latlong(wh_inst['lat'], wh_inst['lon']),
                                        node_type = 'warehouse'
                                        )
        
        route_solution_list: List[Route] = []
        id_node = 0 
        for id_route in solved_instance_df['id_route'].dropna().unique():
            drops_list = solved_instance_df[~(solved_instance_df['is_warehouse']) & 
                                              (solved_instance_df['id_route'] == id_route) 
                                            ].to_dict(orient='records')        
            route = Route(id=id_route, nodes=list(), city = city)
            for drop_inst in drops_list:
                route.nodes.append(Node(id = id_node,
                                        lat =drop_inst['lat'],
                                        lng =drop_inst['lon'],
                                        node_type = 'drop',
                                        warehouse_id = drop_inst['pickup_warehouse_id'],
                                        store_id = drop_inst['store_id'],
                                        req_date = drop_inst.get('req_date',None),
                                        geo_id = city.get_geo_id_from_latlong(drop_inst['lat'],drop_inst['lon'])
                                    )) 

                if not route.has_node_sid(f"w{drop_inst['pickup_warehouse_id']}"):
                    route.nodes.append(warehouses_dict[drop_inst['pickup_warehouse_id']])  
                id_node+=1
        
            route_solution_list.append(route)    
        return cls(routes=route_solution_list, city=city)

    @property
    def features_df(self)-> pd.DataFrame:
        features = ['ft_size','ft_size_drops','ft_size_pickups','ft_size_geo','ft_inter_geo_dist']
        routes_feat_list:List[Dict[str, float]] = []
        for route in self.routes:
            route_features = {feat:getattr(route, feat) for feat in features}
            route_features.update({f'ft_has_geo_{geo_id}':route.ft_has_geo(geo_id) for geo_id in self.city.geos.keys()})
            route_features['id_route'] = route.id 
            routes_feat_list.append(route_features)
        return pd.DataFrame(routes_feat_list)

    @property
    def mip_has_geo(self) -> Dict[str, float]:
        ft_has_geo = {}
        for route,geo in it.product(self.routes, self.city.geos_list):
            ft_has_geo[f'has_geo_{route.id}_{geo.id}'] = route.ft_has_geo(geo.id) #return float
        return ft_has_geo
    
    @property
    def mip_y(self) -> Dict[str, float]:
        mip_y = {}
        for route, node_sid in it.product(self.routes, self.all_nodes_sid):
            mip_y[f'y_{route.id}_{node_sid}'] = 1.0 if route.has_node_sid(node_sid) else 0.0
        return mip_y
    
    @property
    def mip_z(self) -> Dict[str, float]:
        mip_z = {}
        for route, (geo_i, geo_j) in it.product(self.routes,it.combinations(self.city.geos_list, 2)):
            mip_z[f'z_{route.id}_{geo_i.id}_{geo_j.id}'] = 1.0 if (geo_i,geo_j) in route.arc_list else 0.0 
        return mip_z
    
    @cached_property
    def all_nodes_sid(self) -> List[str]:
        nodes = list((it.chain.from_iterable([route.nodes for route in self.routes])))
        nodes_sid = list(set([node.sid for node in nodes]))
        return nodes_sid

    @cached_property
    def routes_dict(self) -> Dict[int, Route]:
        return {route.id:route for route in self.routes}

    @property           
    def cluster_df(self) -> pd.DataFrame:
        node_list = []
        for route in self.routes: 
            for node in route.nodes:
                node_list.append({ 
                'node_sid': node.sid,
                'lat': node.lat,
                'lng': node.lng,
                'geo_id': node.geo_id,
                'node_type': node.node_type,
                'cluster': route.id
                })
        cluster_df = pd.DataFrame(node_list)       
        return cluster_df
    
    @property
    def inter_geo_df(self)-> pd.DataFrame:
        arcs_list = []
        for route in self.routes:
            for geo_i, geo_j in route.arc_list:
                line = LineString([geo_i.centroid, geo_j.centroid]).wkt
                arcs_list.append({'geo_i': geo_i.id,
                                  'geo_j': geo_j.id,
                                  'cluster': route.id,
                                  'shape': line
                })
        return pd.DataFrame(arcs_list)
    
    def plot(self, file_name='plot_map.html'):
        # get data 
        geos_layer_df    = self.city.to_gpd()
        cluster_layer_df = self.cluster_df
        inter_geo_df     = self.inter_geo_df

        # build map
        out_map = KeplerGl(height=400, config=KEPLER_CONFIG)
        # load data
        out_map.add_data(data=geos_layer_df, name='geos')
        out_map.add_data(data=cluster_layer_df, name='cluster')
        out_map.add_data(data=inter_geo_df, name='inter_geo')

        out_map.save_to_html(file_name=file_name)



# TODO change this for a pandera schema
# https://pandera.readthedocs.io/en/stable/schema_models.html
INSTANCE_DF_COLUMNS = ['store_id', 'lon', 'is_warehouse', 
                       'lat', 'pickup_warehouse_id']
CITY_SCL = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')

class RoutingInstance(BaseModel):
    nodes : List[Node]
    city: City
    solution: Optional[RoutingSolution]

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
    

    @classmethod
    def from_df(cls, instance_df = pd.DataFrame, city_inst:City = CITY_SCL ):
        """create an RoutingInstance based on a pandas dataframe with all the request and warehouse points

        Args:
            instance_df ([type], optional): [description]. Defaults to pd.DataFrame.

        Returns:
            RoutingInstance: [description]
        """    

        assert set(INSTANCE_DF_COLUMNS).issubset(set(instance_df.columns) ), 'instance dataframe is missing some columns'

        warehouses_list = instance_df[instance_df.is_warehouse].to_dict(orient='records')
        drops_list      = instance_df[~instance_df.is_warehouse].to_dict(orient='records')
        
        nodes:List[Node] = []
        for wh_inst in warehouses_list:
            wh_id = wh_inst['pickup_warehouse_id']
            nodes.append(Node(id = wh_id,
                              lat = wh_inst['lat'],
                              lng = wh_inst['lon'],
                              geo_id = city_inst.get_geo_id_from_latlong(wh_inst['lat'], wh_inst['lon']),
                              node_type = 'warehouse'
                            ))

        for d_id, drop_inst in enumerate(drops_list):
            drop_inst['id'] = d_id # modify dict on the fly 
            nodes.append(Node(id = d_id,
                              lat =drop_inst['lat'],
                              lng =drop_inst['lon'],
                              node_type = 'drop',
                              warehouse_id = drop_inst['pickup_warehouse_id'],
                              store_id = drop_inst['store_id'],
                              req_date = drop_inst.get('req_date',None),
                              geo_id = city_inst.get_geo_id_from_latlong(drop_inst['lat'],drop_inst['lon'])
                        ))  

        # remove unused geos 
        used_geos = set()
        for node in nodes:
            used_geos.add(node.geo_id)
        
        for geo_id in list(city_inst.geos.keys()):
            if geo_id not in used_geos:
                city_inst.remove_geo(geo_id)
        
        # if solution in df then load
        if 'id_route' not in instance_df.columns:
            solution = None
        else: 
            route_solution_list: List[Route] = []
            for id_route in instance_df['id_route'].dropna().unique():
                drops_ids = [drop_inst['id'] for drop_inst in drops_list if drop_inst['id_route'] == id_route]
                whs_ids   = [drop_inst['pickup_warehouse_id'] for drop_inst in drops_list if drop_inst['id_route'] == id_route]
                route_nodes = [node for node in nodes if (node.id in drops_ids and node.node_type == 'drop') or 
                                                         (node.id in whs_ids and node.node_type == 'warehouse')]
                route_solution_list.append(Route(id=int(id_route) ,city=city_inst, nodes=route_nodes))
            solution = RoutingSolution(routes=route_solution_list, city=city_inst)
        
        return cls(nodes = nodes, city = city_inst, solution=solution)
    
    def get_geo_by_id(self, geo_id: int) -> Geo:
        return self.city.geos[geo_id]

    def get_node_by_sid(self, sid:str):
        if sid not in self.nodes_dict:
            raise ValueError(f'{sid = } not in RoutingInstance.nodes')
        return self.nodes_dict[sid]

    def distance_geos(self, g1_id:int, g2_id:int):
        return self.city.distance_geos(g1_id, g2_id)

    def build_warm_start(self, n_clusters:int, 
                         algorithm:Literal['KMeans','SpectralClustering', 'AgglomerativeClustering'] = 'KMeans') -> RoutingSolution:
        
        """Get a RoutingSolution for this RoutingInstance based on a heuristic approach

        Args:
            n_clusters (int): umber of routers or clusters needed 
            algorithm (Literal[, optional): Clustering Algorithm (). Defaults to 'KMeans'.

        Returns:
            RoutingSolution: [description]
        """                
        
        clusters = range(n_clusters)
        drops_df = self.drops_df.copy()

        cluster_model = eval(f'cluster.{algorithm}(n_clusters={n_clusters})')    
        drops_df['cluster'] = cluster_model.fit_predict(drops_df[['lat', 'lng']])

        warehouses_df = self.warehouses_df[['id','geo_id']]\
                            .rename(columns={'geo_id':'wh_geo_id', 
                                             'id':'wh_id'})
        drops_df = pd.merge(left=drops_df,right= warehouses_df, 
                            left_on='warehouse_id', right_on='wh_id', how='left')

        routes_sid:Dict[int, List[str]] = defaultdict(list)        
        for drop in drops_df.itertuples():
            routes_sid[int(drop.cluster)].append(f'd{drop.id}')
            routes_sid[int(drop.cluster)].append(f'w{drop.warehouse_id}')                                

        routes:List[Route] = []
        for route_id in drops_df['cluster'].unique():
            nodes_list = [self.get_node_by_sid(sid) for sid in set(routes_sid[route_id])]
            routes.append(Route(id=route_id, city =  self.city, nodes = nodes_list))

        return RoutingSolution(routes=routes, city=self.city)

class BetaMarket(BaseModel):
    beta_dict:Dict[str,float]

    @classmethod
    def default(cls):
        return cls(beta_dict = BETA_INIT)

    @property
    def dict(self):
        return self.beta_dict        
class Geodude(BaseModel):
    routing_instance:RoutingInstance
    beta_market:BetaMarket

    def solve(self, max_time_min:int = 30, n_clusters:int = 25 ):
        # max number of clusters ? TODO: there should be a Z* equivalent way of modeling this problem 
        clusters = range(n_clusters)
        routing_instance = self.routing_instance
        beta_dict = self.beta_market.dict
        # ============================ # 
        # ==== optimization model ==== #
        # ============================ # 
        model = mip.Model(name = 'clustering')
        # Instance Parameters 
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

        model.max_seconds = 60 * max_time_min 
        
        # get a warm start 
        warm_start_sol = routing_instance.build_warm_start(n_clusters)
        warm_start_dict = warm_start_sol.mip_y | warm_start_sol.mip_z | warm_start_sol.mip_has_geo
        start_list:List[Tuple[mip.Var, float]] = []
        for (var_name, value_start) in warm_start_dict.items():
            w_var = model.var_by_name(var_name)
            if w_var:
                start_list.append((w_var, value_start))
            else:
                raise ValueError('warm start is missing some variables')
        y_vars =  [var.name for var in y.values()] 
        assert sorted(y_vars) == sorted(list(warm_start_sol.mip_y.keys())), 'error'
        z_vars =  [var.name for var in z.values()] 
        assert sorted(z_vars) == sorted(list(warm_start_sol.mip_z.keys())), 'error'
        has_geo_vars =  [var.name for var in ft_has_geo.values()] 
        assert sorted(has_geo_vars) == sorted(list(warm_start_sol.mip_has_geo.keys())), 'error'

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

        return self.build_routing_sol(routing_instance=routing_instance, y = y)
    
    def build_routing_sol(self,routing_instance:RoutingInstance, y:Dict[Tuple[str, Any],mip.Var]) -> RoutingSolution:
        """based on y dict get a RoutingSolution Object

        Args:
            routing_instance (RoutingInstance): [description]
            y (Dict[Tuple[str, Any],mip.Var]): [description]

        Returns:
            RoutingSolution: [description]
        """        
        
        routes_sid: Dict[int, List[str]] =  defaultdict(list)
        for tuple_key in y.keys():
            if y[tuple_key].x == 1:
                routes_sid[int(tuple_key[1])].append(tuple_key[0])
        
        routes:List[Route] = [] 
        for route_id in routes_sid.keys():
            nodes = [routing_instance.get_node_by_sid(sid) for sid in set(routes_sid[route_id])]
            routes.append(Route(id = route_id, nodes=nodes, city = routing_instance.city))

        return RoutingSolution(routes=routes, city=routing_instance.city)

