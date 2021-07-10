# Data Models + Utilities

import itertools as it
import json
from copy import deepcopy
from datetime import date
from typing import Dict, List, Optional, Tuple, Union

import geopandas
import mip
import pandas as pd
from keplergl import KeplerGl
from numpy import product
from pydantic import BaseModel
from shapely.geometry import LineString, Point, Polygon, shape
from sklearn import linear_model
from sklearn import cluster

from constants import KEPLER_CONFIG


# city meta classes 
class Geo(BaseModel):
    id: Union[int,str]
    name: Optional[str]
    polygon: Polygon
    area: Optional[float]

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
    
    class Config:
        arbitrary_types_allowed = True


class City(BaseModel):
    id: Optional[int]
    name_city: Optional[str]
    geos: Dict[int, Geo]

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
    
    def get_geo(self, lat:float, lng:float) -> Geo: #TODO latlong to point 
        for geo in self.geos.values():
            if geo.contains(lat, lng):
                return geo # pointer 
        return None
    
    def get_geo_id(self, lat:float, lng:float) -> int:
        geo_loc = self.get_geo(lat,lng)  
        if geo_loc is None:
            return None 
        else:
            return geo_loc.id
    
    def remove_geo(self, geo_id:str):
        del self.geos[geo_id]
    
    def to_gpd(self):
        data = [geo.dict() for geo in self.geos.values()]
        geos_df = pd.DataFrame.from_records(data = data)
        geos_df.rename(columns={'polygon':'geometry'}, inplace=True)
        return  geopandas.GeoDataFrame(geos_df) 

# instance classes 
class Warehouse(BaseModel):
    id: int 
    lat: float
    lng: float
    geo_id: Optional[int]

    @property
    def sid(self) -> str:
        return 'w' + str(self.id)
    @property
    def point(self):
        return Point(self.lng,self.lat)
class Drop(BaseModel):
    id: int 
    lat: float
    lng: float
    warehouse_id : int
    store_id : int 
    req_date: date
    schedule_date: Optional[date]
    geo_id: Optional[int]

    @property
    def sid(self)-> str:
        return 'd' + str(self.id)
    
    @property
    def point(self):
        return Point(self.lng,self.lat)

    @property
    def warehouse_sid(self)-> str:
        return 'w' + str(self.warehouse_id)
    
INSTANCE_DF_COLUMNS = ['store_id', 'lon', 'is_warehouse', 
                       'lat', 'pickup_warehouse_id', 'req_date']

class OptInstance(BaseModel):
    warehouses_dict: Dict[int,Warehouse]
    drops_dict: Dict[int,Drop]
    city: City
    
    # cache_distance 
    dist_geos: Optional[Dict[frozenset, float]]

    # solution variables
    sol_cluster:   Optional[List[Dict[str,Union[int,str]]]] # [{node_sid:cluster(int)}]
    sol_arcs_list: Optional[List[Dict[str,Union[int,str]]]] # list arcs {gi,gj,cluster}
    sol_features:  Optional[Dict[str,Dict[int,float]]] # list dict{cluster: value_feature}
    sol_y : Optional[List[Dict[str,int]]] # [{node_sid:cluster(int)}]
    sol_z : Optional[List[Dict[str,int]]] # [{node_sid:cluster(int)}]

    # markeplace_performance
    sol_time_performance: Optional[List[Dict[str,float]]]
    beta_dict: Optional[Dict[str,float]] # {'feat_name': beta_coef} 

    class Config:
        arbitrary_types_allowed = True
    
    @property
    def nodes(self):
        return list(self.drops_dict.values()) + list(self.warehouses_dict.values())

    @property
    def drops(self):
        return list(self.drops_dict.values())

    @property
    def warehouses(self):
        return list(self.warehouses_dict.values())

    @property
    def drops_df(self):
        return pd.DataFrame([drop.dict() for drop in self.drops])
    
    @property
    def warehouses_df(self):
        return pd.DataFrame([wh.dict() for wh in self.warehouses])
    
    @property
    def geos(self) -> List[Geo]:
        return list(self.city.geos.values())
    
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
        """create an OptInstance based on a pandas dataframe with all the request and warehouse points

        Args:
            instance_df ([type], optional): [description]. Defaults to pd.DataFrame.

        Returns:
            OptInstance: [description]
        """    

        assert set(INSTANCE_DF_COLUMNS).issubset(set(instance_df.columns) ), 'instance dataframe is missing some columns'

        city_inst = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')

        warehouses_list = instance_df[instance_df.is_warehouse].to_dict(orient='records')
        drops_list      = instance_df[~instance_df.is_warehouse].to_dict(orient='records')
        
        warehouses_dict = {}
        for wh_inst in warehouses_list:
            wh_id = wh_inst['pickup_warehouse_id']
            warehouses_dict[int(wh_id)] = Warehouse(id = wh_id,
                                                    lat = wh_inst['lat'],
                                                    lng = wh_inst['lon'],
                                                    geo_id = city_inst.get_geo_id(wh_inst['lat'], wh_inst['lon'])
                                                   )

        drops_dict = {}
        for d_id, drop_inst in enumerate(drops_list):
            drops_dict[int(d_id)] = Drop(id = d_id,
                                         lat =drop_inst['lat'],
                                         lng =drop_inst['lon'],
                                         warehouse_id = drop_inst['pickup_warehouse_id'],
                                         store_id = drop_inst['store_id'],
                                         req_date = drop_inst['req_date'],
                                         geo_id = city_inst.get_geo_id(drop_inst['lat'],drop_inst['lon'])
                                        )        
        # remove unused geos 
        used_geos = set()
        for node in it.chain(warehouses_dict.values(),drops_dict.values()):
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

        return cls(warehouses_dict = warehouses_dict, drops_dict= drops_dict, city = city_inst, sol_cluster = sol_cluster)
    
    def get_geo_by_id(self, geo_id: int) -> Geo:
        return self.city.geos[geo_id]

    def get_node_by_sid(self, sid:str):
        if 'w' in sid:
            return self.warehouses_dict[int(sid[1:])]
        elif 'd' in sid:
            return self.drops_dict[int(sid[1:])]
        else:
            return None
    
    def distance_geos(self, g1_id:int, g2_id:int):
        key = frozenset([g1_id, g2_id])
        if self.dist_geos is None:
            self.dist_geos = {}
        if key not in self.dist_geos: # try cache
            self.dist_geos[key] = self.city.geos[g1_id].distance(self.city.geos[g2_id])
        
        return self.dist_geos[key]



    def build_features(self, sol_cluster:Optional[List[Dict[str,int]]]=None):
        """infer features based on a solution `self.sol_cluster` : [{node_sid:cluster}]

        Raises:
            ValueError: [description]
        """
        #check if solution 
        if not self.sol_cluster and not sol_cluster: 
            raise ValueError('No Solution added yet, try using "opt_instance.get_warm_start()" method before')
        
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
            ft_size[c]           = len(clusters_node_dict[c])
            ft_size_drops[c]     = len([nd for nd in clusters_node_dict[c] if 'd' in nd])
            ft_size_pickups[c]   = len([nd for nd in clusters_node_dict[c] if 'w' in nd])
            ft_size_geo[c]       = len(clusters_geo_dict[c])
            
            
            # unpack keys (it doesn't work in one line, no clue why )
            # sum([self.distance_geos(g1,g2)  for (cz,g1,g2) in z.keys() if z[(cz,g1,g2)]==1 and cz == c ])
            keys = []    
            for key in z.keys():
                if key[0] == c and z[key] == 1:
                    keys.append(key)
            ft_inter_geo_dist[c] = sum([ self.distance_geos(key[1],key[2]) for key in keys ])
        
            for geo in self.geos:
                ft_has_geo[(c,geo.id)] = 1 if geo.id in clusters_geo_dict[c] else 0

            self.sol_features = {'ft_size':ft_size, 
                                 'ft_size_drops':ft_size_drops,
                                 'ft_size_pickups':ft_size_pickups,
                                 'ft_has_geo':ft_has_geo,
                                 'ft_size_geo':ft_size_geo,
                                 'ft_inter_geo_dist':ft_inter_geo_dist }
            self.sol_y = y
            self.sol_z = z

    def build_warm_start(self, n_clusters:int, algorithm:str = 'KMeans') -> Dict[str,float]:
        
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
        

    def load_solution_mip_vars(self, y:Dict[Tuple,mip.Var], z:Dict[Tuple,mip.Var]) -> None:
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
              'node_sid': node_c['node_sid'],
              'lat': node.lat,
              'lng': node.lng,
              'geo_id': node.geo_id,
              'node_type': 'drop' if type(node)==Drop else 'warehouse',
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
