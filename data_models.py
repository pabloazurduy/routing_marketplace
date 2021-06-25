# Data Models + Utilities

from numpy import product
import pandas as pd 
from pydantic import BaseModel 
from datetime import date 
from typing import List, Optional, Dict, Tuple, Union
import json
from shapely.geometry import shape, Point, Polygon, LineString
import geopandas
from keplergl import KeplerGl
from constants import KEPLER_CONFIG
import mip
import itertools as it
from sklearn.cluster import KMeans

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

class Solution(BaseModel):
    y: Dict[Tuple,mip.Var]
    z: Dict[Tuple,mip.Var]
    features = Optional[List[Dict[Tuple,mip.Var]]]
    class Config:
        arbitrary_types_allowed = True

    def clusters_df(self):
        cluster_list = []
        for tuple_key in self.y.keys():
            if self.y[tuple_key].x == 1:
                # if node in cluster
                cluster_list.append({'node_sid': tuple_key[0],
                                     'node_type': tuple_key[0][0],
                                     'node_id':int(tuple_key[0][1:]),
                                     'cluster':tuple_key[1] 
                })
            else:
                pass
        cluster_df = pd.DataFrame(cluster_list)
        return cluster_df
    
    def arcs_df(self):
        arcs_list = []
        for tuple_key in self.z.keys():
            if self.z[tuple_key].x == 1:
                arcs_list.append({'cluster':tuple_key[0],
                                  'geo_i': tuple_key[1],
                                  'geo_j':tuple_key[2]
                })                
        arcs_df = pd.DataFrame(arcs_list)
        return arcs_df

class OptInstance(BaseModel):
    warehouses_dict: Dict[int,Warehouse]
    drops_dict: Dict[int,Drop]
    city: City

    solution: Optional[Solution] # name_variable: dict with variables

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
                    
        return cls(warehouses_dict = warehouses_dict, drops_dict= drops_dict, city = city_inst)
    
    def get_warm_start(self, n_clusters:int) -> Dict[str,float]:
        # KMEANS clustering and assignation
        clusters = range(n_clusters)
        drops_df = self.drops_df.copy()
        drops_df['cluster'] = KMeans(n_clusters=n_clusters).fit_predict(drops_df[['lat', 'lng']])
        
        warehouses_df = self.warehouses_df[['id','geo_id']]\
                            .rename(columns={'geo_id':'wh_geo_id', 
                                             'id':'wh_id'})
        drops_df = pd.merge(left=drops_df,right= warehouses_df, 
                            left_on='warehouse_id', right_on='wh_id', how='left')
        
        geo_pairs = [ (g1.id,g2.id) for (g1,g2) in  it.combinations(self.geos, 2) ]
        
        # star all variables in 0 
        y = {f'y_{c}_{node.sid}':0 for (c, node) in it.product(range(n_clusters), self.nodes)}
        z = {f'z_{c}_{g1}_{g2}' :0 for c,(g1,g2) in it.product(clusters, geo_pairs)}
        has_geo = {f'has_geo_{c}_{geo.id}':0 for (c,geo) in it.product(clusters,self.geos)}
        
        for clt in drops_df['cluster'].unique():
            sub_drops = drops_df[drops_df['cluster']==clt]
            # drops & warehouses
            drops = sub_drops['id'].unique()
            warehouses = sub_drops['warehouse_id'].unique()
            y.update({f'y_{clt}_d{id}':1.0 for id in drops})
            y.update({f'y_{clt}_w{id}':1.0 for id in warehouses})
            # geos 
            geos_cluster = set(sub_drops['geo_id'].unique()).union(set(sub_drops['wh_geo_id'].unique()))
            has_geo.update({f'has_geo_{clt}_{geo}':1.0 for geo in geos_cluster})
            z.update({f'z_{clt}_{g1}_{g2}':1.0 for (g1,g2) in geo_pairs if g1 in geos_cluster and g2 in geos_cluster })

        return {**y, **z, **has_geo}

    def get_cluster_df(self):
        cluster_df = self.solution.clusters_df()
        
        node_list = []
        for node_sid in cluster_df.node_sid.unique():
            node = self.get_node_sid(node_sid)
            node_list.append({ 
              'node_sid': node_sid,
              'lat': node.lat,
              'lng': node.lng,
              'geo_id': node.geo_id,
              'warehouse': node.warehouse_id if type(node)==Drop else None,
            })
        node_df = pd.DataFrame(node_list)       
        cluster_df = pd.merge(left = cluster_df, right = node_df, 
                                on='node_sid', how='left')
        return cluster_df
    
    def get_inter_geo_df(self):
        arc_df = self.solution.arcs_df()

        arcs_shapes = []
        for arc in arc_df.itertuples():
            geo_i = self.city.geos[arc.geo_i]
            geo_j = self.city.geos[arc.geo_j]
            arcs_shapes.append(LineString([geo_i.centroid, geo_j.centroid]).wkt)
        arc_df['shape'] = arcs_shapes
        return arc_df

    def get_node_sid(self, sid:str):
        if 'w' in sid:
            return self.warehouses_dict[int(sid[1:])]
        elif 'd' in sid:
            return self.drops_dict[int(sid[1:])]
        else:
            return None
    
    def plot(self, file_name='plot_map.html'):
        # get data 
        geos_layer_df = self.city.to_gpd()
        cluster_layer_df = self.get_cluster_df()
        inter_geo_df = self.get_inter_geo_df()

        # build map
        out_map = KeplerGl(height=400, config=KEPLER_CONFIG)
        # load data
        out_map.add_data(data=geos_layer_df, name='geos')
        out_map.add_data(data=cluster_layer_df, name='cluster')
        out_map.add_data(data=inter_geo_df, name='inter_geo')

        out_map.save_to_html(file_name=file_name)