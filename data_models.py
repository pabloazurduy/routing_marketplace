# Data Models + Utilities

import pandas as pd 
from pydantic import BaseModel 
from datetime import date 
from typing import List, Optional, Dict, Tuple, Union
import json
from shapely.geometry import shape, Point, Polygon
import geopandas
from keplergl import KeplerGl
from constants import KEPLER_CONFIG
import mip

# city meta classes 
class Geo(BaseModel):
    id: Union[int,str]
    name: Optional[str]
    polygon: Polygon
    area: Optional[float]
    class Config:
        arbitrary_types_allowed = True

    def contains(self, lat:float, lng:float) -> bool:
        point = Point(lng,lat)
        return self.polygon.contains(point)

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
                                                    lat =wh_inst['lat'],
                                                    lng =wh_inst['lon'],
                                                    geo_id =  city_inst.get_geo_id(wh_inst['lat'], wh_inst['lon'])
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
            
        return cls(warehouses_dict = warehouses_dict, drops_dict= drops_dict, city = city_inst)
    
    def get_solution_df(self):
        cluster_df = self.solution.clusters_df()
        
        node_list = []
        for node_sid in cluster_df.node_sid.unique():
            node_list.append({ 
              'node_sid': node_sid,
              'lat': self.get_node_sid(node_sid).lat,
              'lng': self.get_node_sid(node_sid).lng,
            })
        node_df = pd.DataFrame(node_list)       
        solution_df = pd.merge(left = cluster_df, right = node_df, 
                                on='node_sid', how='left')
        return solution_df
    
    def get_node_sid(self, sid:str):
        if 'w' in sid:
            return self.warehouses_dict[int(sid[1:])]
        elif 'd' in sid:
            return self.drops_dict[int(sid[1:])]
        else:
            return None
    
    def distance(self, sid_o:str, sid_d:str) -> float:
        
        # TODO optimize this based on a hashmap and 
        # using preload objetcts no-on the fly transformations 

        # return euclidian distance between two nodes (sid)
        org_point = self.get_node_sid(sid_o).point
        des_point = self.get_node_sid(sid_d).point
        return org_point.distance(des_point)
    
    def plot(self, file_name='plot_map.html'):
        # get data 
        geos_layer_df = self.city.to_gpd()
        solution_layer_df = self.get_solution_df()

        # build map
        out_map = KeplerGl(height=400, config=KEPLER_CONFIG)
        # load data
        out_map.add_data(data=geos_layer_df, name='geos')
        out_map.add_data(data=solution_layer_df, name='solution')

        out_map.save_to_html(file_name=file_name)