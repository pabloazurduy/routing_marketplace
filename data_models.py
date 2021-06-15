# Data Models + Utilities

import pandas as pd 
from pydantic import BaseModel 
from datetime import date 
from typing import List, Optional, Dict
import json
from shapely.geometry import shape, Point, Polygon

# city meta classes 
class Comuna(BaseModel):
    id: int
    name_comuna: str
    provincia: str 
    area: float 
    polygon: Polygon

    class Config:
        arbitrary_types_allowed = True

    def contains(self, lat:float, lng:float) -> bool:
        point = Point(lng,lat)
        return self.polygon.contains(point)

class City(BaseModel):
    id: Optional[int]
    name_city: Optional[str]
    comunas: Dict[int, Comuna]

    @classmethod
    def from_geojson(cls, geojson_file_path:str, name_city:Optional[str]=None, id:Optional[int]=None):

        with open(geojson_file_path) as f:
            geojson = json.load(f)
        
        comunas = {}
        for comuna_id, feature in enumerate(geojson['features']):
            print(feature['properties']['NOM_COM'], comuna_id)
            polygon = shape(feature['geometry'])
            comunas[comuna_id] =  Comuna(id = comuna_id,
                                        name_comuna = feature['properties']['NOM_COM'],
                                        provincia = feature['properties']['NOM_PROV'],
                                        area = feature['properties']['SHAPE_Area'],
                                        polygon = polygon)

        return cls(id=id, name_city = name_city, comunas=comunas)
    
    def get_comuna(self, lat:float, lng:float) -> Comuna:
        for comuna in self.comunas.values():
            if comuna.contains(lat, lng):
                return comuna # pointer 
        return None
    
    def get_comuna_id(self, lat:float, lng:float) -> int:
        com_loc = self.get_comuna(lat,lng)  
        if com_loc is None:
            return None 
        else:
            return com_loc.id

# instance classes 
class Warehouse(BaseModel):
    id: int 
    lat: float
    lng: float
    comuna_id: Optional[int]
    
class Drop(BaseModel):
    id: int 
    lat: float
    lng: float
    warehouse : Warehouse
    store_id : int 
    req_date: date
    schedule_date: Optional[date]
    comuna_id: Optional[int]

INSTANCE_DF_COLUMNS = ['store_id', 'lon', 'is_warehouse', 
                       'lat', 'pickup_warehouse_id', 'req_date']

class OptInstance(BaseModel):
    warehouses: List[Warehouse]
    drops: List[Drop]
    city: City

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
        
        warehouses = []
        drops = []
        warehouses_dict = {}
        for wh_inst in warehouses_list:
            wh_id = wh_inst['pickup_warehouse_id']
            warehouses_dict[wh_id] = Warehouse(id = wh_id,
                                               lat =wh_inst['lat'],
                                               lng =wh_inst['lon'],
                                               comuna_id =  city_inst.get_comuna_id(wh_inst['lat'], wh_inst['lon'])
                                               )
        warehouses = list(warehouses_dict.values())
        
        for d_id, drop_inst in enumerate(drops_list):
            drop = Drop(id = d_id,
                        lat =drop_inst['lat'],
                        lng =drop_inst['lon'],
                        warehouse = warehouses_dict[drop_inst['pickup_warehouse_id']],
                        store_id = drop_inst['store_id'],
                        req_date = drop_inst['req_date'],
                        comuna_id = city_inst.get_comuna_id(drop_inst['lat'],drop_inst['lon'])
                        )
            drops.append(drop)
            
        return cls(warehouses = warehouses, drops= drops, city = city_inst)


