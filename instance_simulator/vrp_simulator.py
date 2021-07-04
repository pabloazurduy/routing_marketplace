from datetime import datetime, timedelta, date
import json
import random
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

import h3
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


def multipolygon_2_poligon_list(geo_json):
    """function that convert a multipoligon dict to a list of poligons
    
    Arguments:
        geo_json {dict} -- from a geojson file the geojson_file['geometry'] subdict 
    
    Raises:
        ValueError: [description]
    
    Returns:
        list -- list of Polygons{type='Polygon','coordinates'=[]}
    """
    if 'type' not in geo_json.keys():
        raise ValueError('geo_json not in correct format')
    if geo_json['type'] != 'MultiPolygon':
        raise ValueError('geo_json is not the correct type')
    
    result = []
    poli_obj = dict()
    for set_cor in geo_json['coordinates']:
        poli_obj['type']='Polygon'
        poli_obj['coordinates']=set_cor
        result.append(deepcopy(poli_obj))
        
    return result
    

def geojson_to_hex(filename:str,
                   res=9,
                   geo_json_conformant=True) -> List[str]:
    """ loads a geojson file and convert it to a hexa cluster. In case of multifeatures will only convert the first one.
    
    Arguments:
        filename {str} -- path to designated geojson file to convert to hex
    
    Keyword Arguments:
        res {int} -- hex resolution (default: {9})
        geo_json_conformant {bool} -- geo_json_conformant from 3core.polyfill function (default: {True})
    
    Returns:
        [str] -- json 
    """
    with open(filename) as json_file:
        data = json.load(json_file)
    
    if 'geometry' in data.keys():    
        geo_json = data['geometry']
    elif 'features' in data.keys():    
        geo_json = data['features'][0]['geometry']
    
    to_hexacluster = []
    if geo_json['type']=='Polygon':
        to_hexacluster = list(h3.polyfill(geo_json, res, geo_json_conformant))
    elif geo_json['type']=='MultiPolygon':
        for poligon in multipolygon_2_poligon_list(geo_json):
            new_hex = list(h3.polyfill(poligon, 9, geo_json_conformant))
            to_hexacluster = list(set(to_hexacluster + new_hex))    
    return to_hexacluster
     
def get_random_inner_point(hex_key:str)->Point:
    """ function to get a random point inside a hexagon

    Args:
        hex_key (str): hex key value 

    Returns:
        Point: shapely.geometry.Point with the (x,y) pair on the hexagon 
    """    
    hex_polygon = Polygon(h3.h3_to_geo_boundary(hex_key))
    minx, miny, maxx, maxy = hex_polygon.bounds
    while True:
        random_point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if hex_polygon.contains(random_point):
            return random_point

DEM_WEIGHTS = {'very_high':3.5,
               'high':3,
               'mid':2,
               'low':1
                }
def hex_smother_demand_mapper(hex_list:List[str],  # all the hexacluster list 
                              num_hex_with_high_demand: int = 10, # list with higher demand hex cluster 
                              dem_weights:Dict[str,float] = DEM_WEIGHTS # demand weights
                              ) -> Dict[str,float]: # dict with hex:probability
    
    # cast number of high demand hex                                 
    num_hex_with_high_demand = min(num_hex_with_high_demand, len(hex_list))

    very_high_demand = random.sample(hex_list, k= num_hex_with_high_demand)
    high_demand_inc = set()
    mid_demand_inc = set()

    # smother ring 
    for hex in very_high_demand:
        high_demand_inc.update(h3.k_ring(hex, k=1))
        mid_demand_inc.update(h3.k_ring(hex, k=2))
    
    dem_mapper = {}
    for hex in hex_list:
        if hex in very_high_demand:
            dem_mapper[hex] = dem_weights['very_high']
        elif hex in high_demand_inc:
            dem_mapper[hex] = dem_weights['high']
        elif hex in mid_demand_inc:
            dem_mapper[hex] = dem_weights['mid']
        else:
            dem_mapper[hex] = dem_weights['low']
    return dem_mapper

def simulate_demand_points(dem_mapper:Dict[str,float], # a demand mapper from hex_smother_demand_mapper
                    num_days:int = 1,
                    mean_daily_demand:int = 50,
                    start_date:date = date.today(),
                    seed:int = 1337
                    ) -> pd.DataFrame:
    # set seed
    random.seed(seed)
    np.random.seed(seed)

    daily_demand = np.random.poisson(mean_daily_demand, size = num_days)
    days = [start_date + timedelta(days=x) for x in range(num_days)]
    dem_list = []
    id = 0
    for i, day_date in enumerate(days): # iterate over the days
        day_demand = daily_demand[i]
        for point in range(day_demand): # get all points
            # select a random hex
            point_hex = random.choices(population=list(dem_mapper.keys()),
                                       weights= dem_mapper.values(), 
                                       k=1)
            # get a random point in the hex 
            point_coordinates = get_random_inner_point(point_hex[0])
            dem_list.append({'lat':point_coordinates.x,
                             'long':point_coordinates.y,
                             'day': day_date,
                             'id':id
            })
            id += 1
    dem_df = pd.DataFrame(dem_list)
    return dem_df


def get_random_time(lambda_value:float)-> float:
    return np.random.exponential(lambda_value)

def simulate_warehouses(dem_mapper:Dict[str,float],
                        weights_storages:List[float]=[5,1,1,2,3] , # weights storages 
                        seed:int=1337
                        )-> pd.DataFrame:
    
    random.seed(seed) # set seed for consistency with warehouses 
    point_hex_list = random.choices(population=list(dem_mapper.keys()),
                                    weights= dem_mapper.values(), 
                                    k=len(weights_storages))
    warehouses = []
    for i,storage_w in enumerate(weights_storages):
        point = get_random_inner_point(point_hex_list[i])        
        warehouses.append({'lat': point.x,
                           'long': point.y,
                           'weight': storage_w,
                           'id':i
        })
    return pd.DataFrame(warehouses)

def simulate_demand_instance(geojson_file_path='instance_simulator/geo/santiago.geojson',
                             high_demand_spots:int = 15,
                             weights_storages:List[float]=[5,1,1,2,3],
                             num_days:int=3,
                             mean_daily_demand:int = 50*15,
                             seed:int=1337) -> pd.DataFrame:
        
        random.seed(seed) # set seed 

        hex_list =   geojson_to_hex(filename = geojson_file_path , res = 9)
        dem_mapper = hex_smother_demand_mapper(hex_list, high_demand_spots)
        warehouses_df = simulate_warehouses(dem_mapper, 
                                            weights_storages=weights_storages,
                                            seed=seed)
                                               
        demand_df = simulate_demand_points(dem_mapper, num_days=num_days, 
                                           mean_daily_demand=mean_daily_demand, 
                                           start_date=date.today(),
                                           seed=seed)
        
        # par with where the product is storaged of each demand point
        demand_df['warehouse_id'] = random.choices(population = list(warehouses_df['id']),
                                                   weights = list(warehouses_df['weight']),
                                                   k=len(demand_df))            
        return demand_df, warehouses_df
        

if __name__ == "__main__":
    pass