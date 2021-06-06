import unittest
from collections.abc import Iterable

from numpy import result_type 
from instance_simulator import simulator as sim
import datetime 
import itertools
import pandas as pd 
import h3 
from shapely.geometry import Polygon

class InstanceSimulator(unittest.TestCase):
    def test_get_random_inner_point(self):
        hex_key = h3.geo_to_h3(0, 0, 9)
        random_point = sim.get_random_inner_point(hex_key)
        assert random_point.within(Polygon(h3.h3_to_geo_boundary(hex_key)))

    def test_geojson_to_hex(self):
        geojson_file_path = 'instance_simulator/geojson/santiago.geojson'
        hex_cluster = sim.geojson_to_hex(filename = geojson_file_path , res = 9)
        self.assertIsInstance(hex_cluster, Iterable) 

    def test_hex_smother_demand_mapper(self):
        geojson_file_path = 'instance_simulator/geojson/santiago.geojson'
        hex_list = sim.geojson_to_hex(filename = geojson_file_path , res = 9)
        dem_mapper = sim.hex_smother_demand_mapper(hex_list, 5)
        self.assertIsInstance(dem_mapper, dict)
        self.assertEqual(len(set(dem_mapper.values())) ,4) 
    
    def test_hex_smother_demand_mapper_big_santiago(self):
        geojson_file_path = 'instance_simulator/geo/santiago.geojson'
        hex_list = sim.geojson_to_hex(filename = geojson_file_path , res = 9)
        dem_mapper = sim.hex_smother_demand_mapper(hex_list, 5)
        self.assertIsInstance(dem_mapper, dict)
        self.assertEqual(len(set(dem_mapper.values())) ,4) 

    def test_demand_simulator(self):
        geojson_file_path = 'instance_simulator/geo/santiago.geojson'
        hex_list = sim.geojson_to_hex(filename = geojson_file_path , res = 9)
        dem_mapper = sim.hex_smother_demand_mapper(hex_list, 5)
        for (num_days, mean_daily_demand) in itertools.product([1,3,5],[50,300]):
            result_df = sim.simulate_demand_points(dem_mapper, num_days=num_days, 
                                            mean_daily_demand=mean_daily_demand, 
                                            start_date=datetime.date.today())
            print(len(result_df))
            self.assertIsInstance(result_df, pd.DataFrame)

    def test_warehouses_sim(self):
        geojson_file_path = 'instance_simulator/geo/santiago.geojson'
        hex_list = sim.geojson_to_hex(filename = geojson_file_path , res = 9)
        dem_mapper = sim.hex_smother_demand_mapper(hex_list, 15)

        warehouses_df = sim.simulate_warehouses(dem_mapper, weights_storages=[10,3,4,5,5,4,3,2])
        print(len(warehouses_df))
        self.assertIsInstance(warehouses_df, pd.DataFrame)
        
    def test_instance_sim(self):
        demand_df, warehouses_df = sim.simulate_demand_instance(geojson_file_path='instance_simulator/geo/santiago.geojson',
                                                                high_demand_spots = 15,
                                                                weights_storages=[5,1,1,2,3],
                                                                num_days=3,
                                                                mean_daily_demand = 50*15,
                                                                seed=1337)
        self.assertIsInstance(demand_df, pd.DataFrame)
        self.assertIsInstance(warehouses_df, pd.DataFrame)
                    