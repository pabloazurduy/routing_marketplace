import unittest
from collections.abc import Iterable 
from instance_simulator import hex_simulator as sim

class InstanceSimulator(unittest.TestCase):

    def test_geojson_to_hex(self):
        geojson_file_path = '/instance_simulator/geojson/santiago.geojson'
        hex_cluster = sim.geojson_to_hex(filename = geojson_file_path , res = 9)
        self.assertIsInstance(hex_cluster, Iterable) 

    def test_hex_smother_demand_mapper(self):
        geojson_file_path = '/instance_simulator/geojson/santiago.geojson'
        hex_list = sim.geojson_to_hex(filename = geojson_file_path , res = 9)
        dem_mapper = sim.hex_smother_demand_mapper(hex_list, 10)
        self.assertIsInstance(dem_mapper, dict)

    #def test_last_value(self):
    #    pass