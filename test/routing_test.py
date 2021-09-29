
import unittest

import pandas as pd
import numpy as np 
from datetime import date 
from matching import BetaMarket
from routing import City, Geodude, RoutingInstance, RoutingSolution, Node, Route
from constants import BETA_INIT
import os 
import glob
from typing import List

class RoutingTest(unittest.TestCase): 
    def test_create_simple_routing_instance(self):
        instance_df = pd.read_csv('instances/instance_2021-05-24.csv', sep=';')
        instance_df['req_date'] = np.where(~instance_df['is_warehouse'],  
                                        date(2021,5,24), 
                                        None)
        routing_instance = RoutingInstance.from_df(instance_df)
        self.assertIsInstance(routing_instance, RoutingInstance) 

    def test_create_solved_routing_instance(self):
        # load data
        instance_sol_df    = pd.read_csv('instances/instance_sol_2021-06-08.csv', sep=';')
        routing_instance = RoutingInstance.from_df(instance_sol_df)
        self.assertIsInstance(routing_instance, RoutingInstance) 
        self.assertIsInstance(routing_instance.solution, RoutingSolution) 

    def test_routing_solution_load_df(self):
        # load data
        instance_sol_df    = pd.read_csv('instances/instance_sol_2021-06-08.csv', sep=';')
        city_inst = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')
        routing_solution = RoutingSolution.from_df(instance_sol_df, city = city_inst)
    
    def test_route_has_node_sid(self):
        city = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')
        node = Node(id = 1, lat= -10, lng= -15, node_type = 'warehouse',geo_id = 1)
        route = Route(id=1, nodes=[node], city = city)
        self.assertTrue(route.has_node_sid('w1'))
        self.assertFalse(route.has_node_sid('w2'))
        route.add_node(Node(id = 2, lat= -10, lng= -15, node_type = 'warehouse',geo_id = 1))
        self.assertTrue(route.has_node_sid('w2'))

    def test_routing_solution_features(self):

        instances_df = pd.read_csv('instances/consolidated_instance_sol.csv')
        num_of_warehouses = instances_df['is_warehouse'].sum()

        city_inst        = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')
        routing_solution = RoutingSolution.from_df(instances_df, city = city_inst)
        for route in routing_solution.routes_dict.values():
            print(route.ft_size,route.ft_size_drops ,route.ft_size_pickups)
            self.assertEqual(route.ft_size,route.ft_size_drops + route.ft_size_pickups)
            self.assertLessEqual(route.ft_size_pickups,route.ft_size_drops)
            self.assertLessEqual(route.ft_size_pickups,num_of_warehouses)


    def test_routing_solution_prop(self):
        instance_sol_df    = pd.read_csv('instances/instance_sol_2021-06-08.csv', sep=';')
        routing_instance = RoutingInstance.from_df(instance_sol_df)
        routing_solution = routing_instance.solution
        self.assertIsInstance(routing_solution.cluster_df, pd.DataFrame)
        self.assertIsInstance(routing_solution.inter_geo_df, pd.DataFrame)

    def test_plot_solution(self):
        instance_sol_df    = pd.read_csv('instances/instance_sol_2021-06-08.csv', sep=';')
        routing_instance = RoutingInstance.from_df(instance_sol_df)
        routing_solution = routing_instance.solution
        
        temp_file_path = 'temporally_plot.html'
        try:
            routing_solution.plot(temp_file_path)
        finally:
            os.remove(temp_file_path)

    def test_warm_start(self):
        instance_df = pd.read_csv('instances/instance_2021-05-13.csv', sep=';')
        routing_instance = RoutingInstance.from_df(instance_df)
        warm_start =  routing_instance.build_warm_start(n_clusters = 50)
        self.assertIsInstance(warm_start, RoutingSolution)
    
    #@unittest.skip("slow test")
    def test_geodude(self):
        instance_df = pd.read_csv('instances/instance_2021-05-13.csv', sep=';')
        routing_instance = RoutingInstance.from_df(instance_df)
        geodude_inst = Geodude(routing_instance = routing_instance, beta_market = BetaMarket.default())
        routing_solution = geodude_inst.solve(n_clusters = 25, max_time_min = 5 )
        self.assertIsInstance(routing_solution, RoutingSolution)

    def test_build_beta_dict(self):
        self.assertEqual(BetaMarket.default().dict, BETA_INIT)
    
