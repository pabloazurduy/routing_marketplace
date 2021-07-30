
import unittest

import pandas as pd
import numpy as np 
from datetime import date 
from matching import Abra, BetaMarket, MarketplaceInstance, Clouder
from routing import City, Geodude, RoutingInstance, RoutingSolution
from constants import BETA_TEST, BETA_INIT
import os 
import glob

class RoutingTest(unittest.TestCase): 
    def test_create_simple_routing_instance(self):
        instance_df = pd.read_csv('instance_simulator/real_instances/instance_2021-05-24.csv', sep=';')
        instance_df['req_date'] = np.where(~instance_df['is_warehouse'],  
                                        date(2021,5,24), 
                                        None)
        routing_instance = RoutingInstance.from_df(instance_df)
        self.assertIsInstance(routing_instance, RoutingInstance) 

    def test_create_solved_routing_instance(self):
        # load data
        instance_sol_df    = pd.read_csv('instance_simulator/real_instances/instance_sol_2021-06-08.csv', sep=';')
        routing_instance = RoutingInstance.from_df(instance_sol_df)
        self.assertIsInstance(routing_instance, RoutingInstance) 
        self.assertIsInstance(routing_instance.solution, RoutingSolution) 

    def test_routing_solution_load_df(self):
        # load data
        instance_sol_df    = pd.read_csv('instance_simulator/real_instances/instance_sol_2021-06-08.csv', sep=';')
        city_inst = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')
        routing_solution = RoutingSolution.from_df(instance_sol_df, city = city_inst)


    def test_routing_solution_prop(self):
        instance_sol_df    = pd.read_csv('instance_simulator/real_instances/instance_sol_2021-06-08.csv', sep=';')
        routing_instance = RoutingInstance.from_df(instance_sol_df)
        routing_solution = routing_instance.solution
        self.assertIsInstance(routing_solution.cluster_df, pd.DataFrame)
        self.assertIsInstance(routing_solution.inter_geo_df, pd.DataFrame)

    def test_plot_solution(self):
        instance_sol_df    = pd.read_csv('instance_simulator/real_instances/instance_sol_2021-06-08.csv', sep=';')
        routing_instance = RoutingInstance.from_df(instance_sol_df)
        routing_solution = routing_instance.solution
        
        temp_file_path = 'temporally_plot.html'
        try:
            routing_solution.plot(temp_file_path)
        finally:
            os.remove(temp_file_path)

    def test_warm_start(self):
        instance_df = pd.read_csv('instance_simulator/real_instances/instance_2021-05-13.csv', sep=';')
        routing_instance = RoutingInstance.from_df(instance_df)
        warm_start =  routing_instance.build_warm_start(n_clusters = 50)
        self.assertIsInstance(warm_start, RoutingSolution)
    
    @unittest.skip("slow test")
    def test_geodude(self):
        instance_df = pd.read_csv('instance_simulator/real_instances/instance_2021-05-13.csv', sep=';')
        routing_instance = RoutingInstance.from_df(instance_df)
        geodude_inst = Geodude(routing_instance = routing_instance, beta_market = BetaMarket.default())
        routing_solution = geodude_inst.solve(n_clusters = 25, max_time_min = 5 )
        self.assertIsInstance(routing_solution, RoutingSolution)

    def test_build_beta_dict(self):
        self.assertEqual(BetaMarket.default().dict, BETA_INIT)
    
class MarketplaceTest(unittest.TestCase):
    def build_sim_marketplace(self):
        city_inst = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')
        market = MarketplaceInstance.build_simulated(num_clouders=150, 
                                                     city=city_inst, 
                                                     mean_beta_features = BetaMarket.default())
    
    def test_clouder_sim(self):
        city = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')
        clouder = Clouder.make_fake(id = 1,  
                                    mean_connected_prob=0.3, 
                                    mean_ideal_route = 15,
                                    mean_beta_features = BetaMarket.default(),
                                    geo_prob = {geo.id:1.0/len(city.geos) for geo in city.geos_list},
                                    city = city
                                    )
        self.assertIsInstance(clouder.low_utility_ref, float)
        self.assertIsInstance(clouder.high_utility_ref, float)
        self.assertTrue(clouder.low_utility_ref < clouder.high_utility_ref)

    def test_simulate_auction(self):
        city = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')
        market_sim = MarketplaceInstance.build_simulated(num_clouders=100, 
                                                             city= city, 
                                                             mean_beta_features = BetaMarket.default(), 
                                                             mean_ideal_route=15,
                                                             mean_connected_prob=0.3
                                                             )
    
        self.assertTrue(market_sim.is_fake)

    def test_abra_time_based_beta(self):
        instance_sol_df =    pd.read_csv('instance_simulator/real_instances/instance_sol_2021-06-08.csv', sep=';')
        acceptance_time_df = pd.read_csv('instance_simulator/real_instances/instance_sol_attributes2021-06-08.csv', sep=';')

        routing_instance = RoutingInstance.from_df(instance_sol_df)
        routing_solution = routing_instance.solution

        beta_market = Abra.fit_betas_time_based(routing_solution=routing_solution, acceptance_time_df=acceptance_time_df)
        self.assertIsInstance(beta_market, BetaMarket)
        for beta in beta_market.dict.keys():
            #print(f'{beta}, {beta_market.dict[beta]:0.4f}, {BETA_INIT[beta]:0.4f}')
            self.assertTrue(np.allclose(beta_market.dict[beta], BETA_TEST[beta]))
    
    def test_multi_abra_beta(self):
        
        instances_filenames = glob.glob('instance_simulator/real_instances/instance_sol_2' + "*.csv")
        time_instances_filenames = glob.glob('instance_simulator/real_instances/instance_sol_a' + "*.csv")        


        instance_sol_df    = pd.concat(map(lambda file: pd.read_csv(file, sep=';'), instances_filenames))
        acceptance_time_df = pd.concat(map(lambda file: pd.read_csv(file, sep=';'), time_instances_filenames))
        
        instance_sol_df.drop_duplicates(inplace=True)
        self.assertEqual(sum(acceptance_time_df.id_route.duplicated()),0)
        
        city_inst = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')
        routing_solution = RoutingSolution.from_df(instance_sol_df, city = city_inst)

        beta_market = Abra.fit_betas_time_based(routing_solution=routing_solution, 
                                                acceptance_time_df=acceptance_time_df)