import unittest

import pandas as pd
import numpy as np
from matching import Abra, BetaMarket, MarketplaceInstance, Clouder, MatchingSolution, MatchingSolutionResult
from routing import City, RoutingInstance, RoutingSolution
from constants import BETA_TEST
import glob
import seaborn as sns


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

    def test_simulate_marketplace(self):
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
        
        instances_filenames      = glob.glob('instance_simulator/real_instances/instance_sol_2' + "*.csv")
        time_instances_filenames = glob.glob('instance_simulator/real_instances/instance_sol_a' + "*.csv")        


        instance_sol_df    = pd.concat(map(lambda file: pd.read_csv(file, sep=';'), instances_filenames))
        acceptance_time_df = pd.concat(map(lambda file: pd.read_csv(file, sep=';'), time_instances_filenames))
        
        instance_sol_df.drop_duplicates(inplace=True)
        self.assertEqual(sum(acceptance_time_df.id_route.duplicated()),0)
        
        city_inst = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')
        routing_solution = RoutingSolution.from_df(instance_sol_df, city = city_inst)

        beta_market = Abra.fit_betas_time_based(routing_solution=routing_solution, 
                                                acceptance_time_df=acceptance_time_df)

    def test_time_histogram(self):
        instances_filenames      = glob.glob('instance_simulator/real_instances/instance_sol_2' + "*.csv")
        time_instances_filenames = glob.glob('instance_simulator/real_instances/instance_sol_a' + "*.csv")        
        pass


    def test_simulated_matching(self):
        city_inst = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')
        # create a simulated marketplace 
        market = MarketplaceInstance.build_simulated(num_clouders=400, 
                                                     city=city_inst, 
                                                     mean_beta_features = BetaMarket.default())
        # generate a routing solution
        # instance_sol_df =    pd.read_csv('instance_simulator/real_instances/instance_sol_2021-06-08.csv', sep=';')
        instances_filenames = glob.glob('instance_simulator/real_instances/instance_sol_2' + "*.csv")
        instance_sol_df     = pd.concat(map(lambda file: pd.read_csv(file, sep=';'), instances_filenames))
        routing_solution = RoutingSolution.from_df(instance_sol_df, city = city_inst)
        
        solution_random = market.make_simulated_matching(routes = routing_solution, method='random')
        solution_origin = market.make_simulated_matching(routes = routing_solution, method='origin_based')
        print(f'{solution_random.acceptance_rate = }')
        print(f'{solution_random.final_cost = }')
        print(f'{solution_origin.acceptance_rate = }')
        print(f'{solution_origin.final_cost = }')

        self.assertIsInstance(solution_random.matching_df, pd.DataFrame)
        self.assertTrue(solution_origin.acceptance_rate > solution_random.acceptance_rate)
        self.assertTrue(solution_origin.final_cost < solution_random.final_cost)

    def test_matching_solution_from_df(self):
        city_inst = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')
        # use a routing solution
        instance_sol_df = pd.read_csv('instance_simulator/real_instances/instance_sol_2021-06-08.csv', sep=';')
        routing_solution = RoutingSolution.from_df(instance_sol_df, city = city_inst)
        
        # generate a simulated marketplace
        market = MarketplaceInstance.build_simulated(num_clouders=400, 
                                                     city=city_inst, 
                                                     mean_beta_features = BetaMarket.default())
        matching_random = market.make_simulated_matching(routes = routing_solution, method='random')
        # create a sampling test file TODO: use tempfile when testing
        matching_random.matching_df.to_csv('instance_simulator/matching_sim/matching_random_sim.csv')
        # read file and re-generate MatchingSolution 
        matching_df = pd.read_csv('instance_simulator/matching_sim/matching_random_sim.csv')
        MatchingSolutionResult.from_df(matching_df, routing_solution=routing_solution)
    
    def test_abra_matching_raises(self):
        # use a routing solution
        city_inst = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')
        instances_filenames = glob.glob('instance_simulator/real_instances/instance_sol_2' + "*.csv")
        instance_sol_df     = pd.concat(map(lambda file: pd.read_csv(file, sep=';'), instances_filenames))
        routing_solution = RoutingSolution.from_df(instance_sol_df, city = city_inst)

        # generate a simulated marketplace
        market = MarketplaceInstance.build_simulated(num_clouders=100, 
                                                     city=city_inst, 
                                                     mean_beta_features = BetaMarket.default())
        with self.assertRaises(ValueError): 
            matching_random = market.make_simulated_matching(routes = routing_solution, method='random')

    def test_abra_acceptance_model_fit(self):
        # use a routing solution
        city_inst = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')
        #instance_sol_df = pd.read_csv('instance_simulator/real_instances/instance_sol_2021-06-08.csv', sep=';')
        instances_filenames = glob.glob('instance_simulator/real_instances/instance_sol_2' + "*.csv")
        instance_sol_df     = pd.concat(map(lambda file: pd.read_csv(file, sep=';'), instances_filenames))
        routing_solution = RoutingSolution.from_df(instance_sol_df, city = city_inst)

        # generate a simulated marketplace
        market = MarketplaceInstance.build_simulated(num_clouders=400, 
                                                     city=city_inst, 
                                                     mean_beta_features = BetaMarket.default())
        matching_random = market.make_simulated_matching(routes = routing_solution, method='random')
        abra = Abra()
        abra.fit_acceptance_model(matching_random)
        self.assertIsNotNone(abra.acceptance_model)
        self.assertIsNotNone(abra.acceptance_model_route_features)
        self.assertIsNotNone(abra.acceptance_model_clouder_features)
        self.assertTrue(0<abra.acceptance_model_auc<1)
    
    def test_abra_price_matrix(self):
        # use a routing solution
        city_inst = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')
        instances_filenames = glob.glob('instance_simulator/real_instances/instance_sol_2' + "*.csv")
        instance_sol_df     = pd.concat(map(lambda file: pd.read_csv(file, sep=';'), instances_filenames))
        routing_solution = RoutingSolution.from_df(instance_sol_df, city = city_inst)

        # generate a simulated marketplace
        market = MarketplaceInstance.build_simulated(num_clouders=400, 
                                                     city=city_inst, 
                                                     mean_beta_features = BetaMarket.default())
        matching_random = market.make_simulated_matching(routes = routing_solution, method='random')
        

        abra = Abra()
        abra.fit_acceptance_model(matching_random)
        instance_sol_df = pd.read_csv('instance_simulator/real_instances/instance_sol_2021-06-08.csv', sep=';')
        routing_abra = RoutingSolution.from_df(instance_sol_df, city = city_inst)
        price_matrix = abra.build_price_matrix(routing_abra, market=market, prob_reference=0.65)
        # abra.build_price_matrix_plot(routing_abra, market=market, prob_reference=0.65)
        self.assertIsInstance(price_matrix, dict)

    
    def test_abra_matching(self):
        # use a routing solution
        city_inst = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')
        instances_filenames = glob.glob('instance_simulator/real_instances/instance_sol_2' + "*.csv")
        instance_sol_df     = pd.concat(map(lambda file: pd.read_csv(file, sep=';'), instances_filenames))
        routing_solution = RoutingSolution.from_df(instance_sol_df, city = city_inst)

        # generate a simulated marketplace
        market = MarketplaceInstance.build_simulated(num_clouders=400, 
                                                     city=city_inst, 
                                                     mean_beta_features = BetaMarket.default())
        matching_random = market.make_simulated_matching(routes = routing_solution, method='random')

        # build abra model
        abra = Abra()
        abra.fit_acceptance_model(matching_random)
        instance_sol_df = pd.read_csv('instance_simulator/real_instances/instance_sol_2021-06-08.csv', sep=';')
        routing_abra = RoutingSolution.from_df(instance_sol_df, city = city_inst)
        match_solution = abra.make_matching(routing_abra, market=market, prob_reference=0.65)
        self.assertIsInstance(match_solution, MatchingSolution)
        self.assertTrue(match_solution.total_expected_cost >= 0)
        self.assertTrue(len(match_solution.match) == len(routing_abra.routes))
    
    def test_abra_matching_route_price_plot(self):
        # use a routing solution
        city_inst = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')
        instances_filenames = glob.glob('instance_simulator/real_instances/instance_sol_2' + "*.csv")
        instance_sol_df     = pd.concat(map(lambda file: pd.read_csv(file, sep=';'), instances_filenames))
        routing_solution = RoutingSolution.from_df(instance_sol_df, city = city_inst)

        # generate a simulated marketplace
        market = MarketplaceInstance.build_simulated(num_clouders=400, 
                                                     city=city_inst, 
                                                     mean_beta_features = BetaMarket.default())
        matching_random = market.make_simulated_matching(routes = routing_solution, method='random')

        # build abra model
        abra = Abra()
        abra.fit_acceptance_model(matching_random)
        instance_sol_df = pd.read_csv('instance_simulator/real_instances/instance_sol_2021-06-08.csv', sep=';')
        routing_abra = RoutingSolution.from_df(instance_sol_df, city = city_inst)
        prices = []
        for prob in np.linspace(start=0.20, stop=0.8, num=10):
            match_solution = abra.make_matching(routing_abra, market=market, prob_reference=prob)
            price_list = [{'route_id':route_id, 'route_price':route_price, 'prob':prob} for route_id,route_price in match_solution.expected_price.items()]
            prices += price_list
            print(f'{prob = }, {match_solution.total_expected_cost = }')
        prices_df = pd.DataFrame(prices)
        palette = sns.color_palette("mako", as_cmap=True)
        sns.lineplot(x="prob", y="route_price",hue="route_id", 
                     palette=palette, data=prices_df)