
import unittest

import pandas as pd
import numpy as np 
from datetime import date 
from matching import MarketplaceInstance, Clouder
from routing import City
from constants import BETA_INIT

class MarketplaceSimulation(unittest.TestCase):
            
    def build_sim_marketplace(self):
        city_inst = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')
        market = MarketplaceInstance.build_simulated(num_clouders=150, 
                                                     city=city_inst, 
                                                     mean_beta_features=BETA_INIT)
    
    def test_clouder_sim(self):
        city = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')
        clouder = Clouder.make_fake(id = 1,  
                                    mean_connected_prob=0.3, 
                                    mean_ideal_route = 15,
                                    mean_beta_features = BETA_INIT,
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
                                                             mean_beta_features=BETA_INIT, 
                                                             mean_ideal_route=15,
                                                             mean_connected_prob=0.3
                                                             )
        self.assertTrue(market_sim.is_fake)
