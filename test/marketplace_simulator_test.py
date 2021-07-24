
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
                                    mean_route_len = 15,
                                    mean_beta_features = BETA_INIT,
                                    geo_prob = {geo.id:1.0/len(city.geos) for geo in city.geos_list},
                                    city = city
                                    )
        self.assertIsInstance(clouder.sim_worst_route_utility, float)
        self.assertIsInstance(clouder.sim_best_route_utility, float)
        self.assertTrue(clouder.sim_worst_route_utility < clouder.sim_best_route_utility)

    def test_simulate_auction(self):
        #market = MarketplaceInstance()
        
        #markeplace_sim = MarketplaceInstance.build_marketplace_simulation()
        """
        self.assertIsInstance(markeplace_sim, pd.DataFrame)
        columns = ['clouder_id',
                   'route_id',
                   'accepted'
        ]
        self.asserTrue()
        """
        pass 