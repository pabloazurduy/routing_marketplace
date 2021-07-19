
import unittest

import pandas as pd
import numpy as np 
from datetime import date 
from matching import MarketplaceInstance
from routing import City
from constants import BETA_INIT

class MarketplaceSimulation(unittest.TestCase):
            
    def build_sim_marketplace(self):
        city_inst = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')
        market = MarketplaceInstance.build_simulated(num_clouders=150, 
                                                     city=city_inst, 
                                                     mean_beta_features=BETA_INIT)

    def test_simulate_auction(self):
        market = MarketplaceSim()
        
        markeplace_sim = MarketplaceSim.build_marketplace_simulation(


        )
        self.assertIsInstance(markeplace_sim, pd.DataFrame)
        columns = ['clouder_id',
                   'route_id',
                   'accepted'
        ]
        self.asserTrue()