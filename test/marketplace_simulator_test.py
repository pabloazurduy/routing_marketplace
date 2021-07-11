
import unittest

import pandas as pd
import numpy as np 
from datetime import date 

from instance_simulator.marketplace_simulator import MarketplaceSim
from data_models import RoutingInstance

def get_beta_dict():
    instance_sol_attr  = pd.read_csv('instance_simulator/real_instances/instance_sol_attributes_2021-06-08.csv', sep=';')
    instance_sol_df    = pd.read_csv('instance_simulator/real_instances/instance_sol_2021-06-08.csv', sep=';')

    instance_sol_df['req_date'] = np.where(~instance_sol_df['is_warehouse'], date(2021,6,8), None)
    routing_instance_prev = RoutingInstance.load_instance(instance_sol_df)
    routing_instance_prev.build_features()
    routing_instance_prev.load_markeplace_data(instance_sol_attr)
    routing_instance_prev.fit_betas_time_based()
    return routing_instance_prev.beta_dict

class MarketplaceSimulation(unittest.TestCase):
    
    def test_simulate_routing_instance(self):
        
        routing_instance = RoutingInstance.load_instance()
        routing_instance.get_warm_start()
        
    def build_sim_marketplace(self):
        beta_dict = get_beta_dict()
        market = MarketplaceSim(num_clouders = 150,
                                connected_prob = 0.3,
                                betas = beta_dict, # ignore base has_geo beta values
                                mean_ideal_route = 15,

                                )

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