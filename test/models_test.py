import unittest
from routing import RoutingInstance
import pandas as pd 
import numpy as np 
from datetime import date 

class DataModels(unittest.TestCase):

    def test_create_instance(self):
        # load data
        instance_df = pd.read_csv('instance_simulator/real_instances/instance_2021-05-24.csv', 
                          sep=';')
        # add req_date 
        instance_df['req_date'] = np.where(~instance_df['is_warehouse'],  
                                        date(2021,5,24), 
                                        None)

        routing_instance = RoutingInstance.load_instance(instance_df)
        self.assertIsInstance(routing_instance, RoutingInstance) 

    def test_plot(self):
        instance_sol_df    = pd.read_csv('instance_simulator/real_instances/instance_sol_2021-06-08.csv', sep=';')
        instance_sol_df['req_date'] = np.where(~instance_sol_df['is_warehouse'], date(2021,6,8), None)
        routing_instance_prev = RoutingInstance.load_instance(instance_sol_df)
        routing_instance_prev.build_features()
        routing_instance_prev.plot()