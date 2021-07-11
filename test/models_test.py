import unittest
from data_models import OptInstance
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

        opt_instance = OptInstance.load_instance(instance_df)
        self.assertIsInstance(opt_instance, OptInstance) 

    def test_plot(self):
        instance_sol_df    = pd.read_csv('instance_simulator/real_instances/instance_sol_2021-06-08.csv', sep=';')
        instance_sol_df['req_date'] = np.where(~instance_sol_df['is_warehouse'], date(2021,6,8), None)
        opt_instance_prev = OptInstance.load_instance(instance_sol_df)
        opt_instance_prev.build_features()
        opt_instance_prev.plot()