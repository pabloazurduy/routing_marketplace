import unittest
from data_models import OptInstance
import pandas as pd 
import numpy as np 
from datetime import date 

class Models(unittest.TestCase):

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
