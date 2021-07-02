import pandas as pd 
from data_models import OptInstance
from datetime import date

instance_sol_attr  = pd.read_csv('instance_simulator/real_instances/instance_sol_attributes_2021-06-08.csv', sep=';')
instance_sol_df    = pd.read_csv('instance_simulator/real_instances/instance_sol_2021-06-08.csv', sep=';')

instance_sol_df['req_date'] = date(2021,6,8) 
opt_instance = OptInstance.load_instance(instance_sol_df)
opt_instance.build_features()
opt_instance.load_markeplace_data(instance_sol_attr)
opt_instance.fit_betas_time_based()


