from datetime import date
from pathlib import Path
from routing import Geodude, RoutingInstance
import pandas as pd
import numpy as np 

if __name__ == "__main__":
    
    # create solved instance 
    instance_sol_attr  = pd.read_csv('instance_simulator/real_instances/instance_sol_attributes_2021-06-08.csv', sep=';')
    instance_sol_df    = pd.read_csv('instance_simulator/real_instances/instance_sol_2021-06-08.csv', sep=';')

    instance_sol_df['req_date'] = np.where(~instance_sol_df['is_warehouse'], date(2021,6,8), None)
    routing_instance_prev = RoutingInstance.load_instance(instance_sol_df)
    routing_instance_prev.build_features()
    routing_instance_prev.load_markeplace_data(instance_sol_attr)
    routing_instance_prev.fit_betas_time_based()

    INSTANCES = ['instance_simulator/real_instances/instance_2021-05-13.csv',
                'instance_simulator/real_instances/instance_2021-05-24.csv',
                'instance_simulator/real_instances/instance_2021-05-26.csv',
                'instance_simulator/real_instances/instance_2021-06-08.csv'
    ]

    for instance_path in INSTANCES:

        # load data 
        instance_df = pd.read_csv(instance_path, sep=';')
        # add req_date 
        instance_df['req_date'] = np.where(~instance_df['is_warehouse'], date(2021,5,24), None)
        routing_instance = RoutingInstance.load_instance(instance_df)
        Geodude.run_opt_model(routing_instance, beta_dict=routing_instance_prev.beta_dict, max_time=1)
        routing_instance.plot(file_name=f'plot_map_{Path(instance_path).stem}.html')
