from datetime import date
from pathlib import Path
from constants import BETA_INIT
from routing import Geo, Geodude, RoutingInstance
from matching import Abra
import pandas as pd
import numpy as np 

if __name__ == "__main__":
    
    # create solved instance 
    
    instance_sol_df    = pd.read_csv('instance_simulator/real_instances/instance_sol_2021-06-08.csv', sep=';')
    acceptance_time_df  = pd.read_csv('instance_simulator/real_instances/instance_sol_attributes_2021-06-08.csv', sep=';')
    
    routing_instance = RoutingInstance.load_instance(instance_sol_df)
    routing_solution = routing_instance.solution
    beta_market = Abra.fit_betas_time_based(routing_solution=routing_solution, acceptance_time_df=acceptance_time_df)

    

    INSTANCES = ['instance_simulator/real_instances/instance_2021-05-13.csv',
                 'instance_simulator/real_instances/instance_2021-05-24.csv',
                 'instance_simulator/real_instances/instance_2021-05-26.csv',
                 'instance_simulator/real_instances/instance_2021-06-08.csv'
    ]

    for instance_path in INSTANCES:
        # load data 
        instance_df = pd.read_csv(instance_path, sep=';')
        instance_df['req_date'] = np.where(~instance_df['is_warehouse'], date(2021,5,24), None)
        
        routing_instance = RoutingInstance.load_instance(instance_df)
        routing_model = Geodude(routing_instance=routing_instance, beta_market = beta_market)
        routing_solution = routing_model.solve(max_time_min=30)

        routing_solution.plot(file_name=f'instance_results_test/plot_map_{Path(instance_path).stem}.html')
