from datetime import date
from pathlib import Path
from routing import  RoutingModel, RoutingInstance
from matching import BetaMarket
import pandas as pd
import numpy as np 

if __name__ == "__main__":
    
    # create solved instance 
    
    instance_sol_df    = pd.read_csv('instances/instance_sol_2021-06-08.csv', sep=';')
    acceptance_time_df  = pd.read_csv('instances/instance_sol_attributes2021-06-08.csv', sep=';')
    
    routing_instance = RoutingInstance.from_df(instance_sol_df)
    routing_solution = routing_instance.solution
    beta_market = BetaMarket.default() #MatchingModel.fit_betas_time_based(routing_solution=routing_solution, acceptance_time_df=acceptance_time_df)


    INSTANCES = ['instances/instance_2021-05-13.csv',
                 'instances/instance_2021-05-24.csv',
                 'instances/instance_2021-05-26.csv',
                 'instances/instance_2021-06-08.csv'
    ]

    for instance_path in INSTANCES:
        # load data 
        instance_df = pd.read_csv(instance_path, sep=';')
        instance_df['req_date'] = np.where(~instance_df['is_warehouse'], date(2021,5,24), None)
        
        routing_instance = RoutingInstance.from_df(instance_df)
        routing_model = RoutingModel(routing_instance = routing_instance, beta_market = beta_market)
        routing_solution = routing_model.solve(max_time_min=10)

        routing_solution.plot(file_name=f'instance_results/plot_map_{Path(instance_path).stem}.html')
