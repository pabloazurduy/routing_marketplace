import pandas as pd 
import glob
from routing import City, RoutingSolution
from matching import BetaMarket, MarketplaceInstance



if __name__ == "__main__":
    city_inst = City.from_geojson('instance_simulator/geo/region_metropolitana_de_santiago/all.geojson')
    # geo_dict = {geo.id:geo.name for geo in city_inst.geos_list}
    # create a simulated marketplace 
    market = MarketplaceInstance.build_simulated(num_clouders=400, 
                                                    city=city_inst, 
                                                    mean_beta_features = BetaMarket.default())
    # generate a routing solution
    # instance_sol_df =    pd.read_csv('instances/instance_sol_2021-06-08.csv', sep=';')
    instances_filenames = glob.glob('instances/instance_sol_2' + "*.csv")
    instance_sol_df     = pd.concat(map(lambda file: pd.read_csv(file, sep=';'), instances_filenames))
    routing_solution = RoutingSolution.from_df(instance_sol_df, city = city_inst)
    
    solution_random = market.make_simulated_matching(routes = routing_solution, method='random')
    solution_origin = market.make_simulated_matching(routes = routing_solution, method='origin_based')
    print(f'{solution_random.acceptance_rate = }')
    print(f'{solution_random.final_cost = }')
    print(f'{solution_origin.acceptance_rate = }')
    print(f'{solution_origin.final_cost = }')

    #assertIsInstance(solution_random.matching_df, pd.DataFrame)
    #self.assertTrue(solution_origin.acceptance_rate > solution_random.acceptance_rate)
    #self.assertTrue(solution_origin.final_cost < solution_random.final_cost)
