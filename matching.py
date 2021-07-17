from pydantic import BaseModel
from typing import Dict, Optional
from routing import Geo, City
import random
import numpy as np 
from scipy.special import expit

class Clouder(BaseModel):
    id: int
    origin: Geo
    #trips_count: Optional[int]
    is_fake:bool = False

    # FakeClouderParams
    sim_connect_prob:Optional[float]
    sim_ideal_route_len:Optional[int]
    sim_price_elasticity:Optional[float]
    sim_beta_features:Optional[Dict[str,float]]
    sim_route_utility_low: Optional[float]
    sim_route_utility_up: Optional[float]

    @classmethod
    def make_fake(cls,id:int, mean_connect_prob:float, mean_route_len:int, mean_beta_features:Dict[str,float], 
                    geo_prob:Dict[Geo,float], seed:int = 1337):
        
        np.random.seed(seed)
        random.seed(seed)

        sim_geo = random.choices(population = list(geo_prob.keys()),
                                 weights = list(geo_prob.values())
                                )
        b_conn_dist = 5 
        sim_connect_prob = np.random.beta(a = (mean_connect_prob*b_conn_dist)/(1-mean_connect_prob), b =b_conn_dist)
        sim_route_len = np.random.poisson(lam = mean_route_len)
        
        sim_beta_features = {}
        for feat_name in mean_beta_features.keys():
            mean_param = abs(mean_beta_features[feat_name])
            sim_beta_features[feat_name] = np.random.gamma(shape =  mean_param, scale = 1) # E(X) = shape*scale 

        sim_price_elasticity = np.random.gamma(shape =  10, scale = 1)

        return cls(id = id, origin = sim_geo,  is_fake = True, sim_connect_prob = sim_connect_prob,
                   sim_ideal_route_len=sim_route_len, sim_price_elasticity = sim_price_elasticity)

    def sim_match_route(self, route_features:Dict[str,float], route_price:float)-> float: # IP acceptance
        if not self.is_fake: 
            raise ValueError('This method only works for simulated Clouder')

        route_utility = (sum([self.sim_beta_features[feat]*route_features[feat] for feat in route_features.keys() if feat != 'ft_size_pickups']) 
                            +  route_price*self.sim_price_elasticity
                            + self.sim_beta_features['ft_size_pickups'](self.sim_ideal_route_len-((route_features['ft_size_pickups']-self.sim_ideal_route_len)**2)) # a quadratic term for route ft_size_pickups
                        )
        
        return expit(route_utility)
    

class MarketplaceInstance(BaseModel):
    clouders:Dict[int,Clouder]
    

    @classmethod
    def build_simulated(cls, num_clouders:int, city:City, mean_beta_features:Dict[str,float], 
                        mean_connected_prob:float = 0.3, mean_ideal_route:int = 15):
        
        geo_prob = {geo:1.0/len(city.geos) for geo in city.geos_list}
        fake_clouders_dict = {}

        for clouder_id in range(num_clouders):
            fake_clouders_dict[clouder_id] = Clouder.make_fake(id = clouder_id, 
                                                                mean_connect_prob = mean_connected_prob, 
                                                                mean_route_len = mean_ideal_route, 
                                                                mean_beta_features = mean_beta_features, 
                                                                geo_prob = geo_prob
                                                                )

        return cls(clouders = fake_clouders_dict)
    
class Abra(BaseModel):
    pass
