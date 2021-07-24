from os import stat
from pydantic import BaseModel
from typing import Dict, Optional, List
from routing import Geo, City, Route
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
    sim_beta_price:Optional[float]
    sim_beta_origin:Optional[float]
    sim_beta_features:Optional[Dict[str,float]] # lineal features 
    sim_worst_route_utility: Optional[float]  # low  level utility reference 
    sim_best_route_utility: Optional[float]   # high level utility reference 
    
    @classmethod
    def make_fake(cls,id:int, mean_connected_prob:float, mean_ideal_route:int, 
                  mean_beta_features:Dict[str,float], geo_prob:Dict[int,float], # geo_id, weight 
                  city:City, seed:int = 1337):
        
        np.random.seed(seed)
        random.seed(seed)

        sim_geo = random.choices(population = [ city.geos[geo_id] for geo_id in geo_prob.keys() ],
                                 weights = list(geo_prob.values())
                                )[0]
        b_conn_dist = 5 # b parameter (in beta) for connected probability 
        sim_connect_prob = np.random.beta(a = (mean_connected_prob*b_conn_dist)/(1-mean_connected_prob), b =b_conn_dist)
        sim_ideal_route_len = np.random.poisson(lam = mean_ideal_route)
        
        sim_beta_features = {}
        for feat_name in mean_beta_features.keys():
            mean_param = abs(mean_beta_features[feat_name])
            sim_beta_features[feat_name] =np.sign(mean_beta_features[feat_name]) * np.random.gamma(shape =  mean_param, scale = 1) # E(X) = shape*scale 
        
        sim_beta_origin = min(sim_beta_features.values()) # min value from all features 
        sim_beta_price = np.random.gamma(shape =  10, scale = 1)

        best_route  = Route.make_fake_best(sim_beta_features, sim_ideal_route_len, 
                                           sim_beta_price, sim_beta_origin, geo_origin=sim_geo,
                                           price_by_node= 3500, city=city)

        worst_route = Route.make_fake_worst(sim_beta_features, sim_ideal_route_len, 
                                            sim_beta_price, sim_beta_origin, geo_origin=sim_geo, 
                                            price_by_node= 800, city=city)

        best_route_utility = cls.route_utility(best_route, sim_beta_features, sim_beta_price,
                                                sim_beta_origin, sim_geo, sim_ideal_route_len)
                                            
        worst_route_utility = cls.route_utility(worst_route, sim_beta_features, sim_beta_price,
                                                sim_beta_origin, sim_geo, sim_ideal_route_len)

        return cls(id = id, origin = sim_geo, is_fake = True, sim_connect_prob = sim_connect_prob,
                   sim_ideal_route_len=sim_ideal_route_len, sim_beta_price = sim_beta_price, 
                   sim_worst_route_utility = worst_route_utility, sim_best_route_utility = best_route_utility
                   )


    def sim_route_utility(self, route:Route)-> float:
        if not self.is_fake: 
            raise ValueError('This method only works for simulated Clouder')
        utility = self.route_utility(route,
                                  self.sim_beta_features,
                                  self.sim_beta_price,
                                  self.sim_beta_origin,
                                  self.origin,
                                  self.sim_ideal_route_len
                                  )
        return utility

    def sim_route_acceptance_prob(self, route:Route)-> float: # IP acceptance
        route_utility = self.sim_route_utility(route)
        # sigm(-4) ~ 0, sigm(4) ~1 normalize utility to get desired result using a linear extrapolation
        # https://en.wikipedia.org/wiki/Linear_equation#Two-point_form
        pi_max = self.sim_best_route_utility
        pi_min = self.sim_worst_route_utility
        norm_route_utility = 8/(pi_max-pi_min)*(route_utility-pi_min) - 4
        return expit(norm_route_utility)

    @staticmethod
    def route_utility(route:Route, sim_beta_features:Dict[str,float], sim_beta_price_elasticity:float, 
                      sim_beta_origin:float, origin:Geo, sim_ideal_route_len:int) -> float:

        feat_has_geo = [feat for feat in sim_beta_features.keys() if 'ft_has_geo' in feat]
        feat_prop    = [feat for feat in sim_beta_features.keys() if feat not in feat_has_geo and 'ft_size_pickups' not in feat]

        route_utility = (sum([sim_beta_features[feat]*getattr(route,feat) for feat in feat_prop]) + # general linear features 
                         sum([sim_beta_features[feat]*route.ft_has_geo(int(feat.split('_')[-1])) for feat in feat_has_geo]) + # has_geo features  
                         sim_beta_price_elasticity * route.price +   # route price term
                         sim_beta_origin * route.centroid_distance(origin.centroid) + # origin feature 
                         sim_beta_features['ft_size_pickups']*(sim_ideal_route_len-((route.ft_size_pickups-sim_ideal_route_len)**2)) # a quadratic term for route ft_size_pickups
                        )
        
        return route_utility

class MarketplaceInstance(BaseModel):
    clouders_dict:Dict[int,Clouder]

    @property
    def clouders(self)-> List[Clouder]:
        return self.clouders_dict.values()

    @property
    def is_fake(self)->bool:
        return any(clouder.is_fake for clouder in self.clouders)

    @classmethod
    def build_simulated(cls, num_clouders:int, city:City, mean_beta_features:Dict[str,float], 
                        mean_connected_prob:float = 0.3, mean_ideal_route:int = 15):
        
        geo_prob = {geo.id:1.0/len(city.geos) for geo in city.geos_list}
        fake_clouders_dict = {}

        for clouder_id in range(num_clouders):
            fake_clouders_dict[clouder_id] = Clouder.make_fake(id = clouder_id, 
                                                                mean_connected_prob = mean_connected_prob, 
                                                                mean_ideal_route = mean_ideal_route, 
                                                                mean_beta_features = mean_beta_features, 
                                                                geo_prob = geo_prob, 
                                                                city =city
                                                                )

        return cls(clouders_dict = fake_clouders_dict)
    
class Abra(BaseModel):
    pass
