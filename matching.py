from __future__ import annotations

import random
from os import stat
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.random import beta
from pydantic import BaseModel
from scipy.special import expit
from sklearn import linear_model

from routing import City, Geo, Route, RoutingSolution, BetaMarket



class ClouderSimParams(BaseModel):
    # FakeClouderParams
    connect_prob:float
    ideal_route_len:int
    beta_price:float
    beta_origin:float
    beta_features:Dict[str,float] # lineal features 
    worst_route: Route  # low  level utility reference 
    best_route: Route   # high level utility reference 

    @classmethod
    def build(cls, seed:int, mean_connected_prob:float, mean_ideal_route:int, mean_beta_features:BetaMarket,
             geo_origin:Geo, city:City) -> ClouderSimParams:
        mean_beta_features_dict = mean_beta_features.dict 
        np.random.seed(seed)

        b_conn_dist = 5 # b parameter (in beta) for connected probability 
        connect_prob = np.random.beta(a = (mean_connected_prob*b_conn_dist)/(1-mean_connected_prob), b =b_conn_dist)
        ideal_route_len = np.random.poisson(lam = mean_ideal_route)
        
        beta_features = {}
        for feat_name in mean_beta_features_dict.keys():
            mean_param = abs(mean_beta_features_dict[feat_name])
            beta_features[feat_name] =np.sign(mean_beta_features_dict[feat_name]) * np.random.gamma(shape =  mean_param, scale = 1) # E(X) = shape*scale 
        
        beta_origin = min(beta_features.values()) # min value from all features 
        beta_price = np.random.gamma(shape =  10, scale = 1)

        best_route  = Route.make_fake_best(beta_features, ideal_route_len, 
                                           beta_price, beta_origin, geo_origin=geo_origin,
                                           price_by_node= 3500, city=city)

        worst_route = Route.make_fake_worst(beta_features, ideal_route_len, 
                                            beta_price, beta_origin, geo_origin=geo_origin, 
                                            price_by_node= 800, city=city)
        return cls(connect_prob = connect_prob, ideal_route_len = ideal_route_len, beta_price=beta_price, beta_origin=beta_origin, 
                   beta_features=beta_features, worst_route = worst_route, best_route=best_route)

class Clouder(BaseModel):
    id: int
    origin: Geo
    #trips_count: Optional[int]
    sim_params: Optional[ClouderSimParams]

    @property
    def is_fake(self) -> bool:
        return self.sim_params is not None
    
    @property
    def low_utility_ref(self):
        """The low_utility_ref property."""
        if self.sim_params:
            return self.sim_route_utility(self.sim_params.worst_route)
        else:
            return None

    @property
    def high_utility_ref(self):
        """The low_utility_ref property."""
        if self.sim_params:
            return self.sim_route_utility(self.sim_params.best_route)
        else:
            return None

    @classmethod
    def make_fake(cls,id:int, mean_connected_prob:float, mean_ideal_route:int, 
                  mean_beta_features:BetaMarket, geo_prob:Dict[int,float], # geo_id, weight 
                  city:City, seed:int = 1337):
        
        random.seed(seed)
        geo_origin = random.choices(population = [ city.geos[geo_id] for geo_id in geo_prob.keys() ],
                            weights = list(geo_prob.values())
                        )[0]
        
        sim_params = ClouderSimParams.build(seed=seed,
                                            mean_connected_prob=mean_connected_prob,
                                            mean_ideal_route=mean_ideal_route,
                                            mean_beta_features=mean_beta_features,
                                            geo_origin=geo_origin,
                                            city=city)      

        return cls(id = id, origin = geo_origin, sim_params=sim_params)

    def sim_route_utility(self, route:Route)-> float:
        if self.sim_params is None: 
            raise ValueError('This method only works for simulated Clouder')

        feat_has_geo = [feat for feat in self.sim_params.beta_features.keys() if 'ft_has_geo' in feat]
        feat_prop    = [feat for feat in self.sim_params.beta_features.keys() if feat not in feat_has_geo and 'ft_size_pickups' not in feat]

        route_utility = (sum([self.sim_params.beta_features[feat]*getattr(route,feat) for feat in feat_prop]) + # general linear features 
                         sum([self.sim_params.beta_features[feat]*route.ft_has_geo(int(feat.split('_')[-1])) for feat in feat_has_geo]) + # has_geo features  
                         self.sim_params.beta_price * route.price +   # route price term
                         self.sim_params.beta_origin * route.centroid_distance(self.origin.centroid) + # origin feature 
                         self.sim_params.beta_features['ft_size_pickups']*(self.sim_params.ideal_route_len-((route.ft_size_pickups-self.sim_params.ideal_route_len)**2)) # a quadratic term for route ft_size_pickups
                        )
        return route_utility 

    def sim_route_acceptance_prob(self, route:Route)-> float: # IP acceptance
        if self.sim_params is None: 
            raise ValueError('This method only works for simulated Clouder')

        # sigm(-4) ~ 0, sigm(4) ~1 normalize utility to get desired result using a linear extrapolation
        # https://en.wikipedia.org/wiki/Linear_equation#Two-point_form        
        route_utility = self.sim_route_utility(route)
        pi_max        = self.high_utility_ref
        pi_min        = self.low_utility_ref
        norm_route_utility = 8/(pi_max-pi_min)*(route_utility-pi_min) - 4
        return expit(norm_route_utility)

class MatchinSolution(BaseModel):
    pass

class MarketplaceInstance(BaseModel):
    clouders_dict:Dict[int,Clouder]

    @property
    def clouders(self)-> List[Clouder]:
        return list(self.clouders_dict.values())

    @property
    def is_fake(self)->bool:
        return any(clouder.is_fake for clouder in self.clouders)

    @classmethod
    def build_simulated(cls, num_clouders:int, city:City, mean_beta_features:BetaMarket, 
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
    
    def make_matching(self, routes:RoutingSolution, method:str = 'random') -> MatchingSolution:
        pass

class Abra(BaseModel):
    @staticmethod    
    def fit_betas_time_based(routing_solution:RoutingSolution, acceptance_time_df:pd.DataFrame) -> BetaMarket:

        acceptance_time_df['acceptance_time_min'] =( pd.to_datetime(acceptance_time_df['route_acceptance_timestamp'])
                                                   - pd.to_datetime(acceptance_time_df['route_creation_timestamp'])
                                                ).dt.total_seconds()/60 
        sol_df = pd.DataFrame(acceptance_time_df[['id_route','acceptance_time_min']])        
        
        features_df = routing_solution.features_df

        train_df = pd.merge(left = sol_df, right = features_df, how='left', on ='id_route')

        model = linear_model.Lasso(alpha=0.1)
        #linear_model.LassoLars(alpha=.1)
        #linear_model.Ridge(alpha=.5)
        x_df = train_df[train_df.columns.difference(['acceptance_time_min', 'id_route'])]
        model.fit(X = x_df , y = train_df['acceptance_time_min'] )
        
        # To print OLS summary  
        # from statsmodels.api import OLS
        # result = OLS(train_df['acceptance_time_min'],x_df).fit_regularized('sqrt_lasso')
        # with open('summary.txt', 'w') as fh:
        #     fh.write(OLS(train_df['acceptance_time_min'],x_df).fit().summary().as_text())
        # print(result.params)

        beta_dict = {col:model.coef_[i] for i,col in enumerate(x_df.columns)}
        return BetaMarket(beta_dict=beta_dict)

