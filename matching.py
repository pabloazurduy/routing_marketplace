from __future__ import annotations

import random
from os import stat
from typing import Dict, List, Optional, Union
from copy import deepcopy

import numpy as np
import pandas as pd
from numpy.random import beta
from pydantic import BaseModel
from scipy.special import expit
from sklearn import linear_model

from routing import City, Geo, Route, RoutingSolution, BetaMarket
from constants import FEATURES_STAT_SAMPLE


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
                                           price_by_node= 5500, city=city)

        worst_route = Route.make_fake_worst(beta_features, ideal_route_len, 
                                            beta_price, beta_origin, geo_origin=geo_origin, 
                                            price_by_node= 1000, city=city)
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
    
    @property
    def worst_route(self) ->Optional[Route]:
        if self.sim_params:
            return self.sim_params.worst_route
        else:
            return None

    @property
    def best_route(self) ->Optional[Route]:
        if self.sim_params:
            return self.sim_params.best_route
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

    def sim_route_utility(self, route:Route, detailed_dict:bool = False, normalize:bool=True)-> Union[float, Dict]:
        if self.sim_params is None: 
            raise ValueError('This method only works for simulated Clouder')
        assert route.price is not None, 'Route has no price defined'

        feat_has_geo = [feat for feat in self.sim_params.beta_features.keys() if 'ft_has_geo' in feat]
        feat_prop    = [feat for feat in self.sim_params.beta_features.keys() if feat not in feat_has_geo and 'ft_size' not in feat]
        
        util_by_item = { 'linear_gral_features_util':    sum([self.sim_params.beta_features[feat]*getattr(route,feat) for feat in feat_prop]), # general linear features 
                         'linear_has_geo_features_util': sum([self.sim_params.beta_features[feat]*route.ft_has_geo(int(feat.split('_')[-1])) for feat in feat_has_geo]), # has_geo features  
                         'price_util':     self.sim_params.beta_price * np.sqrt(route.price), # route price term
                         'origin_util':    self.sim_params.beta_origin * route.centroid_distance(self.origin.centroid), # origin feature 
                         'route_len_util': self.sim_params.beta_features['ft_size']*(self.sim_params.ideal_route_len-((route.ft_size_pickups-self.sim_params.ideal_route_len)**2)) # a quadratic term for route ft_size_pickups
            }
        if normalize:
            route_utility =  sum([ FEATURES_STAT_SAMPLE[feat_group]['weight']*(util_by_item[feat_group] - FEATURES_STAT_SAMPLE[feat_group]['mean'])\
                                  /FEATURES_STAT_SAMPLE[feat_group]['std'] 
                                   for feat_group in util_by_item.keys()]
                                ) 
        else:
            route_utility = sum(util_by_item.values())
        
        if detailed_dict:
            return util_by_item
        else:
            return route_utility 

    def sim_route_acceptance_prob(self, route:Route)-> float: # IP acceptance
        if self.sim_params is None: 
            raise ValueError('This method only works for simulated Clouder')
        # sigm(-20) ~ 0, sigm(6) ~1 normalize utility to get desired result using a linear extrapolation
        # https://en.wikipedia.org/wiki/Linear_equation#Two-point_form        
        # 
        route_utility = self.sim_route_utility(route)
        pi_max        = self.high_utility_ref
        pi_min        = self.low_utility_ref
        y_pi_max      =  1.5
        y_pi_min      = -30
        norm_route_utility = (y_pi_max-y_pi_min)/(pi_max-pi_min)*(route_utility-pi_min) + y_pi_min
        return expit(norm_route_utility)

class MatchingSolution(BaseModel):
    pass    

class MatchingSolutionResult(BaseModel):
    matching_df:pd.DataFrame
    routes: Dict[int, Route] # route_id, Route 
    clouders: Dict[int, Clouder]
    class Config:
        arbitrary_types_allowed = True
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
                                                                city =city,
                                                                seed= clouder_id
                                                                )

        return cls(clouders_dict = fake_clouders_dict)
    
    def make_simulated_matching(self, routes:RoutingSolution, 
                                method:str = 'random', 
                                increasing_price_it:int = 3, 
                                increasing_price_pct:float = 0.1,
                                initial_price_by_node:float = 1500,
                                seed:int=1334) -> MatchingSolutionResult:
        random.seed(seed)
        routes_dict = deepcopy(routes.routes_dict)
        clouders_dict = deepcopy(self.clouders_dict)

        match_result = []
        match_history:Dict[Tuple[int,int]] = []
        while len(routes_dict)>0:
            match = self.sim_match(routes_dict,clouders_dict,method=method)
            match_history.extend(match)
            for pair in match:
                route = routes_dict[pair[0]]
                num_old_maches = sum(1 for pair in match_history if pair[0]==route.id)-1
                route.price = initial_price_by_node * route.ft_size * (1+ increasing_price_it*num_old_maches)
                clouder = clouders_dict[pair[1]]
                accepted_trip = random.random() < clouder.sim_route_acceptance_prob(route)
                
                best_route_util  = clouder.sim_route_utility(clouder.best_route, detailed_dict =True)
                route_util       = clouder.sim_route_utility(route, detailed_dict =True)
                diff_util        = {key: best_route_util[key] - route_util[key] for key in route_util}
                # worst_route_util = clouder.sim_route_utility(clouder.worst_route, detailed_dict =True)

                best_route_util  = { f'best_{key}' : value for key, value in best_route_util.items() }
                route_util       = { f'route_{key}' : value for key, value in route_util.items() }
                route_diff_util  = { f'best-route_{key}' : value for key, value in diff_util.items() }
                # worst_route_util = { f'worst_{key}' : value for key, value in worst_route_util.items() }

                match_result.append({'route_id': route.id ,
                                     'couder_id': clouder.id,
                                     'route_price': route.price, 
                                     'clouder_prob': clouder.sim_route_acceptance_prob(route),
                                     'clouder_util': clouder.sim_route_utility(route),
                                     'clouder_low_util':clouder.low_utility_ref,
                                     'clouder_high_util':clouder.high_utility_ref,
                                     'accepted_trip':accepted_trip,
                                     **best_route_util,
                                     **route_util,
                                     **route_diff_util
                                     #**worst_route_util,
                })
                if accepted_trip:
                    # if trip accepted we remove both from the matching
                    del routes_dict[route.id]
                    del clouders_dict[clouder.id]

        solution = MatchingSolutionResult(matching_df = pd.DataFrame(match_result),
                                          routes = routes.routes_dict,
                                          clouders = self.clouders_dict)
        return solution


    @staticmethod
    def sim_match(routes_dict:Dict[int,Route], clouders_dict: Dict[int, Clouder], method:str = 'random') -> List[(int,int)]:
        if method == 'random':
            return list(zip(routes_dict.keys(), clouders_dict.keys()))
        else:
            raise NotImplemented
                

class Abra(BaseModel):
    """Matching Model

    Args:
        BaseModel ([type]): [description]

    Returns:
        [type]: [description]
    """    

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

    @staticmethod
    def make_matching(routes:RoutingSolution, market:MarketplaceInstance) -> MatchingSolution:
        raise NotImplemented