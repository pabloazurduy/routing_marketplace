from __future__ import annotations
from functools import cached_property

import random
from typing import Dict, List, Optional, Union, Tuple
from copy import deepcopy
import itertools as it 

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.special import expit
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import xgboost as xgb
import networkx as nx
import seaborn as sns

from routing import City, Geo, Route, RoutingSolution, BetaMarket
from constants import FEATURES_STAT_SAMPLE, ROUTE_FEATURES


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
                                           price_by_node= 6500, city=city)

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
        # https://en.wikipedia.org/wiki/Sigmoid_function
        route_utility = self.sim_route_utility(route)
        pi_max        = self.high_utility_ref
        pi_min        = self.low_utility_ref
        y_pi_max      =  1
        y_pi_min      = -3
        norm_route_utility = (y_pi_max-y_pi_min)/(pi_max-pi_min)*(route_utility-pi_min) + y_pi_min
        return expit(norm_route_utility) #* self.sim_params.connect_prob

    def ft_origin_distance(self, route:Route) -> float:
        return route.centroid_distance(self.origin.centroid)

class MatchingSolution(BaseModel):
    match: List[Tuple[int,int]] #route, clouder
    expected_price: Optional[Dict[int,float]] #route, price
    
    @property 
    def total_expected_cost(self):
        return sum(self.expected_price.values())
    
class MatchingSolutionResult(BaseModel):
    matching_df:pd.DataFrame # add clouder origin geo
    clouders: Dict[int, Clouder]
    routes: Dict[int, Route] # route_id, Route 
    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)
    
    @property
    def acceptance_rate(self):
        return self.matching_df['accepted_trip'].mean()
    
    @property
    def final_cost(self) -> float:
        return self.matching_df[self.matching_df['accepted_trip']]['route_price'].sum()
    
    @classmethod
    def from_df(cls, matching_df:pd.DataFrame, routing_solution: RoutingSolution) -> MatchingSolutionResult:
        clouders: Dict[int, Clouder] = {} 
        for clouder in matching_df[['clouder_id','clouder_origin']].drop_duplicates().to_dict('records'):
            geo = routing_solution.city.get_geo_from_name(clouder['clouder_origin'])
            clouders[clouder['clouder_id']] = Clouder(id = clouder['clouder_id'], origin = geo)
        return cls(matching_df= matching_df, clouders=clouders, routes = routing_solution.routes_dict)

    def get_master_df(self, routes_features:List[str] = ROUTE_FEATURES, clouder_features:List[str] = ['origin_distance'] )-> pd.DataFrame:
        master_df = self.matching_df[['route_id','clouder_id','clouder_origin','route_price','accepted_trip']].copy()
        feat_routes:List[Dict[str,float]] = []
        for route in self.routes.values():
            feat_dict = route.get_features_dict(routes_features)
            feat_dict['route_id'] = route.id 
            feat_routes.append(feat_dict)
        routes_features_df = pd.DataFrame(feat_routes)    
        master_df = pd.merge(master_df, routes_features_df, on='route_id', how='right')
        
        if 'origin_distance' in clouder_features:    
            city = list(self.routes.values())[0].city # TODO add this as arg
            
            origin_features:List[Dict[str,float]] = []
            
            for route in self.routes.values():
                geo_names = master_df[master_df['route_id'] == route.id]['clouder_origin'].unique()
                for geo_name in geo_names:
                    geo = city.get_geo_from_name(geo_name)
                    distance = route.centroid_distance(geo.centroid)
                    origin_features.append({'route_id': route.id,
                                            'clouder_origin':geo_name,
                                            'origin_distance':distance
                                            })
            
            origin_feat_df = pd.DataFrame(origin_features)

            master_df = pd.merge(master_df, origin_feat_df, on=['route_id', 'clouder_origin'], how='right')
        
        return master_df

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
                        mean_connected_prob:float = 0.8, mean_ideal_route:int = 15):
        
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
                                increasing_price_it:int = 1, 
                                increasing_price_pct:float = 0.15,
                                initial_price_by_node:float = 1500,
                                seed:int=1334) -> MatchingSolutionResult:
        random.seed(seed)
        routes_dict = deepcopy(routes.routes_dict)
        clouders_dict = deepcopy(self.clouders_dict)

        if len(clouders_dict) < len(routes_dict):
            raise ValueError(f' number of clouders in simulation ({len(clouders_dict)}) is less than the number of routes to match ({len(routes_dict)})')

        match_result = []
        match_history:List[Tuple[int,int]] = []
        while len(routes_dict)>0:

            if method == 'random':
                match = self.sim_match_random(routes_dict, clouders_dict)
            elif method == 'origin_based':
                match = self.sim_match_origin_based(routes_dict, clouders_dict)
            else:
                raise NotImplemented

            match_history.extend(match)
            for pair in match:
                route = routes_dict[pair[0]]
                num_old_maches = sum(1 for pair in match_history if pair[0]==route.id)-1
                route.price = initial_price_by_node * route.ft_size * (1+ increasing_price_pct * (num_old_maches / increasing_price_it))
                clouder = clouders_dict[pair[1]]
                accepted_trip = random.random() < clouder.sim_route_acceptance_prob(route)
                # best_route_util  = clouder.sim_route_utility(clouder.best_route, detailed_dict =True)
                # route_util       = clouder.sim_route_utility(route, detailed_dict =True)
                # diff_util        = {key: best_route_util[key] - route_util[key] for key in route_util}
                # worst_route_util = clouder.sim_route_utility(clouder.worst_route, detailed_dict =True)

                # best_route_util  = { f'best_{key}' : value for key, value in best_route_util.items() }
                # route_util       = { f'route_{key}' : value for key, value in route_util.items() }
                # route_diff_util  = { f'best-route_{key}' : value for key, value in diff_util.items() }
                # worst_route_util = { f'worst_{key}' : value for key, value in worst_route_util.items() }

                match_result.append({'route_id': route.id ,
                                     'clouder_id': clouder.id,
                                     'clouder_origin': clouder.origin.name,
                                     'route_price': route.price, 
                                     'clouder_prob': clouder.sim_route_acceptance_prob(route),
                                     'clouder_util': clouder.sim_route_utility(route),
                                     'clouder_low_util':clouder.low_utility_ref,
                                     'clouder_high_util':clouder.high_utility_ref,
                                     'accepted_trip':accepted_trip,
                                     #**best_route_util,
                                     #**route_util,
                                     #**route_diff_util
                                     #**worst_route_util,
                })
                if accepted_trip:
                    # if trip accepted we remove both from the matching
                    del routes_dict[route.id]
                    del clouders_dict[clouder.id]
            #print(f'{len(clouders_dict)} {len(routes_dict)} {len(match_history)} {clouder.sim_route_acceptance_prob(route)}')

        solution = MatchingSolutionResult(matching_df = pd.DataFrame(match_result),
                                          routes = routes.routes_dict,
                                          clouders = self.clouders_dict)
        return solution


    @staticmethod
    def sim_match_random(routes_dict:Dict[int,Route], clouders_dict: Dict[int, Clouder]) -> List[Tuple[int,int]]:
        # WARNING this random function does not have a seed set because was always called from a nested random method
        clouders_ids = list(clouders_dict.keys())
        random.shuffle(clouders_ids)
        return list(zip(routes_dict.keys(),clouders_ids))
    
    @staticmethod            
    def sim_match_origin_based(routes_dict:Dict[int,Route], clouders_dict: Dict[int, Clouder]) -> List[Tuple[int,int]]:
        match: List[Tuple[int,int]] = []
        available_clouders = list(clouders_dict.values())
        routes_list = list(routes_dict.values())
        random.shuffle(routes_list)
        for route in routes_list:
            sorted_clouders = sorted(available_clouders, key= lambda x: route.centroid_distance(x.origin.centroid))
            match.append((route.id, sorted_clouders.pop(0).id))
            available_clouders = sorted_clouders
        return match                

class Abra(BaseModel):
    # TODO separate this into AcceptanceModel class
    acceptance_model: Optional[xgb.Booster]
    acceptance_model_route_features: Optional[List[str]]
    acceptance_model_clouder_features: Optional[List[str]]
    acceptance_model_auc: Optional[float]
    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)


    def fit_acceptance_model(self, matching_result: MatchingSolutionResult, 
                             route_features:List[str] = ROUTE_FEATURES, 
                             clouder_features:List[str] = ['origin_distance', 'route_price'],
                             ) -> None:
        master_df = matching_result.get_master_df(route_features, clouder_features)
        train_df, test_df = train_test_split(master_df, test_size=0.25)

        dtrain = xgb.DMatrix(train_df[route_features + clouder_features], 
                             label=train_df['accepted_trip'].astype(int))
        dtest = xgb.DMatrix(test_df[route_features + clouder_features], 
                            label=test_df['accepted_trip'].astype(int))

        monotone_tuple = tuple(1 if feat =='route_price' else 0 for feat in route_features + clouder_features)
        param = {'max_depth': 2, 
                 'eta': 1, 
                 'objective': 'binary:logistic', 
                 'eval_metric':'auc', 
                 'monotone_constraints': str(monotone_tuple),
                 'tree_method':'exact'
                 }
        evallist = [(dtrain, 'train'), (dtest, 'eval')]
        xgboost_model = xgb.train(param, dtrain, evals=evallist, num_boost_round=5)
        self.acceptance_model = xgboost_model
        self.acceptance_model_route_features = route_features 
        self.acceptance_model_clouder_features = clouder_features
        self.acceptance_model_auc = float((xgboost_model.eval(dtest)).split(':')[1])
    
    @staticmethod    
    def fit_betas_time_based(routing_solution:RoutingSolution, acceptance_time_df:pd.DataFrame, 
                            time_cap_min:float = 60*2) -> BetaMarket:

        acceptance_time_df['acceptance_time_min'] =( pd.to_datetime(acceptance_time_df['route_acceptance_timestamp'])
                                                   - pd.to_datetime(acceptance_time_df['route_creation_timestamp'])
                                                ).dt.total_seconds()/60 
        # data cap 
        # acceptance_time_df['acceptance_time_min'] =  np.minimum(acceptance_time_df['acceptance_time_min'], time_cap_min)                                                       
        
        # exploration 
        # acceptance_time_df['acceptance_time_hrs'] = acceptance_time_df['acceptance_time_min']/60
        # acceptance_time_df[acceptance_time_df['acceptance_time_hrs'] <=7]['acceptance_time_hrs'].hist(bins=100)
        # acceptance_time_df['acceptance_time_hrs'].hist(bins=100)
        
        sol_df = pd.DataFrame(acceptance_time_df[['id_route','acceptance_time_min']])        
        features_df = routing_solution.features_df

        train_df = pd.merge(left = sol_df, right = features_df, how='left', on ='id_route')

        model = linear_model.Lasso(alpha=0.1)
        #linear_model.LassoLars(alpha=.1)
        #linear_model.Ridge(alpha=.5)
        x_df = train_df[train_df.columns.difference(['acceptance_time_min', 'id_route'])]
        model.fit(X = x_df , y = train_df['acceptance_time_min'] )
        
        #To print OLS summary  
        from statsmodels.api import OLS
        result = OLS(train_df['acceptance_time_min'],x_df).fit_regularized('sqrt_lasso')
        with open('summary.txt', 'w') as fh:
            fh.write(OLS(train_df['acceptance_time_min'],x_df).fit().summary().as_text())
        print(result.params)

        beta_dict = {col:model.coef_[i] for i,col in enumerate(x_df.columns)}
        return BetaMarket(beta_dict=beta_dict)

    def make_matching(self, routing_solution:RoutingSolution, market:MarketplaceInstance, 
                      prob_reference:float) -> MatchingSolution:
        # use Karp Algorithm to solve the bipartite minimum weight matching
        # https://networkx.org/documentation/stable/reference/algorithms/bipartite.html#module-networkx.algorithms.bipartite.matching
        # https://towardsdatascience.com/matching-of-bipartite-graphs-using-networkx-6d355b164567
        
        price_matrix = self.build_price_matrix(routing_solution, market, prob_reference)
        weighted_edges = [ (route_id, clouder_id, price) for (route_id, clouder_id), price in price_matrix.items()]
        graph = nx.Graph()
        graph.add_weighted_edges_from(weighted_edges)
        match_dict = nx.bipartite.minimum_weight_full_matching(graph)
        
        match_list:List[Tuple[int, int]] = []
        expected_price:Dict[int,float] = {}
        for route in routing_solution.routes:
            match_list.append((route.id, match_dict[route.id]))
            expected_price[route.id] = price_matrix[(route.id, match_dict[route.id])]
        return MatchingSolution(match = match_list, expected_price = expected_price)

     
    def build_price_matrix(self, routing_solution:RoutingSolution, market:MarketplaceInstance, 
                           prob_reference:float) -> Dict[Tuple[int,int],float]:
        """ Based on previous acceptance model fitted estimate the estimated price to pay 
            to all clouders in the market in order to get at least `prob_reference` probability
            of acceptance. 

        Args:
            routing_solution (RoutingSolution): Set of all routes to be evaluated
            market (MarketplaceInstance): Set of clouders to match with 
            prob_reference (float): min acceptance probability to infer the price

        Returns:
            Dict[Tuple[int,int],float]: `(route_id, clouder_id): route_price`,
        """                           
        price_matrix:Dict[Tuple[int,int],float] = {}
        for route, clouder in it.product(routing_solution.routes, market.clouders):
            price_matrix[(route.id, clouder.id)] = self.find_min_price_route_clouder(route, clouder, prob_reference)
        return price_matrix
    
    def build_price_matrix_plot(self, routing_solution:RoutingSolution, market:MarketplaceInstance, 
                                prob_reference:float) -> None:    
        price_matrix =  self.build_price_matrix(routing_solution, market, prob_reference)
        price_matrix_tuples = [(route_id, clouder_id, price) for (route_id, clouder_id), price in price_matrix.items()]
        price_matrix_df = pd.DataFrame(price_matrix_tuples).pivot(0,1,2)
        sns.heatmap(price_matrix_df, annot=False, annot_kws={"size": 7})
    
    def find_min_price_route_clouder(self,route:Route, clouder:Clouder, prob_reference:float) -> float:
        assert self.acceptance_model is not None, 'There must be a acceptance model fitted'
        
        samples = 25
        price_by_node_range =  np.linspace(start=800, stop=6000, num= samples) * route.ft_size
        route_feat = route.get_features_dict(self.acceptance_model_route_features)
        origin_distance =  clouder.ft_origin_distance(route)
        # build prediction dataframe
        predict_df = pd.DataFrame([route_feat for i in range(samples)])
        predict_df['route_price'] = price_by_node_range
        predict_df['origin_distance'] = origin_distance

        dpredict = xgb.DMatrix(predict_df[self.acceptance_model_route_features + 
                                          self.acceptance_model_clouder_features])
        predict_df['acceptance_prob'] = self.acceptance_model.predict(dpredict)
        
        sub_predict_df = predict_df[predict_df['acceptance_prob'] >= prob_reference]
        if len(sub_predict_df) >0:
            return sub_predict_df['route_price'].min() 
        else:
            # print(f'highest probability reached {predict_df["acceptance_prob"].max()} below threshold = {prob_reference}')
            return predict_df['route_price'].max()
