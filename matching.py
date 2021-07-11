from pydantic import BaseModel
from typing import Dict

class MarketplaceInstance(BaseModel):
    
    @classmethod
    def build_simulation(cls, sim_params):
        return cls()
    
    @staticmethod
    def betas_from_time_df(delivery_instance, cost_df)-> Dict[str,float]:
        delivery_instance.get_features()
        beta_dict = {}
        return beta_dict

    def get_betas(self) -> Dict[str,float]:
        beta_dict = {}
        return beta_dict
    
class Abra(BaseModel):
    pass
