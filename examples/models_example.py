from typing import Optional
from pydantic import BaseModel
from shapely.geometry import  Point

class CustomPoint(Point):

    def __init__(self, *args):
        super().__init__(*args)

    @property 
    def lat(self):
        return self.x
    @property 
    def lng(self):
        return self.y

class Drop(BaseModel, CustomPoint):
    date: Optional[str]


a = Drop(0.1,0.1)
b = Drop(0.2,0.2)
print(a.distance(b))