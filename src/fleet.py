from entities import Boat
from typing import List

class Fleet:
    def __init__(self, boats: List[Boat]):
        self.boats = boats
    
    def sync_read(self):
        for boat in self.boats:
            boat.measure_sonar()
            