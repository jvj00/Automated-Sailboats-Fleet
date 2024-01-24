from typing import Optional
from entities.boat import Boat
from entities.wind import Wind
from environment import SeabedMap


class World:
    def __init__(self, gravity, wind: Wind, seabed: Optional[SeabedMap] = None):
        self.gravity_z = gravity
        self.wind = wind
        self.seabed = seabed
    
    def update(self, boats: list[Boat], dt):
        for b in boats:
            b.apply_forces(self.wind, dt)
            b.move(dt)
            b.apply_acceleration_to_velocity(dt)
            b.apply_friction(self.gravity_z, dt)
