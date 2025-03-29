#Solution class, contains a set of stations and a total value (which can only be calcualated from within the main class)
#Has functions for mutating, sexually reproducing and cloning (asexual reproduction)
import numpy as np
import random
from ChargingStation import ChargingStation
class Solution:
    
    def __init__(self, stations, totalValue):
        self.stations=stations
        self.totalValue = totalValue

    #minor mutatation involves a slight shift (shift_amount) in x,y co-ordinates. Major mutation involves a complete randomisation of a station.
    #The bounds of the map have to also be passed into the function. So if your map is 200, bounds should be 50 (i.e, -50)
    def mutate(self, minor_mutation_rate=0.2, shift_amount=10, major_mutation_rate=0.025,bounds=50):
        for station in self.stations:
            if random.random() < minor_mutation_rate:
                dx = random.uniform(-shift_amount, shift_amount)
                dy = random.uniform(-shift_amount, shift_amount)
                station.x = min(max(station.x + dx, -bounds), bounds)
                station.y = min(max(station.y + dy, -bounds), bounds)
            if random.random()<major_mutation_rate:
                station.x = random.uniform(-bounds, bounds)
                station.y = random.uniform(-bounds, bounds)



    #Let's get it on! Reproduction time! We're talking about SEX people
    def sexual_reproduction(self, other):
        # Mix stations: take half from self, half from other
        split = len(self.stations) // 2
        child_stations = self.stations[:split] + other.stations[split:]

        # Deep copy stations to avoid shared references
        new_stations = [ChargingStation(s.x, s.y, s.radius) for s in child_stations]

        # Return a new solution with un-evaluated value (set to 0, will evaluate after)
        return Solution(new_stations, totalValue=0.0)

    def clone(self):
        # Deep copy all stations
        new_stations = [ChargingStation(s.x, s.y, s.radius) for s in self.stations]
        return Solution(new_stations, self.totalValue)