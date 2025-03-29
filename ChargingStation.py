#A charging station class. has methods for returning the sum of all points within its radius
import numpy as np
class ChargingStation:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = int(radius)

    def area(self):
        return 3.14159 * self.radius ** 2

    def __str__(self):
        return f"Circle at ({self.x}, {self.y}) with radius {self.radius}"
    