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
    
    def get_coverage_value(self, Z_padded, X_padded, Y_padded, mask):
        # Find the closest grid indices to the station's (x, y)
        dist = np.sqrt((X_padded - self.x)**2 + (Y_padded - self.y)**2)
        center_i, center_j = np.unravel_index(np.argmin(dist), dist.shape)

        r = self.radius
        # Extract the patch from the heightmap
        Z_patch = Z_padded[center_i - r:center_i + r + 1, center_j - r:center_j + r + 1]

        # Apply the mask
        total_value = Z_patch[mask].sum()
        return total_value