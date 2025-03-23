import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from ChargingStation import ChargingStation
import random

# Constants
size = 200  # Grid resolution
chargingStationRadius=10
padding = chargingStationRadius*2       #Here so calculating values inside circles don't go out of bounds
numberOfChargingStations=8

# Generate a random heightmap
Z = np.random.rand(size, size)  # Random values between 0 and 1

# Smooth the terrain using a Gaussian filter
Z = gaussian_filter(Z, sigma=15)  # Adjust sigma for smoothness
Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z)) 

# Create a grid
x_range = np.linspace(-50, 50, size)
y_range = np.linspace(-50, 50, size)
X, Y = np.meshgrid(x_range, y_range)
X_padded = np.pad(X, pad_width=padding, mode='edge')
Y_padded = np.pad(Y, pad_width=padding, mode='edge')
Z_padded = np.pad(Z, pad_width=padding, mode='constant', constant_values=0)

#define a mask (This is used to efficiently calculate the the sum of values within the radius of a charging station)
def create_circular_mask(radius):
    radius= int(radius)
    diameter = radius * 2 + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    return mask

circle_mask = create_circular_mask(chargingStationRadius)

# Plot the terrain
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X, Y, Z, levels=20, cmap="jet")
plt.colorbar(contour, label="Height")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Randomly Generated Contour Plot")

# ✅ Add a Charging Station as a circle
station = ChargingStation(x=-49, y=-49, radius=chargingStationRadius)
value = station.get_coverage_value(Z_padded, X_padded, Y_padded, circle_mask)
print("Coverage value:", value)

#display circle

# Keep circles proportional
ax.set_aspect('equal')




def createSetOfChargingStations(numberOfChargingStations, radius, mapsize):
    stations = []

    #The +-1 here is to stop the circles from being on the edge of the map, causing error. 
    min_coord = -mapsize / 4 + 1
    max_coord = mapsize / 4 - 1

    while numberOfChargingStations > 0:
        x = random.randint(int(min_coord), int(max_coord))
        y = random.randint(int(min_coord), int(max_coord))
        station = ChargingStation(x, y, radius)
        stations.append(station)
        numberOfChargingStations-= 1

    return stations

stations = createSetOfChargingStations(numberOfChargingStations, chargingStationRadius, size)

for s in stations:
    value = s.get_coverage_value(Z_padded, X_padded, Y_padded, circle_mask)
    print(f"Station at ({s.x:.2f}, {s.y:.2f}) → value: {value:.3f}")
    circle = plt.Circle((s.x, s.y), s.radius, color='black', fill=False, linewidth=1)
    ax.add_patch(circle)

plt.show()