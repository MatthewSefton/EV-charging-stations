import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import csv
import os
import random
from matplotlib.colors import LinearSegmentedColormap
import contextily as ctx
import geopandas as gpd
import matplotlib.patheffects as pe
import osmnx as ox
from shapely.geometry import Point

def generate_newcastle_traffic_heatmap():
    # Step 1: Parse the station reference CSV to get Newcastle stations and coordinates
    stations_file = 'road_traffic_counts_station_reference.csv'
    
    # Dictionary to store station information
    newcastle_stations = {}
    
    try:
        with open(stations_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check if the station is in Newcastle LGA
                if row['lga'] == 'Newcastle':
                    try:
                        station_id = row['station_id']
                        lat = float(row['wgs84_latitude'])
                        lon = float(row['wgs84_longitude'])
                        road_name = row['road_name']
                        newcastle_stations[station_id] = (lat, lon, road_name)
                    except (ValueError, KeyError) as e:
                        print(f"Error processing station: {e}")
                        continue
    except Exception as e:
        print(f"Error reading station file: {e}")
        return
    
    print(f"Found {len(newcastle_stations)} stations in Newcastle LGA")
    
    if not newcastle_stations:
        print("No stations found in Newcastle LGA. Exiting.")
        return
    
    # New: Read actual traffic count data from a file
    print("Reading traffic count data...")
    traffic_counts_file = 'road_traffic_counts_yearly_summary.csv'
    actual_traffic_counts = {}
    
    try:
        with open(traffic_counts_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    station_id = row['station_id']
                    if station_id in newcastle_stations and 'traffic_count' in row and row['traffic_count']:
                        count = float(row['traffic_count'])
                        # Store the actual traffic count
                        if station_id not in actual_traffic_counts or count > actual_traffic_counts[station_id]:
                            actual_traffic_counts[station_id] = count
                except (ValueError, KeyError) as e:
                    print(f"Error processing traffic count: {e}")
                    continue
        print(f"Found {len(actual_traffic_counts)} traffic count records for Newcastle stations")
    except Exception as e:
        print(f"Could not read traffic counts file: {e}")
        print("Continuing with synthetic data generation...")

    # Step 2: Generate traffic data, preferring actual counts when available
    print("Generating traffic data based on station locations and counts...")
    traffic_data = []
    
    # Generate traffic counts for each station
    center_lat = np.mean([station[0] for station in newcastle_stations.values()])
    center_lon = np.mean([station[1] for station in newcastle_stations.values()])
    
    for station_id, (lat, lon, road_name) in newcastle_stations.items():
        # Use actual traffic count if available, otherwise generate synthetic data
        if station_id in actual_traffic_counts:
            count = actual_traffic_counts[station_id]
            print(f"Station {station_id} ({road_name}): {count:.1f} vehicles (ACTUAL DATA)")
        else:
            distance_from_center = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
            road_factor = 1.0
            major_road_keywords = ['highway', 'pacific', 'main', 'newcastle', 'king', 'hunter']
            if any(keyword in road_name.lower() for keyword in major_road_keywords):
                road_factor = 2.0
            
            base_count = 5000 * road_factor / (1 + 10 * distance_from_center)
            count = base_count * (0.7 + 0.6 * random.random())
            print(f"Station {station_id} ({road_name}): {count:.1f} vehicles (SYNTHETIC)")
        
        traffic_data.append((lat, lon, count))
    
    if not traffic_data:
        print("Failed to generate traffic data. Exiting.")
        return
    
    print(f"Generated {len(traffic_data)} traffic data points for Newcastle")
    
    # Step 3: Create a grid for the heatmap
    traffic_array = np.array(traffic_data)
    lats = traffic_array[:, 0]
    lons = traffic_array[:, 1]
    counts = traffic_array[:, 2]
    
    lat_min, lat_max = np.min(lats), np.max(lats)
    lon_min, lon_max = np.min(lons), np.max(lons)
    
    lat_pad = (lat_max - lat_min) * 0.15
    lon_pad = (lon_max - lon_min) * 0.15
    lat_min -= lat_pad
    lat_max += lat_pad
    lon_min -= lon_pad
    lon_max += lon_pad
    
    grid_size = 500
    lat_grid = np.linspace(lat_min, lat_max, grid_size)
    lon_grid = np.linspace(lon_min, lon_max, grid_size)
    
    traffic_grid = np.zeros((grid_size, grid_size))
    
    # Adjust intensity calculation in the grid loop
    for lat, lon, count in traffic_data:
        i_center = int((lat - lat_min) / (lat_max - lat_min) * (grid_size - 1))
        j_center = int((lon - lon_min) / (lon_max - lon_min) * (grid_size - 1))
        
        radius = 30
        for i in range(max(0, i_center - radius), min(grid_size, i_center + radius + 1)):
            for j in range(max(0, j_center - radius), min(grid_size, j_center + radius + 1)):
                distance = np.sqrt((i - i_center)**2 + (j - j_center)**2)
                if distance <= radius:
                    # Reduce the intensity multiplier to smooth out extreme variations
                    intensity = count * np.exp(-0.5 * (distance / radius)**2) * 1.2
                    traffic_grid[i, j] += intensity
    
    # Apply a stronger smoothing filter
    smoothed_grid = gaussian_filter(traffic_grid, sigma=15)
    
    # Normalize the smoothed grid to reduce unevenness
    smoothed_grid = (smoothed_grid - np.min(smoothed_grid)) / (np.max(smoothed_grid) - np.min(smoothed_grid))
    
    # NEW: Export the map data for use in Main.py
    export_map_data_for_main(lon_grid, lat_grid, smoothed_grid)
    
    # Step 5: Create the heatmap visualization
    plt.figure(figsize=(15, 12))
    colors = [(0, 0.8, 0, 0.6), (1, 1, 0, 0.7), (1, 0.5, 0, 0.8), (1, 0, 0, 0.9)]
    cmap_name = 'traffic_density'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    
    ax = plt.gca()
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    
    satellite_added = False
    try:
        ctx.add_basemap(ax, crs='EPSG:4326')
        
        satellite_added = True
        print("Added Stamen Toner Lite basemap successfully")
    except Exception as e:
        print(f"Could not add Stamen Toner Lite basemap: {e}")
    
    if satellite_added:
        try:
            print("Fetching road network data from OpenStreetMap...")
            center_lat = (lat_min + lat_max) / 2
            center_lon = (lon_min + lon_max) / 2
            
            lat_distance = 111000 * (lat_max - lat_min)
            lon_distance = 111000 * np.cos(np.radians(center_lat)) * (lon_max - lon_min)
            dist = max(lat_distance, lon_distance) / 2
            
            G = ox.graph_from_point((center_lat, center_lon), dist=dist, network_type='drive')
            edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
            
            traffic_gdf = gpd.GeoDataFrame(
                {'count': counts},
                geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
                crs="EPSG:4326"
            )
            
            min_count = min(counts)
            max_count = max(counts)
            count_range = max_count - min_count if max_count > min_count else 1
            
            
            
            # Green (low) -> Yellow (medium) -> Red (high)
            traffic_cmap = plt.cm.get_cmap('RdYlGn_r')
            
            # Create a custom traffic colormap: Green -> Orange -> Red
            traffic_colors = [(0, 0.8, 0), (1, 0.5, 0), (0.8, 0, 0)]  # Green -> Orange -> Red
            traffic_cmap = LinearSegmentedColormap.from_list('traffic_cmap', traffic_colors, N=100)

            # First, draw all roads in light green as base (light traffic)
            print("Drawing base road network in green...")
            for _, edge in edges.iterrows():
                ax.plot(*edge['geometry'].xy, color='green', linewidth=0.8, alpha=0.5,
                        solid_capstyle='round', zorder=1)
            
            print("Mapping traffic data to road network...")
            # Dictionary to track roads that have been processed with traffic data
            processed_roads = {}
            
            for i, (lat, lon, count) in enumerate(traffic_data):
                point = Point(lon, lat)
                station_influence_radius = 0.01  # Increased radius for longer road highlighting
                
                # Find roads within the influence radius
                nearby_roads = []
                for idx, edge in edges.iterrows():
                    distance = edge['geometry'].distance(point)
                    if distance < station_influence_radius:
                        nearby_roads.append((idx, edge, distance))
                
                if not nearby_roads:
                    continue
                
                # Sort by distance to prioritize closest roads
                nearby_roads.sort(key=lambda x: x[2])
                norm_count = (count - min_count) / count_range
                
                color = traffic_cmap(norm_count)
                
                # Highlight roads with traffic color, fading with distance
                for idx, edge, distance in nearby_roads:
                    # Skip if already processed with higher traffic (lower distance)
                    edge_key = str(idx)
                    if edge_key in processed_roads and processed_roads[edge_key] < distance:
                        continue
                    
                    # Calculate fade based on distance
                    fade_factor = 1.0 - (distance / station_influence_radius)
                    

                    
                    # Adjust alpha based on distance
                    alpha = 0.8 * fade_factor
                    
                    # Draw the road with traffic color
                    ax.plot(*edge['geometry'].xy, color=color, linewidth=1, alpha=alpha,
                            solid_capstyle='round', zorder=3,
                            path_effects=[pe.Stroke(linewidth=1.0, foreground='black', alpha=0.3),
                                         pe.Normal()])
                    
                    # Mark road as processed
                    processed_roads[edge_key] = distance
            
            print("Successfully added traffic-based road heat map")
        except ImportError:
            print("Could not import required libraries for road highlighting.")
            print("Install with: pip install geopandas osmnx shapely")
        except Exception as e:
            print(f"Could not create road heat map: {str(e)}")
    
    heatmap = ax.imshow(
        smoothed_grid,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin='lower',
        cmap=cm,
        alpha=0.4,
        aspect='auto'
    )
        
    cbar = plt.colorbar(heatmap, label='Estimated Traffic Density')
    cbar.set_label('Estimated Traffic Density', size=14)
    plt.title('Traffic Density Heatmap - Newcastle (Satellite View)', fontsize=18)
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    
    station_lats = [newcastle_stations[sid][0] for sid in newcastle_stations]
    station_lons = [newcastle_stations[sid][1] for sid in newcastle_stations]
    plt.scatter(station_lons, station_lats, c='white', s=25, alpha=0.9, 
                edgecolor='black', linewidth=0.7, label='Traffic Stations')
    
    plt.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig('newcastle_traffic_satellite_map.png', dpi=300)
    print("Satellite heatmap saved as 'newcastle_traffic_satellite_map.png'")
    
    plt.figure(figsize=(12, 10))
    plt.imshow(
        smoothed_grid,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin='lower',
        cmap=cm,
        aspect='auto'
    )
    plt.colorbar(label='Estimated Traffic Density')
    plt.title('Traffic Density Heatmap - Newcastle LGA (Clean View)', fontsize=16)
    plt.scatter(station_lons, station_lats, c='blue', s=20, alpha=0.8, label='Traffic Stations')
    plt.legend(loc='upper right')
    plt.savefig('newcastle_traffic_heatmap_clean.png', dpi=300)
    print("Clean heatmap saved as 'newcastle_traffic_heatmap_clean.png'")
    
    plt.close('all')

# Add this new function to export the data
def export_map_data_for_main(lon_grid, lat_grid, traffic_values):
    """
    Export the map data in a format usable by Main.py
    
    Parameters:
    -----------
    lon_grid : array-like
        Grid of longitude values
    lat_grid : array-like
        Grid of latitude values
    traffic_values : 2D array
        Traffic density values for each point in the grid
    """
    output_dir = '../map_data'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'newcastle_traffic_map.npz')
    X, Y = np.meshgrid(lon_grid, lat_grid)
    Z = traffic_values.T  # Transpose for correct orientation
    
    # Ensure Z is normalized
    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
    
    np.savez(output_file, X=X, Y=Y, Z=Z)
    print(f"Map data exported to {output_file} for use in Main.py")
    
    csv_file = os.path.join(output_dir, 'newcastle_traffic_map.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['X', 'Y', 'Z'])
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                writer.writerow([X[i,j], Y[i,j], Z[i,j]])
    print(f"Map data also exported as CSV to {csv_file}")

if __name__ == "__main__":
    generate_newcastle_traffic_heatmap()
