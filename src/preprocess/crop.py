import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

lat_min, lat_max = (39.625, 40.875)
lon_min, lon_max = (115.5, 117.25)
new_resolution = 0.008333
def find_closest_grid_index(station_lat, station_lon, location="beijing"):
    if location == "beijing":
        lat_range = np.arange(lat_min, lat_max, new_resolution)
        lon_range = np.arange(lon_min, lon_max, new_resolution)
    elif location == "uk":
        lat_range = np.arange(50.0, 58.5, 0.05)  
        lon_range = np.arange(-7.5, 1.875, 0.05) 
    lat_index = min(range(len(lat_range)), key=lambda i: abs(lat_range[i] - station_lat))
    lon_index = min(range(len(lon_range)), key=lambda i: abs(lon_range[i] - station_lon))
    return lat_index, lon_index

df = pd.read_csv("/mnt/disk2/ducanh/AQInterpolation/station_data/beijing/locations.csv")
max_lat = max(df['lat'].values)
max_lon = max(df['lon'].values)
min_lat = min(df['lat'].values)
min_lon = min(df['lon'].values)

max_idx = find_closest_grid_index(max_lat, max_lon)
min_idx = find_closest_grid_index(min_lat, min_lon)
for file in tqdm(os.scandir("/mnt/disk2/ducanh/gap_filling/processed_data/era5_extracted/merged_idw")):
    arr = np.load(file.path)
    cropped = arr[:, min_idx[0]:max_idx[0] + 1, min_idx[1]:max_idx[1] + 1]
    import pdb; pdb.set_trace()
    np.save("/mnt/disk2/ducanh/gap_filling/processed_data/crop/era/" + file.name, cropped)