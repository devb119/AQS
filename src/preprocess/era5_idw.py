import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm
from scipy.spatial import cKDTree
import json

def inverse_distance_weighting(x, y, z, xi, yi, k=4):
    """
    Perform Inverse Distance Weighting (IDW) interpolation for 2D data.

    Parameters:
    - x, y: Arrays of x and y coordinates of the data points.
    - z: Array of values at the data points.
    - xi, yi: Arrays of x and y coordinates where interpolated values are sought.
    - k: Number of nearest neighbors to consider for weighting.

    Returns:
    - zi: Array of interpolated values at (xi, yi).
    """
    # Create a KDTree for efficient nearest neighbor search
    tree = cKDTree(list(zip(x, y)))
    # Find the k nearest neighbors for each (xi, yi) point
    distances, indices = tree.query(list(zip(xi.ravel(), yi.ravel())), k=k)
    
    # Handle the case where a point is exactly at a data point to avoid division by zero
    distances[distances == 0] = 1e-10
    
    # Calculate the inverse of the distances
    weights = 1 / distances
    
    # Calculate the weighted average of values
    zi = np.sum(weights * z[indices], axis=1) / np.sum(weights, axis=1)
    zi = zi.reshape(xi.shape)
    
    return zi

def format_name(old_name):
    filename_arr = old_name.split("_")
    filename_arr[1] = filename_arr[1].zfill(2)
    filename_arr[2] = filename_arr[2].zfill(2)
    filename_arr[3] = filename_arr[3].zfill(2)
    filename_arr[4] = filename_arr[4].zfill(2)
    filename_arr[5] = filename_arr[5].zfill(6)
    new_name = "_".join(filename_arr)
    return new_name

# Grid bounds
lat_min, lat_max = (39.625, 40.875)
lon_min, lon_max = (115.5, 117.25)

# Define the new grid resolution
new_resolution = 0.008333

# Create list of other features to merge into 1 file
features = sorted(folder.name for folder in os.scandir('/mnt/disk2/ducanh/gap_filling/processed_data/era5_extracted'))
# Remove feature that is currently looping through
features.remove('2 metre temperature')
features.remove('merged_idw')
features.remove('Total precipitation')

i = 0
# Order of data in dimension 0:
fields = ['lat', 'lon', '2 metre temperature', '10 metre U wind component', '10 metre V wind component', 'Boundary layer height', 'Surface pressure']
era5_path = "/mnt/disk2/ducanh/gap_filling/processed_data/era5_extracted/"

for file in tqdm(os.scandir(era5_path + "2 metre temperature/")):
    # Read the 2 metre temperature data
    data = np.load(file)
    # Merge other features to the same grid
    for other_feature in features:
        other_feature_data = np.load(f"{era5_path}{other_feature}/{file.name}")
        other_data = other_feature_data[2, :, :].reshape(1, 13, 24)
        data = np.concatenate((data, other_data), axis=0)
    
    # After having merged, merged data have shape (8, 13, 24)
    # Having the data of 1 timestep ready, now we use idw to interpolate data
    lat_new = np.arange(lat_min, lat_max, new_resolution)
    lon_new = np.arange(lon_min, lon_max, new_resolution)
    
    # Prepare the grid coordinates for IDW interpolation
    xi, yi = np.meshgrid(lon_new, lat_new)
    
    # Recreate dataframe in the same form of merra_2
    count = 0
    df = pd.DataFrame(columns=fields)
    for i in range(13):
        for j in range(24):
            df.loc[count] = data[:, i, j]
            count += 1
    
    df = df.sort_values(by=['lat', 'lon'])
    # Prepare the data for interpolation
    values = df.drop(columns=['lat', 'lon'])  # Exclude non-numeric and non-coordinate columns for interpolation
    
    # Perform IDW interpolation for each variable using 4 nearest points
    idw_interpolated_data = {}
    for column in values:
        z = df[column].values
        idw_interpolated_data[column] = inverse_distance_weighting(df['lon'].values, df['lat'].values, z, xi, yi, k=4)
    # create data array for each timestep
    arr = []
    arr.append(yi)
    arr.append(xi)
    
    # ['lat', 'lon', '10 metre U wind component', '10 metre V wind component', '2 metre temperature', 'Boundary layer height', 'Surface pressure']
    for key in sorted(idw_interpolated_data.keys()):
        arr.append(idw_interpolated_data[key])
    arr = np.array(arr)
    np.save(f"/mnt/disk2/ducanh/gap_filling/processed_data/era5_extracted/merged_idw/{format_name(file.name)}", arr)