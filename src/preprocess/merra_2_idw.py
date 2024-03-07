import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import os
from tqdm.auto import tqdm

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

# Define the new grid resolution
new_resolution = 0.008333

for file in tqdm(os.scandir('/mnt/disk2/ducanh/AQInterpolation/satellite_data/processed_data/merra-2')):
    # Read data
    df = pd.read_csv(file)
    for time in tqdm(df['time'].unique()):
        data = df[df['time'] == time]
        # Calculate the bounds of the new grid
        lat_min, lat_max = data['lat'].min(), data['lat'].max()
        lon_min, lon_max = data['lon'].min(), data['lon'].max()

        # Create the new grid
        lat_new = np.arange(lat_min, lat_max, new_resolution)
        lon_new = np.arange(lon_min, lon_max, new_resolution)

        # Prepare the data for interpolation
        values = data.drop(columns=['lat', 'lon', 'time'])  # Exclude non-numeric and non-coordinate columns for interpolation

        # Prepare the grid coordinates for IDW interpolation
        xi, yi = np.meshgrid(lon_new, lat_new)

        # Perform IDW interpolation for each variable using 4 nearest points
        idw_interpolated_data = {}
        for column in values:
            z = data[column].values
            idw_interpolated_data[column] = inverse_distance_weighting(data['lon'].values, data['lat'].values, z, xi, yi, k=4)

        # create data array for each timestep
        arr = []
        arr.append(yi)
        arr.append(xi)
        import pdb; pdb.set_trace()
        # Loop through data value in sorted order
        # [lat, lon, 'AOD', 'black_carbon', 'dust', 'organic_carbon', 'sea_salt', 'sulfate']
        for key in sorted(idw_interpolated_data.keys()):
            arr.append(idw_interpolated_data[key])
        arr = np.array(arr)
        # np.save(f"/mnt/disk2/ducanh/gap_filling/processed_data/merra-2-idw/{time.split()[0]}.{time.split()[1]}.npy", arr)