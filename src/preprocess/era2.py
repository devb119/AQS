import xarray as xr
import matplotlib.pyplot as plt
import pygrib
from tqdm import tqdm
import csv
import os 
import numpy as np
grib_file_path = "/mnt/disk2/ducanh/gap_filling/gap-filling/ERA-5/adaptor.mars.internal-1708372465.3820732-3085-6-b27cd5f6-8574-408b-8d80-ba60dea7f917.grib"

# def 
# Open the GRIB file
grbs = pygrib.open(grib_file_path)
# Iterate through all the messages in the GRIB file
list_stored = []
list_name = ['Surface pressure', 'Boundary layer height', 'Total precipitation', 'Total precipitation', '10 meter V wind component', '10 metre U wind component', '2 metre temperature']


for grb in tqdm(grbs):

    # Accessing specific data, for example, the data values and lat/lon coordinates
    
    data = grb.values
    list_name.append(grb.name)
    lats, lons = grb.latlons()
    date = grb.dataDate
    time = grb.dataTime
    name = grb.name
    if not os.path.exists(f"processed_data/era5/{name}"):
        os.makedirs(f"processed_data/era5/{name}")
    
    np_filepath = f"processed_data/era5/{name}/{date}.npy"
    if os.path.exists(np_filepath):
        raise("Path is existing")
    saved_data = np.stack([lats, lons, data])
    np.save(np_filepath, saved_data)

