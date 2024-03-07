import xarray as xr
import matplotlib.pyplot as plt
import pygrib
from tqdm import tqdm
import csv
import os 
import numpy as np
grib_file_path = "/mnt/disk2/ducanh/AQInterpolation/satellite_data/gap-filling/ERA-5/2017.grib"

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
    if not os.path.exists(f"/mnt/disk2/ducanh/gap_filling/processed_data/era5_extracted/{name}"):
        os.makedirs(f"/mnt/disk2/ducanh/gap_filling/processed_data/era5_extracted/{name}")
    count = 0
    np_filepath = f"/mnt/disk2/ducanh/gap_filling/processed_data/era5_extracted/{name}/{grb.year}_{grb.month}_{grb.day}_{grb.hour}_{grb.minute}_{count}.npy"
    # count = 1
    while(True):
        if os.path.exists(np_filepath):
            count += 1
            np_filepath = f"/mnt/disk2/ducanh/gap_filling/processed_data/era5_extracted/{name}/{grb.year}_{grb.month}_{grb.day}_{grb.hour}_{grb.minute}_{count}.npy"
        else:
            break
        #     # breakpoint()

            
    saved_data = np.stack([lats, lons, data])
    np.save(np_filepath, saved_data)

