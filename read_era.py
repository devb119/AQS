import numpy as np
import math 
from tqdm import tqdm

import pathlib

position_arr = np.zeros((30 * 6,30 * 8,2))
for i in range(30 * 6):
    for j in range(30 * 8):
        position_arr[i,j][0] = i 
        position_arr[i,j][1] = j

coor_arr = np.zeros((30 * 6,30 * 8,2))
coor_arr[:,:,0] = 39.625 + 0.25  * position_arr[:,:,0] / 30
coor_arr[:,:,1] = 115.5 + 0.25  * position_arr[:,:,1] / 30

import os
data_list = ['2 metre temperature', '10 metre U wind component', '10 metre V wind component', 'Boundary layer height', 'Surface pressure', 'Total precipitation']


for month in range(1,13):
    for day in range(1,32):
        for hour in range(24):
            list_arr = []
            for metro_name in data_list:
                try:
                    arr =np.load(f"processed_data/era5/{metro_name}/2015_{month}_{day}_{hour}_0_0.npy")
                    new_value = np.zeros((30 * 6,30 * 8,1))
                    for i in tqdm(range(180)):
                        for j in range(240):
                            lat,lon = coor_arr[i,j,0], coor_arr[i,j,1]
                            a = math.floor(4 * (40.7 - lat))
                            b = math.ceil(4 * (lon - 113.79 ))
                            new_value[i,j,:] = arr[2,a,b]
                    print("Add to list")
                    list_arr.append(new_value)
                except:
                    pass
            if len(list_arr) !=0:
                saved_data = np.concatenate(list_arr,-1)
            
            breakpoint()
            if not os.path.exists(f"converto_acoording_merra_combined/era5/"):
                os.makedirs(f"converto_acoording_merra_combined/era5/")
            np_filepath = f"converto_acoording_merra_combined/era5/2015_{month}_{day}_{hour}.npy"
            np.save(np_filepath, saved_data)
breakpoint()