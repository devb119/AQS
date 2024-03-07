import pandas as pd 
import numpy as np
df = pd.read_csv("/mnt/disk2/ducanh/AQInterpolation/satellite_data/processed_data/merra-2/20130310.csv")
arr = df.values

arr = np.reshape(arr, (48,24,9))
new_arr = np.expand_dims(arr,1)
new_arr = np.repeat(new_arr,900,1)
position_arr = np.zeros((30,30,3))
for i in range(30):
            for j in range(30):
                position_arr[i,j][0] = i 
                position_arr[i,j][1] = j
                position_arr[i,j][2] = i * j
breakpoint()
