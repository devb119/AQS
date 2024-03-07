import os
from tqdm.auto import tqdm
import numpy as np

# [lat, lon, 'AOD', 'black_carbon', 'dust', 'organic_carbon', 'sea_salt', 'sulfate']
# ['lat', 'lon', '10 metre U wind component', '10 metre V wind component', '2 metre temperature', 'Boundary layer height', 'Surface pressure']
min_max = {"min": [9999999]*6, "max": [-9999999]*6}
for file in tqdm(os.scandir("/mnt/disk2/ducanh/gap_filling/processed_data/merra-2-idw")):
    data = np.load(file.path)
    for i in range(2, 7):
        feature_values = data[i,:,:].flatten()
        if min_max["min"][i - 2] > min(feature_values):
            min_max["min"][i - 2] = min(feature_values)
        if min_max["max"][i -2] < max(feature_values):
            min_max["max"][i - 2] = max(feature_values)

import json
with open("min_max_era.json", "w") as file:
    json.dump(min_max, file)