import os
import pathlib
import json
def get_file_path_dict():
    path_dict = {}
    dir_path = '/mnt/disk2/ducanh/AQInterpolation/satellite_data/processed_data/merra-2'
    for i, file in enumerate(sorted(list(pathlib.Path(dir_path).glob("2015*.csv")))):
        path_dict[i] = f"/mnt/disk2/ducanh/AQInterpolation/satellite_data/processed_data/merra-2/{file.name}"
    with open("merra_map_dict.json","w") as f:
        json.dump(path_dict, f)
    # return path_dict


get_file_path_dict()