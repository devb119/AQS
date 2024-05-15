import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
)
from datetime import datetime, timedelta
import torch

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_data_array(args):
    file_path = args.data_path
    columns = args.features    
    location_df = pd.read_csv(file_path + "locations.csv")
    
    station = location_df["station"].values
    location = location_df.values[:, 1:]
    location_ = location[:, [0, 1]]

    list_arr = []
    for i in station:
        df = pd.read_csv(file_path + f"{i}.csv")[columns]
        df = df.fillna(method="ffill")
        arr = df.astype(float).values
        arr = np.expand_dims(arr, axis=1)
        list_arr.append(arr)
    list_arr = np.concatenate(list_arr, axis=1)
    pm2_5 = list_arr[:,:,0]
    corr = pd.DataFrame(pm2_5).corr().values
    del df 
    del arr
    del location
    del location_df
    # breakpoint()
    return list_arr, location_, station, corr


def preprocess_pipeline(df, args):
    scaler = MinMaxScaler((-1, 1))
    (a, b, c) = df.shape
    res = np.reshape(df, (-1, c))
    
    # Cut ourlier points 
    for i in range(c):
        threshold = np.percentile(res[:, i], 95)
        res[:, i] = np.where(res[:, i] > threshold, threshold, res[:, i])

    res_ = scaler.fit_transform(res)
    # gan lai wind_angle cho scaler
    res_aq = res_.copy()
    res_climate = res_.copy()
    res_aq = np.reshape(res_aq, (-1, b, c))
    res_climate = np.reshape(res_climate, (-1, b, c))
    # res = np.reshape(res, (-1, b, c))
    idx_climate = args.idx_climate
    trans_df = res_aq[:, :, :]
    climate_df = res_climate[:, :, idx_climate:] # bo feature cuoi vi k quan tam huong gio
    # Fill nan for data_arr
    for i in range(12):
        df = pd.DataFrame(trans_df[:, i, :])
        df = df.ffill()
        trans_df[:, i, :] = df.to_numpy()
    del res_aq
    del res_climate 
    del res
    return np.nan_to_num(trans_df), climate_df, scaler

def get_date_from_index(index, dataset="hourly"):
    if dataset == "uk" or dataset == "beijing":
        base_date = datetime(2013, 3, 1, 0, 0)  # Base date: 1/3/2013 0h00
        time_to_add = timedelta(hours=index)  # Calculate timedelta based on index
    elif dataset == "daily":
        base_date = datetime(2017, 1, 1, 0, 0)  # Base date: 1/1/2017 0h00
        time_to_add = timedelta(days=index)  # Calculate timedelta based on index
    elif dataset == "india":
        base_date = datetime(2018, 2, 1, 11, 0)
        time_to_add = timedelta(hours=index)
    rs = base_date + time_to_add
    d1 = datetime(2020, 1, 1, 0, 0)
    if rs >= d1 and dataset == "daily":
        rs = datetime(2019, 12, 31, 0, 0)
    return rs  # Return the calculated datetime object


new_resolution = 0.008333
def find_closest_grid_index(station_lat, station_lon, location="beijing"):
    if location == "beijing":
        lat_min, lat_max = (39.625, 40.875)
        lon_min, lon_max = (115.5, 117.25)
        lat_range = np.arange(lat_min, lat_max, new_resolution)[31:84]
        lon_range = np.arange(lon_min, lon_max, new_resolution)[82:138]
    elif location == "uk" or location == "uk_old":
        return 0, 0
    elif location == "india":
        lat_min, lat_max = (28.45, 28.9)
        lon_min, lon_max = (76.9, 77.4)
        lat_range = np.arange(lat_min, lat_max, new_resolution)
        lon_range = np.arange(lon_min, lon_max, new_resolution)
    lat_index = min(range(len(lat_range)), key=lambda i: abs(lat_range[i] - station_lat))
    lon_index = min(range(len(lon_range)), key=lambda i: abs(lon_range[i] - station_lon))
    return lat_index, lon_index

def save_checkpoint(model, path):
    checkpoints = {
        "model_dict": model.state_dict(),
    }
    torch.save(checkpoints, path)

def load_model(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path)["model_dict"])