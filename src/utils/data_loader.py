from torch.utils.data import Dataset
from src.utils.utils import *
import random

import pandas as pd 
import numpy as np

class AQDataset(Dataset):
    def __init__(
        self,
        data_df,
        climate_df,
        location_df,
        input_len,
        merra_scaler,
        era_scaler,
        test=False,
        valid=False,
        args=None,
        test_station = None,
        ):
        super().__init__()
        self.args = args
        self.merra_scaler = merra_scaler
        self.era_scaler = era_scaler
        
        self.position_arr = self.init_position()
        
        self.test_station = test_station
        self.train_station = args.train_station
        
        assert not (test and test_station == None), "pha test yeu cau nhap tram test"
        assert not (
            test_station in args.train_station
        ), "Train station id couldn't in list of testing station"
        
        self.features = ['AOD', 'black_carbon', 'dust', 'organic_carbon', 'sea_salt', 'sulfate']
        self.dataset = args.dataset
        self.satellite_in_features = args.satellite_in_features
        
        self.input_len = input_len
        self.test = test
        self.valid = valid
        self.data_df = data_df
        
        self.location = location_df
        self.climate_df = climate_df
        self.n_st = len(self.train_station) - 1
        
        self.train_cpt = args.train_pct
        self.test_cpt = args.test_pct
        self.n_samples = len(data_df)
        
        # Satellite data set up 
        self.data_dir = args.data_dir
        self.era5_dir = args.era5_dir
        
        # phan data train thi khong lien quan gi den data test 
        self.X_train = data_df[:int(self.n_samples * self.train_cpt)]
        self.X_valid = data_df[int(self.n_samples * self.train_cpt): - int(self.n_samples * self.test_cpt)]
        self.X_test = data_df[- int(self.n_samples * self.test_cpt) : ]
        # self.climate_train = climate_df[:int(self.n_samples * self.train_cpt)]
        
        X_satellite = list(range(self.n_samples))
        self.X_satellite_train = X_satellite[:int(self.n_samples * self.train_cpt)]
        self.X_satellite_valid = X_satellite[int(self.n_samples * self.train_cpt): - int(self.n_samples * self.test_cpt)]
        self.X_satellite_test = X_satellite[- int(self.n_samples * self.test_cpt) : ]
        # test data
        if self.test:
            print("Initial test dataset")
            # self.climate_test = climate_df[ int(self.n_samples * self.test_cpt) :, self.test_station,: ]
            self.X = self.X_test[:, self.train_station,:]

            self.l_test = self.get_reverse_distance_matrix(
                self.train_station, self.test_station
            ) 
            self.Y_test = self.X_test[:, self.test_station, 0]
            self.G_test = self.get_adjacency_matrix(self.train_station)
            # self.cli = self.climate_test
                
                
        elif self.valid:
            print("Initial valid dataset")
            # self.climate_valid = climate_df[int(self.n_samples * self.train_cpt): int(self.n_samples * self.test_cpt), self.test_station,:]
            # phan data test khong lien quan gi data train 
            self.X = self.X_valid[:, self.train_station,:]
            self.l_test = self.get_reverse_distance_matrix(
                self.train_station, self.test_station
            )
            self.Y_test = self.X_valid[:, self.test_station, 0]
            self.G_test = self.get_adjacency_matrix(self.train_station)
            # self.cli = self.climate_valid
            
    def get_distance(self, coords_1, coords_2):
        import geopy.distance
        return geopy.distance.geodesic(coords_1, coords_2).km

    def get_distance_matrix(self, list_col_train_int, target_station):
        matrix = []
        for i in list_col_train_int:
            matrix.append(
                self.get_distance(self.location[i], self.location[target_station])
            )
        res = np.array(matrix)
        return res
    
    def get_reverse_distance_matrix(self, list_col_train_int, target_station):
        distance_matrix = self.get_distance_matrix(list_col_train_int, target_station)
        reverse_matrix = 1 / distance_matrix
        return reverse_matrix / reverse_matrix.sum()

    def get_adjacency_matrix(self, list_col_train_int, target_station_int=None):
        adjacency_matrix = []
        for j, i in enumerate(list_col_train_int):
            distance_matrix = self.get_distance_matrix(list_col_train_int, i)
            distance_matrix[j] += 15
            reverse_dis = 1 / distance_matrix
            adjacency_matrix.append(reverse_dis / reverse_dis.sum())
        adjacency_matrix = np.array(adjacency_matrix)
        adjacency_matrix = np.expand_dims(adjacency_matrix, 0)
        adjacency_matrix = np.repeat(adjacency_matrix, self.input_len, 0)
        return adjacency_matrix
    
    def init_position(self):
        position_arr = np.zeros((30 * 6,30 * 8,3))
        for i in range(30 * 6):
            for j in range(30 * 8):
                position_arr[i,j][0] = i 
                position_arr[i,j][1] = j
                position_arr[i,j][2] = i * j
        return np.reshape(position_arr,(-1,3))
    
    # [lat, lon, 'AOD', 'black_carbon', 'dust', 'organic_carbon', 'sea_salt', 'sulfate']
    def load_merra(self, index):
        if self.test:
            dataset_index = self.X_satellite_test[index]
        elif self.valid:
            dataset_index = self.X_satellite_valid[index]
        else:
            dataset_index = self.X_satellite_train[index]
        date_obj = get_date_from_index(dataset_index)
        filename = date_obj.strftime("%Y-%m-%d.%H:30:00.npy")
        arr = np.load(self.data_dir + filename)
        for i in range(2, 8):
            min_val = self.merra_scaler['min'][i - 2]
            max_val = self.merra_scaler['max'][i - 2]
            arr[i, :, :] = ((arr[i, :, :] - min_val) * (1 - (-1)) / (max_val - min_val)) + (-1)
        # Remove one variable along the first dimension for feature selection
        # Keep first 2 vars (lat, lon), skip AOD, keep the remainings
        # new_arr = np.concatenate((arr[:6], arr[7:]), axis=0)
        # new_arr = arr[:7]
        return arr
    
    # ['lat', 'lon', '10 metre U wind component', '10 metre V wind component', '2 metre temperature', 'Boundary layer height', 'Surface pressure']
    def load_era5(self, index):
        if self.test:
            dataset_index = self.X_satellite_test[index]
        elif self.valid:
            dataset_index = self.X_satellite_valid[index]
        else:
            dataset_index = self.X_satellite_train[index]
        date_obj = get_date_from_index(dataset_index)
        filename = date_obj.strftime("%Y_%m_%d_%H_00_00.npy")
        arr = np.load(self.era5_dir + filename)
        for i in range(2, 7):
            min_val = self.era_scaler['min'][i - 2]
            max_val = self.era_scaler['max'][i - 2]
            arr[i, :, :] = ((arr[i, :, :] - min_val) * (1 - (-1)) / (max_val - min_val)) + (-1)
        # new_arr = np.concatenate((arr[:5], arr[6:]), axis=0)
        new_arr = arr[:6]
        return new_arr

    def convert_data_to_Cartesian(self, arr):
        new_arr = np.expand_dims(arr,1)
        new_arr = np.repeat(new_arr,900,1)
        new_arr = np.reshape(new_arr,(-1,new_arr.shape[-1]))
        
        return np.concatenate((new_arr, self.position_arr),-1)
        
    
    def __getitem__(self,index: int):
        merra = self.load_merra(index)
        era = self.load_era5(index)
        concat_data = np.concatenate((merra, era[2:,:,:]), axis=0)
        
        list_G = []
        if self.test or self.valid:
            x = self.X[index : index + self.input_len, :]  
            y = self.Y_test[index + self.input_len - 1]  # float
            G = self.G_test #shape [12,5,5]
            l = self.l_test # shape (5,)
            lat_index, lon_index = find_closest_grid_index(self.location[self.test_station][0], self.location[self.test_station][1])
            list_G = [G]
        else:
            # random select a station in list of train station, this station is chosen as target station for training process
            picked_target_station_int = random.choice(self.train_station)
            lat_index, lon_index = find_closest_grid_index(self.location[picked_target_station_int][0], self.location[picked_target_station_int][1])
            list_selected_train_station = list(
                set(self.train_station) - set([picked_target_station_int])
            )
            x = self.X_train[index : index + self.input_len, list_selected_train_station, :]
            
            y = self.X_train[index + self.input_len - 1, picked_target_station_int, 0]
            
            # climate = self.climate_train[
            #     index + self.input_len - 1, picked_target_station_int, :
            # ]
            G = self.get_adjacency_matrix(
                list_selected_train_station, picked_target_station_int
            )
            list_G = [G]
            l = self.get_reverse_distance_matrix(
                list_selected_train_station, picked_target_station_int
            )
        sample = {
            "X": x,
            "merra": merra,
            "era": era,
            "X_satellite": concat_data,
            "Y": np.array([y]),
            "l": np.array(l),
            "climate": 1,
            "lat_lon": (lat_index, lon_index)
        }
        
        sample["G"] = np.stack(list_G,-1)
        
        return sample
    
    def __len__(self,):
        if self.test:
            
            return len(self.X_satellite_test) - self.input_len
        if self.valid:
            return len(self.X_satellite_valid) - self.input_len
        else:
            # return 18532
            return len(self.X_satellite_train) - self.input_len