from src.utils.args import get_options
from src.utils.utils import *
import logging
import wandb
# from src.models.stdgi import Attention_STDGI
from src.models.stdgi import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import tqdm 
from src.utils.early_stopping import EarlyStopping
import os
from src.models.decoder import Decoder, CombineModel
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_absolute_error,
)
import numpy as np

def mdape(y_true, y_pred):
	return np.median((np.abs(np.subtract(y_true, y_pred)/ y_true))) * 100

def cal_acc(y_prd, y_grt):
    mae = mean_absolute_error(y_grt, y_prd)
    mse = mean_squared_error(y_grt, y_prd, squared=True)
    mape = mean_absolute_percentage_error(y_grt, y_prd)
    rmse = mean_squared_error(y_grt, y_prd, squared=False)
    corr = np.corrcoef(np.reshape(y_grt, (-1)), np.reshape(y_prd, (-1)))[0][1]
    r2 = r2_score(y_grt, y_prd)
    mdape_ = mdape(y_grt,y_prd)
    return mae, mse, mape, rmse, r2, corr,mdape_


def test_combined_model(combined_model, dataloader, device, scaler,args):
    combined_model.eval()
    combined_loss_mean = []
    list_prd = []
    list_grt = []
    with torch.no_grad():
        mse_loss = torch.nn.MSELoss()
        for data in tqdm(dataloader):
            # breakpoint()
            x, G, l, cnn_x, y = data['X'].float().to(device), data['G'].float().to(device), data['l'].float().to(device), data['X_satellite'].to(device), data['Y'].float().to(device)
            lat_index, lon_index = data['lat_lon']
            data = (x,G,l, cnn_x, lat_index, lon_index, y)

            # Test normal flow
            # output = combined_model.test(cnn_x, lat_index, lon_index)
            # list_prd += torch.squeeze(output).cpu().detach().tolist()
            # list_grt += torch.squeeze(y).cpu().detach().tolist()
            # loss = mse_loss(torch.reshape(output, y.shape), y)
            # combined_loss_mean.append(loss.item())
            
            # Test for GNN only
            gnn_representation = combined_model.get_idw_representation(x,G,l)
            gnn_prediction = combined_model.get_predicted_gnn(gnn_representation)
            list_prd += torch.squeeze(gnn_prediction).cpu().detach().tolist()
            list_grt += torch.squeeze(y).cpu().detach().tolist()
            loss = mse_loss(torch.squeeze(gnn_prediction,1), y)
            combined_loss_mean.append(loss.item())
        # if args.use_wandb:
        #     wandb.step({"testing loss": loss.item()})

        a_max = scaler.data_max_[0]
        a_min = scaler.data_min_[0]
        list_grt = (np.array(list_grt) + 1) / 2 * (a_max - a_min) + a_min
        list_prd = (np.array(list_prd) + 1) / 2 * (a_max - a_min) + a_min
        list_grt_ = [float(i) for i in list_grt]
        list_prd_ = [float(i) for i in list_prd]

    return sum(combined_loss_mean)/len(combined_loss_mean), list_prd_, list_grt_

def test_1_loss(combined_model, dataloader, device, scaler):
    combined_model.eval()
    combined_loss_mean = []
    list_prd = []
    list_grt = []
    with torch.no_grad():
        mse_loss = torch.nn.MSELoss()
        for data in tqdm(dataloader):
            # breakpoint()
            x, G, l, cnn_x, y = data['X'].float().to(device), data['G'].float().to(device), data['l'].float().to(device), data['X_satellite'].to(device), data['Y'].float().to(device)
            lat_index, lon_index = data['lat_lon']
            data = (x,G,l, cnn_x, lat_index, lon_index, y)

            output = combined_model(data)
            list_prd += torch.squeeze(output).cpu().detach().tolist()
            list_grt += torch.squeeze(y).cpu().detach().tolist()
            loss = mse_loss(torch.reshape(output, y.shape), y)
            combined_loss_mean.append(loss.item())

        a_max = scaler.data_max_[0]
        a_min = scaler.data_min_[0]
        list_grt = (np.array(list_grt) + 1) / 2 * (a_max - a_min) + a_min
        list_prd = (np.array(list_prd) + 1) / 2 * (a_max - a_min) + a_min
        list_grt_ = [float(i) for i in list_grt]
        list_prd_ = [float(i) for i in list_prd]

    return sum(combined_loss_mean)/len(combined_loss_mean), list_prd_, list_grt_

def test_cnn_combine(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    scaler,
    ):
    model.to(device)
    model.eval()
    list_prd = []
    list_grt = []
    combined_loss_mean = []
    with torch.inference_mode():
        mse_loss = nn.MSELoss()
        for data in tqdm(dataloader):
            X, y = data["X_satellite"].type(torch.float).to(device), data["Y"].type(torch.float).to(device)
            grid_lat, grid_lon = data["lat_lon"]
            output = model.test(X, grid_lat, grid_lon)
            list_prd += torch.squeeze(output).cpu().detach().tolist()
            list_grt += torch.squeeze(y).cpu().detach().tolist()
            loss = mse_loss(output, y)
            combined_loss_mean.append(loss.item())
            # import pdb; pdb.set_trace()
        # if args.use_wandb:
        #     wandb.step({"testing loss": loss.item()})    
        a_max = scaler.data_max_[0]
        a_min = scaler.data_min_[0]
        list_grt = (np.array(list_grt) + 1) / 2 * (a_max - a_min) + a_min
        list_prd = (np.array(list_prd) + 1) / 2 * (a_max - a_min) + a_min
        list_grt_ = [float(i) for i in list_grt]
        list_prd_ = [float(i) for i in list_prd]
        print("Test done")
    return sum(combined_loss_mean)/len(combined_loss_mean), list_prd_, list_grt_