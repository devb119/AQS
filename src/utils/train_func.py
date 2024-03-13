import numpy as np
import pandas as pd 
import torch
from tqdm import tqdm
from tqdm.auto import tqdm 
from torch import nn
from torch.utils.data import DataLoader
import wandb
from typing import List
from src.utils.test_func import test_cnn_combine, test_combined_model, test_1_loss

def train_atten_stdgi(
    stdgi, dataloader, optim_e, optim_d, criterion, device, n_steps=2, n_iterations= 200, early_stopping_stdgi= None, args= None
):
    # wandb.watch(stdgi, criterion, log="all", log_freq=100)
    '''
    Sử dụng train Attention_STDGI model 
    '''
    epoch_loss = 0
    iteration_loss = 0
    stdgi.train()
    count = 0
    
    for data in tqdm(dataloader): 
        if not early_stopping_stdgi.early_stop:
            count += 1
            for i in range(n_steps):
                optim_d.zero_grad()
                x = data["X"].to(device).float()
                G = data["G"][:,:,:,:,0].to(device).float()  

                output = stdgi(x, x, G)
                lbl_1 = torch.ones(output.shape[0], output.shape[1], 1)
                lbl_2 = torch.zeros(output.shape[0], output.shape[1], 1)
                lbl = torch.cat((lbl_1, lbl_2), -1).to(device)
                d_loss = criterion(output, lbl)
                d_loss.backward()
                optim_d.step()

            optim_e.zero_grad()
            x = data["X"].to(device).float()
            G = data["G"][:,:,:,:,0].to(device).float()  
            output = stdgi(x, x, G)
            lbl_1 = torch.ones(output.shape[0], output.shape[1], 1)
            lbl_2 = torch.zeros(output.shape[0], output.shape[1], 1)
            lbl = torch.cat((lbl_1, lbl_2), -1).to(device)
            e_loss = criterion(output, lbl)
            e_loss.backward()
            optim_e.step()
            epoch_loss += e_loss.detach().cpu().item()
            iteration_loss += e_loss.detach().cpu().item()

            if count % n_iterations == 0:
                print(iteration_loss)
                iteration_loss = iteration_loss/ n_iterations

                early_stopping_stdgi(iteration_loss, stdgi)
                if args.use_wandb:
                    wandb.log({"Loss/ Training stdgi loss": iteration_loss})  
                iteration_loss = 0
        else:
            break
    return epoch_loss / len(dataloader)
    

def train_combined_model(combined_model, dataloader, test_dataloaders, optimizer, device, args, epoch, early_stopping, scaler, iteration_counter):
    combined_model.train()
    combined_loss_mean = []
    epoch_loss1, epoch_loss2, epoch_loss3 = 0, 0, 0
    for data in tqdm(dataloader):
        if not early_stopping.early_stop:
            x, G, l, cnn_x, y = data['X'].float().to(device), data['G'].float().to(device), data['l'].float().to(device), data['X_satellite'].to(device), data['Y'].float().to(device)
            lat_index, lon_index = data['lat_lon']
            data = (x,G,l, cnn_x, lat_index, lon_index, y)
            weights = (0.3, 0.5, 0.2)
            # # Increase L3 loss to learn representation
            # if epoch >= 10:
            #     weights = (0.5, 0.45, 0.05)
            
            # Test GNN only
            weights = (0, 1, 0)
            optimizer.zero_grad()
            loss_dict = combined_model.get_combined_loss(weights, data)
            loss1 = loss_dict['loss1']
            loss2 = loss_dict['loss2']
            loss3 = loss_dict['loss3']
            combined_loss = loss_dict['combined_loss']
            combined_loss.backward()
            optimizer.step()
            combined_loss_mean.append(combined_loss)
            iteration_counter.increase()
            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()
            epoch_loss3 += loss3.item()
            if args.use_wandb:
                if iteration_counter.value % args.n_iterations == 0:
                    ## Valid
                    list_testing_loss = []
                    for test_dataloader in test_dataloaders:
                        testing_loss, list_prd, list_grt = test_combined_model(combined_model, test_dataloader, device, scaler, args)
                        list_testing_loss.append(testing_loss)
                    mean_testing_loss = sum(list_testing_loss)/ len(list_testing_loss)
                    early_stopping(mean_testing_loss, combined_model)
                    
                    # Logging
                    wandb.log({"iteration_loss1": loss1.item(),
                            "iteration_loss2": loss2.item(),
                            "iteration_loss3": loss3.item(),
                            "iteration_combined_loss": combined_loss.item(),
                            "valid_loss": mean_testing_loss})
        else:
            break

    # Return epoch summary results
    return sum(combined_loss_mean)/len(combined_loss_mean), {"Training/loss1": epoch_loss1 / len(dataloader),
                "Training/loss2": epoch_loss2 / len(dataloader),
                "Training/loss3": epoch_loss3 / len(dataloader),
                "Training/combined_loss": combined_loss.item()}

def train_1_loss(combined_model, dataloader, test_dataloaders, optimizer, device, args, early_stopping, scaler, iteration_counter):
    combined_model.train()
    epoch_loss = 0
    for data in tqdm(dataloader):
        if not early_stopping.early_stop:
            x, G, l, x_satellite, y = data['X'].to(torch.float32).to(device), data['G'].to(torch.float32).to(device), data['l'].to(torch.float32).to(device), data['X_satellite'].to(device), data['Y'].to(torch.float32).to(device)
            lat_index, lon_index = data['lat_lon']
            data = (x,G,l, x_satellite, lat_index, lon_index, y)
            loss = combined_model.get_loss(data)
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            epoch_loss += loss
            iteration_counter.increase()
            if args.use_wandb:
                if iteration_counter.value % args.n_iterations == 0:
                    ## Valid
                    list_testing_loss = []
                    for test_dataloader in test_dataloaders:
                        testing_loss, list_prd, list_grt = test_1_loss(combined_model, test_dataloader, device, scaler)
                        list_testing_loss.append(testing_loss)
                    mean_testing_loss = sum(list_testing_loss)/ len(list_testing_loss)
                    early_stopping(mean_testing_loss, combined_model)
                    wandb.log({"valid_loss": mean_testing_loss,
                            "train_loss": loss.item()})
        else:
            break

    # Return epoch summary results
    return epoch_loss/len(dataloader)


def train_cnn_combine(
    combined_model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloaders: List[DataLoader],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    early_stopping = None,
    scaler=None,
    args = None
    ):
    combined_model.to(device)
    combined_model.train()
    # Init loss
    train_loss = 0
    count = 0
    iteration_loss = 0
    # Training loop
    for data in train_dataloader:
        count += 1
        # Send data to GPU
        X, y = data["X_satellite"].type(torch.float).to(device), data["Y"].type(torch.float).to(device)
        grid_lat, grid_lon = data["lat_lon"]
        
        # Compute loss after forward passing through CNN model and decoder model
        loss = combined_model.get_loss(X, grid_lat, grid_lon, y)
        train_loss += loss.detach().cpu().item()
        iteration_loss += loss.detach().cpu().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        # Log loss every n_iterations
        if count % args.n_iterations == 0:
            count = 0
            iteration_loss /= args.n_iterations
            
            list_testing_loss = []
            for test_dataloader in test_dataloaders:
                testing_loss, list_prd, list_grt = test_cnn_combine(combined_model, test_dataloader, device, scaler, args)
                list_testing_loss.append(testing_loss)
            mean_testing_loss = sum(list_testing_loss)/ len(list_testing_loss)
            if args.use_wandb:
                wandb.log({"Training/Iteration training loss CNN": iteration_loss} )
                wandb.log({"Training/Iteration valid loss CNN": mean_testing_loss} )
            iteration_loss = 0
            early_stopping(mean_testing_loss, combined_model)
    
    # Calculate aggregated training loss
    train_loss /= len(train_dataloader)
    return train_loss