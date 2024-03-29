from src.utils.data_loader import AQDataset
from src.models.linear import AttentionEncoder
from src.models.fcn import FCN
from src.models.stdgi import Attention_STDGI
from src.models.combine_1_loss import Combine1Loss
from src.models.decoder import Decoder
from src.utils.args import get_options
from torch.utils.data import DataLoader
from src.utils.utils import *
from tqdm.auto import tqdm
from src.utils.early_stopping import EarlyStopping
from src.utils.train_func import *
from src.utils.counter import Counter
from src.utils.test_func import test_1_loss, cal_acc
from time import time

import matplotlib.pyplot as plt
import json
import torch.nn as nn
import torch
import logging
import os
import wandb


if __name__ == '__main__':
    ############################### SETTING UP DEVICE AND ARGS ##########################################
    args = get_options()
    seed_everything(args.seed)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s :: %(levelname)s :: %(message)s"
    )
    with open("min_max_merra.json", "r") as f:
        merra_dict = json.load(f)
    with open("min_max_era.json", "r") as f:
        era_dict = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        config = vars(args)
    except IOError as msg:
        args.error(str(msg))
    
    ############################### PREPROCESS AND BUILD DATASET ##########################################
    # Combine all df to a array
    list_arr, location_, station, corr = get_data_array(args)
    
    # Scale into (-1,1)
    data_arr, climate_arr, scaler = preprocess_pipeline(list_arr, args)
    
    train_dataset = AQDataset(data_arr, climate_arr, location_, 12, merra_dict, era_dict, args=args)
    
    # import pdb; pdb.set_trace()
    
    
    stdgi = Attention_STDGI(
        in_ft=args.input_dim,
        out_ft=args.output_stdgi,
        en_hid1=args.en_hid1,
        en_hid2=args.en_hid2,
        dis_hid=args.dis_hid,
        stdgi_noise_min=args.stdgi_noise_min,
        stdgi_noise_max=args.stdgi_noise_max,
        num_input_station=args.num_input_station,
    ).to(device)
    l2_coef = 0.0
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    stdgi_optimizer_e = torch.optim.Adam(
        stdgi.encoder.parameters(), lr=args.lr_stdgi, weight_decay=l2_coef
    )
    
    stdgi_optimizer_d = torch.optim.Adam(
        stdgi.disc.parameters(), lr=args.lr_stdgi, weight_decay=l2_coef
    )
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    
    if not os.path.exists(f"output/{args.group_name}/checkpoint/"):
        print(f"Make dir output/{args.group_name}/checkpoint/ ...")
        os.makedirs(f"output/{args.group_name}/checkpoint/")
    
    early_stopping_stdgi = EarlyStopping(
        patience=args.patience,
        verbose=True,
        delta=args.delta_stdgi,
        path=f"output/{args.group_name}/checkpoint/stdgi_{args.name}.pt",
    )
    wandb_dict = {}
    ####### Initial wandb #######
    if args.use_wandb:
        wandb.init(
            entity="aiotlab",
            project="AQ_Interpolation",
            group=args.group_name,
            name=f"{args.name}",
            config=config,
        )
    
    # import pdb; pdb.set_trace()
    ###### Train STDGI to get GCN ##########
    start_stdgi = time()
    stdgi.train()
    for i in range(args.num_epochs_stdgi):
        if early_stopping_stdgi.early_stop:
            break
        loss = train_atten_stdgi(
                    stdgi,
                    train_dataloader,
                    stdgi_optimizer_e,
                    stdgi_optimizer_d,
                    bce_loss,
                    device,
                    n_steps=2,
                    n_iterations = 50,
                    early_stopping_stdgi = early_stopping_stdgi,
                    args = args
                )
    ## Load  best stdgi model
    # load_model(stdgi, f"/mnt/disk2/ducanh/gap_filling/output/ver1_beijing/checkpoint/stdgi_gap_filling.pt")
    end_stdgi = time()
    # load_model(stdgi, f"output/{args.group_name}/checkpoint/stdgi_{args.name[:-4]}.pt")
    load_model(stdgi, f"output/{args.group_name}/checkpoint/stdgi_{args.name}.pt")
    print(f"Stdgi training time: {end_stdgi - start_stdgi} seconds")
    
    # Training with decoder
    feature_linear = AttentionEncoder(in_features=12, out_features=64, num_hidden_units=256, query_dim=11, atten_mode="feature").to(device)
    temporal_linear = AttentionEncoder(in_features=args.satellite_in_features, out_features=64, num_hidden_units=256, query_dim=11, atten_mode="temporal").to(device)
    fcn = FCN(in_features=args.satellite_in_features, out_features=64, num_hidden_units=256).to(device)
    if args.satellite_handler == "fcn" or args.satellite_handler == "temporal_att" or args.satellite_handler == "temporal_att":
        decoder_in_ft = 128
    elif args.satellite_handler == "gnn":
        decoder_in_ft = 64
    elif args.satellite_handler == "concat":
        decoder_in_ft = 192
    decoder = Decoder(in_ft=decoder_in_ft, out_ft=1, fc_hid_dim=args.decoder_hid, cnn_hid_dim=args.decoder_hid).to(device)
    combined_model = Combine1Loss(stdgi.encoder, feature_linear, temporal_linear, fcn, decoder, args.satellite_handler)
    optimizer_combined_model = torch.optim.Adam(combined_model.parameters(), lr= args.lr_stdgi)
    # schedular = torch.optim.lr_scheduler.StepLR(optimizer_combined_model, step_size=1)
    
    test_dataloaders = []
    for test_station in args.valid_station:
        test_dataset  = AQDataset(data_df= data_arr, climate_df=climate_arr, location_df=location_, seq_len=12,
                                    valid=True,test=False,args=args,test_station=test_station,
                                    merra_scaler=merra_dict,
                                    era_scaler=era_dict)
        test_dataloader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
            )
        test_dataloaders.append(test_dataloader)
        
    print("Start training combine 1 loss model")
    early_stopping_decoder = EarlyStopping(
        patience=args.patience,
        verbose=True,
        delta=args.delta_decoder,
        path=f"output/{args.group_name}/checkpoint/decoder_{args.name}.pt",
    )
    
    iteration_counter = Counter()
    # Load pretrained L2
    # load_model(combined_model, 'output/ver1_beijing/checkpoint/decoder_train_L2.pt')
    start_combine = time()
    for ep in range(args.decoder_epochs):
        if not early_stopping_decoder.early_stop:
            training_loss = train_1_loss(combined_model, 
                                                        train_dataloader, 
                                                        test_dataloaders, 
                                                        optimizer_combined_model, 
                                                        device, 
                                                        args,
                                                        early_stopping_decoder, 
                                                        scaler,
                                                        iteration_counter)
            # schedular.step()
            
        if args.use_wandb:
            wandb.log({"epoch_loss": training_loss})
    end_combine = time()
    print(f"main model training time: {end_combine - start_combine} seconds")
    load_model(combined_model, f"output/{args.group_name}/checkpoint/decoder_{args.name}.pt")
    # test
    list_acc = []
    predict = {}
    
    print("Start testing")
    start_test = time()
    for test_station in args.test_station:
        test_dataset  = AQDataset(data_df= data_arr, climate_df=climate_arr, location_df=location_, seq_len=12,
                                    valid=False,test=True,args=args,test_station=test_station,
                                    merra_scaler=merra_dict,
                                    era_scaler=era_dict)

        test_dataloader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
        testing_loss, list_prd, list_grt = test_1_loss(combined_model, test_dataloader, device, scaler)
        output_arr = np.concatenate(
            (np.array(list_grt).reshape(-1, 1), np.array(list_prd).reshape(-1, 1)),
            axis=1,
        )
        out_df = pd.DataFrame(output_arr, columns=["ground_truth", "prediction"])
        out_dir = "output/{}/".format(args.dataset)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        out_df.to_csv(out_dir + f"Station_{test_station}_{args.name}.csv")
        mae, mse, mape, rmse, r2, corr_, mdape = cal_acc(list_prd, list_grt)
        list_acc.append([test_station, mae, mse, mape, mdape, rmse, r2, corr_])
        predict[test_station] = {"grt": list_grt, "prd": list_prd}
        print("Test Accuracy: {}".format(mae, mse, corr))

    end_test = time()
    print(f"Test time: {end_test - start_test} seconds")
    for test_station in args.test_station:
        df = pd.DataFrame(data=predict[test_station], columns=["grt", "prd"])
        if args.use_wandb:
            wandb.log({f"Station_{test_station}": df})
    tmp = np.array(list_acc).mean(0)
    list_acc.append(tmp)
    df = pd.DataFrame(
        np.array(list_acc),
        columns=["STATION", "MAE", "MSE", "MAPE", "MDAPE", "RMSE", "R2", "CORR"],
    )
    # saved_log = "_".join(args.features)
    saved_log = args.group_name
    import os
    if not os.path.exists(f"log_infor/{saved_log}"):
        os.makedirs(f"log_infor/{saved_log}")
    df.to_csv(f"log_infor/{saved_log}/acc_{args.name}.csv")
    print(df)
    if args.use_wandb:
        wandb.log({"test_acc": df})
    for test_station in args.test_station:
        prd = predict[test_station]["prd"]
        grt = predict[test_station]["grt"]

        df_stat = pd.DataFrame({"Predict": prd, "Groundtruth": grt})
        x = len(grt)
        fig, ax = plt.subplots(figsize=(40, 8))
        # ax.figure(figsize=(20,8))
        ax.plot(np.arange(x), grt[:x], label="grt")
        ax.plot(np.arange(x), prd[:x], label="prd")
        ax.legend()
        ax.set_title(f"Tram_{test_station}")
        if args.use_wandb:
            wandb.log({"Tram_{}".format(test_station): wandb.Image(fig)})
            wandb.log({"Tram_{}_pred_gt".format(test_station): df_stat})
    if args.use_wandb:
        wandb.finish()