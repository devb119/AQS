import argparse

def get_options():
    parser = argparse.ArgumentParser()
    
    #Global config
    parser.add_argument("--seed", type=int, default= 42)
    
    #Dataset config
    parser.add_argument("--dataset", type=str, default="beijing")
    parser.add_argument("--train_pct", default=0.6, type=float)
    parser.add_argument("--test_pct", default=0.2, type=float)
    parser.add_argument("--seq_len", type=int, default= 12)
    
    ############################ Architecture config ######################################
    
    ##Stdgi architecture
    parser.add_argument("--output_stdgi", type=int, default= 64)
    parser.add_argument("--en_hid1", type=int, default=64)
    parser.add_argument("--en_hid2", type=int,default=64)
    parser.add_argument("--decoder_hid", type=int,default=256)
    parser.add_argument("--dis_hid", type=int, default=64)
    parser.add_argument("--stdgi_noise_min", type=float, default=0.4)
    parser.add_argument("--stdgi_noise_max", type=float, default=0.7)
    
    ## Satellite architecture
    parser.add_argument("--satellite_in_features", type=int, default= 11)
    parser.add_argument("--satellite_hid", type=int, default= 256)
    parser.add_argument("--decoder_epochs", type=int, default = 50)
    parser.add_argument("--satellite_handler", type=str, default="temporal_att", choices=["temporal_att", "feature_att", "concat", "gnn"])
    
    #Wandb config
    parser.add_argument("--group_name", type=str, default="Test group")
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--name", type=str, default= "Test")
    #Optimizer config
    # parser
    parser.add_argument("--lr_stdgi", type=float, default= 0.001)
    parser.add_argument("--lr_combine", type=float, default= 0.001)
    parser.add_argument("--batch_size", type=int, default= 32)
    parser.add_argument("--num_epochs_stdgi",type=int, default= 10)
    parser.add_argument("--num_epochs_cnn",type=int, default= 10)
    parser.add_argument("--num_epochs_linear",type=int, default= 10)
    parser.add_argument("--n_iterations", type= int , default= 50)
    # Early stopping config
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--delta_stdgi", type=float, default=0.01)
    parser.add_argument("--delta_decoder", type=float, default=0.01)
    
    args = parser.parse_args()
    
    args.session_name = f"Run test"
    
    if args.dataset == "uk":
        args.data_path = "/mnt/disk2/ducanh/AQInterpolation/station_data/uk/preprocessed"
        args.data_dir = "/mnt/disk2/ducanh/gap_filling/processed_data/uk/merra"
        args.era5_dir = "/mnt/disk2/ducanh/gap_filling/processed_data/uk/era/merged_idw"
        args.train_station = [0,1,2,3,4,5,6,7,8,9]
        args.valid_station = [10,11,12,13]
        args.test_station = [14,15,16,17]
        
        args.features = ["PM2.5","PM10","NO2","O3","TEMP","wd","WSPM"]
        args.idx_climate = 4
        args.input_dim = len(args.features)
        args.num_input_station = 10
    elif args.dataset == "beijing":
        args.data_dir = "/mnt/disk2/ducanh/gap_filling/processed_data/crop/merra/"
        args.era5_dir = "/mnt/disk2/ducanh/gap_filling/processed_data/crop/era/"
        args.data_path = "/mnt/disk2/ducanh/AQInterpolation/station_data/beijing/"
        args.train_station = [0,1,2,3,4]
        args.valid_station = [5,6,7]
        args.test_station = [8,9,10,11]
        
        # args.features = ["PM2.5","PM10","SO2","NO2","CO", "O3","PRES", "RAIN", "TEMP","WSPM", 'black_carbon', 'dust', 'organic_carbon', 'SO2', 'sea_salt', 'AOD']
        args.features = ["PM2.5","PM10","SO2","NO2","CO", "O3","PRES", "RAIN", "TEMP","WSPM", "DEWP"]
        args.satellite_features = ['AOD', 'black_carbon', 'dust', 'organic_carbon', 'sea_salt', 'sulfate', '10 metre U wind component', '10 metre V wind component', '2 metre temperature', 'Boundary layer height', 'Surface pressure']
        args.idx_climate = 6
        args.input_dim = len(args.features)
        # args.input_dim = len(args.features) + 8
        args.num_input_station = 5
    else:
        raise("Not correct dataset name")

    return args

