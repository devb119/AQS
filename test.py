from src.utils.data_loader import AQDataloader
import json

with open("merra_map_dict.json", "r") as f:
    data_dict = json.load(f)
    

dataset = AQDataloader(None, data_dict)
samples = dataset[0]

breakpoint()