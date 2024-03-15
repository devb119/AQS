import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

X = []
MAE = []
MSE = []
MAPE = []
MDAPE = []
RMSE = []
R2 = []
CORR = []
accs = {"MAE": [], "MSE": [], "MAPE": [], "MDAPE": [], "RMSE": [], "R2": [], "CORR": []}
for file in os.scandir("/mnt/disk2/ducanh/gap_filling/log_infor/satellite_feature_selection"):
    if file.name.startswith("acc_self"):
        feature_name = file.name.split(".")[0][19:]
        X.append(feature_name)
        acc = pd.read_csv(file)
        # import pdb; pdb.set_trace()
        for key in accs.keys():
            accs[key].append(acc[key][4])

acc_main = pd.read_csv("/mnt/disk2/ducanh/gap_filling/log_infor/PM2.5_PM10_SO2_NO2_CO_O3_PRES_RAIN_TEMP_WSPM_DEWP/acc_gap_filling.csv")
acc_no_satellite = pd.read_csv("/mnt/disk2/ducanh/gap_filling/log_infor/PM2.5_PM10_SO2_NO2_CO_O3_PRES_RAIN_TEMP_WSPM_DEWP/acc_gap_filling_gnn.csv")
X.append("full")
X.append("no_satellite")
for key in accs.keys():
    accs[key].append(acc_main[key][4])
    accs[key].append(acc_no_satellite[key][4])

acc_name = sys.argv[1]
np_acc = np.array(accs[acc_name])
sorted_indices = np.argsort(np_acc)
np_X = np.array(X)

# import pdb; pdb.set_trace()
X_axis = np.arange(len(X))
plt.figure(figsize=(12,8))
plt.barh(X_axis, np_acc[sorted_indices], 0.4, label = acc_name)
plt.yticks(X_axis, np_X[sorted_indices])
plt.ylabel("Experiment names")
plt.title(f"{acc_name} metrics")
plt.legend()
plt.savefig(f"./attention/acc_{acc_name}.png")