import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def mdape(y_true, y_pred):
	  return np.median((np.abs(np.subtract(y_true, y_pred)/ y_true))) * 100
# Load your preprocessed dataset
# Assuming a DataFrame `df` that includes columns for PM2.5, AOD, meteorological and land use variables,
# as well as a 'Location' column for random effects and 'Date' for indexing

# Example structure of df
# df = pd.DataFrame({
#     'Date': ['2020-01-01', '2020-01-02', ...],
#     'Location': ['Hanoi', 'Hanoi', ...],
#     'PM2_5': [12.4, 11.7, ...],
#     'AOD': [0.32, 0.30, ...],
#     'Temperature': [24, 25, ...],
#     'Humidity': [80, 82, ...],
#     'PBLH': [1200, 1150, ...],
#     'NDVI': [0.3, 0.32, ...],
#     'RoadDensity': [100, 102, ...]
# })

df = pd.read_csv("/mnt/disk2/ducanh/gap_filling/src/preprocess/data_mem.csv")
stations = ["Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan", "Gucheng", "Huairou", "Nongzhanguan", "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong"]
train_df = df.loc[(df["station"] == stations[0]) | (df["station"] == stations[1]) | (df["station"] == stations[2]) | 
                  (df["station"] == stations[3]) | (df["station"] == stations[4]) | (df["station"] == stations[5]) | (df["station"] == stations[6]) | (df["station"] == stations[7])]
valid_df = df.loc[(df["station"] == stations[5]) | (df["station"] == stations[6]) | (df["station"] == stations[7])]
test_df = df.loc[(df["station"] == stations[8]) | (df["station"] == stations[9]) | (df["station"] == stations[10]) | 
                  (df["station"] == stations[11])]
train_df = train_df.ffill().fillna(0)
test_df = test_df.ffill().fillna(0)
print("Max index in train_df:", train_df.index.max())
print("Max index in test_df:", test_df.index.max())
# breakpoint()
# model_formula = 'PM2_5 ~ AOD + black_carbon + dust + organic_carbon + sea_salt + sulfate + u_wind + v_wind + temperature + PBLH + pressure'
model_formula = 'PM2_5 ~ AOD + black_carbon + dust + organic_carbon + sea_salt + sulfate + u_wind + v_wind + temperature + PBLH + pressure'
mixed_lm = smf.mixedlm(model_formula, train_df, groups=train_df["station"], re_formula="~1")
result = mixed_lm.fit()
print(result.summary())
test_rs = result.predict(test_df)
test = {"PM2_5": [], "predicted_PM2_5": []}
test['PM2_5'] = test_df['PM2_5'].values
test['predicted_PM2_5'] = test_rs
# print(test[['PM2_5', 'predicted_PM2_5']].head())
mse = mean_squared_error(test['PM2_5'], test['predicted_PM2_5'])
mae = mean_absolute_error(test['PM2_5'], test['predicted_PM2_5'])
r2 = r2_score(test["PM2_5"], test["predicted_PM2_5"])
rmse = mean_squared_error(test["PM2_5"], test["predicted_PM2_5"], squared=False)
mape = mean_absolute_percentage_error(test["PM2_5"], test["predicted_PM2_5"])
corr = np.corrcoef(np.reshape(test["PM2_5"], (-1)), np.reshape(test["predicted_PM2_5"], (-1)))[0][1]
mdape_ = mdape(test["PM2_5"], test["predicted_PM2_5"])

print("MSE:", mse)
print("MAE:", mae)
print("R2:", r2)
print("RMSE:", rmse)
print("MAPE:", mape)
print("MDAPE:", mdape_)
print("CORR:", corr)