import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def mdape(y_true, y_pred):
	  return np.median((np.abs(np.subtract(y_true, y_pred)/ y_true))) * 100
stations = ["Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan", "Gucheng", "Huairou", "Nongzhanguan", "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong"]
# Step 1: Data Preparation
# Creating a synthetic dataset using Pandas DataFrame
np.random.seed(42)  # for reproducibility
n_samples = 1000
n_features = 11
feature_names = [f"feature_{i}" for i in range(n_features)]

# Generate random data as feature matrix (X) and target vector (y)
df = pd.read_csv("/mnt/disk2/ducanh/gap_filling/src/preprocess/data_rf_17k.csv")
df = df.ffill().fillna(0)
X = df.copy()
# X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=feature_names)
y = df[["station", "PM2_5"]].copy()

# Split the data into training and testing sets
# X_train = X.loc[(X["station"] == stations[0]) | (X["station"] == stations[1]) | (X["station"] == stations[2]) | 
#                   (X["station"] == stations[3]) | (X["station"] == stations[4]) | (X["station"] == stations[5]) | (X["station"] == stations[6]) | (X["station"] == stations[7])].drop(columns=['PM2_5', 'station', 'year'])
X_train = X.loc[(X["station"] == stations[0]) | (X["station"] == stations[1]) | (X["station"] == stations[2]) | 
                  (X["station"] == stations[3]) | (X["station"] == stations[4])].drop(columns=['PM2_5', 'station', 'year'])

# y_train = y.loc[(y["station"] == stations[0]) | (y["station"] == stations[1]) | (y["station"] == stations[2]) | 
#                   (y["station"] == stations[3]) | (y["station"] == stations[4]) | (y["station"] == stations[5]) | (y["station"] == stations[6]) | (y["station"] == stations[7])].drop(columns=["station"])
y_train = y.loc[(y["station"] == stations[0]) | (y["station"] == stations[1]) | (y["station"] == stations[2]) | 
                  (y["station"] == stations[3]) | (y["station"] == stations[4])].drop(columns=["station"])
X_test = X.loc[(X["station"] == stations[8]) | (X["station"] == stations[9]) | (X["station"] == stations[10]) | 
                  (X["station"] == stations[11])].drop(columns=['PM2_5', 'station', "year"])
y_test = y.loc[(y["station"] == stations[8]) | (y["station"] == stations[9]) | (y["station"] == stations[10]) | 
                  (y["station"] == stations[11]) ].drop(columns=["station"])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# y_train = y_train.ravel()
# Step 2: Model Building
# Initialize the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Step 3: Training the model
rf_model.fit(X_train, y_train.values.ravel())

# Step 4: Prediction
y_pred = rf_model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mape = mean_absolute_percentage_error(y_test, y_pred)
corr = np.corrcoef(np.reshape(y_test, (-1)), np.reshape(y_pred, (-1)))[0][1]
import pdb; pdb.set_trace()
mdape_ = mdape(y_test, y_pred)


print("MSE:", mse)
print("MAE:", mae)
print("R2:", r2)
print("RMSE:", rmse)
print("MAPE:", mape)
print("MDAPE:", mdape_)
print("CORR:", corr)
