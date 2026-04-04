import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("C:\\Users\\acer\\Desktop\\Chandigarh University\\Major Project-1\\research_dataset.csv")

df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(" ", "_")

print("Columns in Dataset:")
print(df.columns)

if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )
    df = df.sort_values("Timestamp")
    df = df.drop(columns=["Timestamp"])


target_column = [col for col in df.columns if "Energy" in col][0]
print("Detected Target Column:", target_column)

X_all = df.drop(columns=[target_column])
y_all = df[target_column]

X_all = pd.get_dummies(X_all, drop_first=True)
print("Feature Shape After Encoding:", X_all.shape)


scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X_all)
y_scaled = scaler_y.fit_transform(y_all.values.reshape(-1,1))

def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)

window_size = 120

X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size)

print("Sequence Shape:", X_seq.shape)

split = int(0.8 * len(X_seq))

X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# Classical models use only last timestep
X_train_flat = X_train[:, -1, :]
X_test_flat = X_test[:, -1, :]

def evaluate(y_true, y_pred):
    y_true_inv = scaler_y.inverse_transform(y_true.reshape(-1,1))
    y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1,1))

    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    mape = np.mean(np.abs((y_true_inv - y_pred_inv) / y_true_inv)) * 100
    r2 = r2_score(y_true_inv, y_pred_inv)

    return mae, rmse, mape, r2

results = {}

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_flat, y_train)
y_pred_lr = lr.predict(X_test_flat)
results["Linear Regression"] = evaluate(y_test, y_pred_lr)

# SVR
svr = SVR()
svr.fit(X_train_flat, y_train.ravel())
y_pred_svr = svr.predict(X_test_flat)
results["SVR"] = evaluate(y_test, y_pred_svr)

# Random Forest
rf = RandomForestRegressor(n_estimators=50)
rf.fit(X_train_flat, y_train.ravel())
y_pred_rf = rf.predict(X_test_flat)
results["Random Forest"] = evaluate(y_test, y_pred_rf)

# XGBoost
xgb = XGBRegressor(n_estimators=100, learning_rate=0.05)
xgb.fit(X_train_flat, y_train.ravel())
y_pred_xgb = xgb.predict(X_test_flat)
results["XGBoost"] = evaluate(y_test, y_pred_xgb)

# LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(window_size, X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)
y_pred_lstm = model.predict(X_test)
results["LSTM"] = evaluate(y_test, y_pred_lstm)

# Final Model Comparison Results

print("\n📊 Final Model Comparison Results:\n")

print(f"{'':<20}{'MAE':<12}{'RMSE':<12}{'MAPE (%)':<12}{'R2 Score':<12}")

for model, metrics in results.items():
    mae, rmse, mape, r2 = [round(x, 6) for x in metrics]
    print(f"{model:<20}{mae:<12}{rmse:<12}{mape:<12}{r2:<12}")
    

# Actual vs LSTM Prediction


y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1,1))
y_pred_lstm_inv = scaler_y.inverse_transform(y_pred_lstm.reshape(-1,1))

plt.figure(figsize=(12,6))
plt.plot(y_test_inv[:1000], label="Actual", linewidth=1.2)
plt.plot(y_pred_lstm_inv[:1000], label="LSTM Prediction", linewidth=1.2)
plt.title("Actual vs LSTM Prediction")
plt.xlabel("Time Steps")
plt.ylabel("Energy Consumption (kWh)")
plt.legend()
plt.tight_layout()
plt.show()

# Model Performance Comparison (With Values)

comparison_df = pd.DataFrame(results).T
comparison_df.columns = ["MAE", "RMSE", "MAPE (%)", "R2 Score"]

metrics = ["MAE", "RMSE", "MAPE (%)", "R2 Score"]
models = comparison_df.index

fig, ax = plt.subplots(figsize=(12,6))

x = np.arange(len(models))
width = 0.2

bars1 = ax.bar(x - 1.5*width, comparison_df["MAE"], width, label="MAE")
bars2 = ax.bar(x - 0.5*width, comparison_df["RMSE"], width, label="RMSE")
bars3 = ax.bar(x + 0.5*width, comparison_df["MAPE (%)"], width, label="MAPE (%)")
bars4 = ax.bar(x + 1.5*width, comparison_df["R2 Score"], width, label="R2 Score")

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45)
ax.set_title("Model Performance Comparison")
ax.set_ylabel("Metric Value")
ax.legend()

# Add values above bars
def add_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)

add_values(bars1)
add_values(bars2)
add_values(bars3)
add_values(bars4)

plt.tight_layout()
plt.show()