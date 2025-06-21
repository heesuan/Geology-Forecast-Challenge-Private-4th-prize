Python 3.10.5 (tags/v3.10.5:f377153, Jun  6 2022, 16:14:13) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ------------------- Data Loading -------------------

train = pd.read_csv('/kaggle/input/geology-forecast-challenge-open/data/train.csv')
test = pd.read_csv('/kaggle/input/geology-forecast-challenge-open/data/test.csv')

# Save geology_id for later use in submission
test_ids = test["geology_id"]

# Split train into input (X: -299 to 0) and target (y: 1 to 300)
X_train_full = train.iloc[:, 1:301]
y_train_full = train.iloc[:, 301:601]

# Extract input columns from test data
X_test_full = test.iloc[:, 1:301]

# ------------------- Missing Value Utilities -------------------

# Find first observed column index (i.e., first non-NaN)
def find_k(series):
    notna_index = series.notna()
    if notna_index.any():
        return int(series.index[notna_index.argmax()])
    return None  # Return None if all values are NaN

# Linear interpolation
def Linear_prediction(x_known, y_known, x_missing):
    model = LinearRegression()
    model.fit(x_known.reshape(-1, 1), y_known)
    return model.predict(x_missing.reshape(-1, 1))

# PCHIP interpolation
def PCHIP_prediction(x_known, y_known, x_missing):
    interpolator = PchipInterpolator(x_known, y_known, extrapolate=True)
    return interpolator(x_missing)

# KNN interpolation
def KNN_prediction(x_known, y_known, x_missing, k=3):
    model = KNeighborsRegressor(n_neighbors=min(k, len(x_known)))
    model.fit(x_known.reshape(-1, 1), y_known)
    return model.predict(x_missing.reshape(-1, 1))

# Univariate spline interpolation
def Spline_prediction(x_known, y_known, x_missing, s=0):
    spline = UnivariateSpline(x_known, y_known, s=s, ext=0)
    return spline(x_missing)

# Evaluate smoothness using variance of second derivative
def evaluate_smoothness(y_full):
    dy = np.gradient(y_full)
    ddy = np.gradient(dy)
    return np.var(ddy)

# Unified imputation function using multiple methods
def predict_missing_values(data):
    filled_data = data.copy()

    for i in range(data.shape[0]):
        row_data = data.iloc[i, 1:301]
        k_value = find_k(row_data)
        if k_value is None:
            continue

        k_index = row_data.index.get_loc(str(k_value))
        x_known = row_data.iloc[k_index+1:].dropna()
        y_target = row_data.iloc[:k_index]
        if x_known.empty or y_target.empty:
            continue

        x_known_idx = x_known.index.astype(float).to_numpy()
        y_known = x_known.to_numpy()
        x_missing_idx = y_target.index.astype(float).to_numpy()

        candidates = {}
        try:
            candidates["linear"] = Linear_prediction(x_known_idx, y_known, x_missing_idx)
        except:
            pass
        try:
            candidates["pchip"] = PCHIP_prediction(x_known_idx, y_known, x_missing_idx)
        except:
            pass
        try:
            candidates["knn"] = KNN_prediction(x_known_idx, y_known, x_missing_idx)
        except:
            pass
        try:
            candidates["spline"] = Spline_prediction(x_known_idx, y_known, x_missing_idx)
        except:
            pass

        best_method = None
        best_score = float("inf")
        for method, y_pred in candidates.items():
            combined = np.concatenate([y_pred, y_known])
            score = evaluate_smoothness(combined)
            if score < best_score:
                best_score = score
                best_method = method

        if best_method:
            filled_data.iloc[i, 1:k_index+1] = candidates[best_method]
            print(best_method)

    return filled_data

# ------------------- Apply Imputation -------------------

filled_train = predict_missing_values(train)
filled_train.to_csv("filled_train.csv", index=False)
print("Missing values in train filled and saved to 'filled_train.csv'.")

filled_test = predict_missing_values(test)
filled_test.to_csv("filled_test.csv", index=False)
print("Missing values in test filled and saved to 'filled_test.csv'.")

# Remove geology_id and prepare clean input data
X_train_clean = filled_train.iloc[:, 1:301]
X_test_clean = filled_test.iloc[:, 1:301]

# ------------------- LSTM Model Training -------------------

# Normalize input features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test_clean)

X_train_seq = X_train_scaled.reshape(-1, 300, 1)
X_test_seq = X_test_scaled.reshape(-1, 300, 1)

# Define LSTM model
model = Sequential([
    LSTM(128, return_sequences=False, input_shape=(300, 1)),
    Dense(256, activation='relu'),
    Dense(300)
])
model.compile(optimizer='adam', loss='mse')

# Train model (no validation split)
model.fit(X_train_seq, y_train_full.values, epochs=100, batch_size=16, verbose=1)

# Predict on test set
y_test_pred = model.predict(X_test_seq)

# ------------------- Generate Submission File -------------------

columns_new = [str(i+1) for i in range(300)]
for r in range(1, 10):
    for pos in range(1, 301):
        columns_new.append(f"r_{r}_pos_{pos}")

if y_test_pred.shape[1] < len(columns_new):
    y_test_pred = np.pad(y_test_pred, ((0, 0), (0, len(columns_new) - y_test_pred.shape[1])), constant_values=np.nan)

submission = pd.DataFrame(y_test_pred, columns=columns_new)
submission.insert(0, "geology_id", test_ids)

# Repeat the base prediction across all r_{r}_pos_{pos} fields
copied_part = submission.iloc[:, 1:301].copy().values
for start in range(301, 3001, 300):
    submission.iloc[:, start:start+300] = copied_part

submission.to_csv("voting_submission.csv", index=False)
print("✅ voting_submission.csv created")