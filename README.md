# Geology-Forecast-Challenge-Private-4th-prize

The experiments were conducted using Google Colab with a 4T GPU, allowing the full process to complete within approximately 3 to 4 minutes. On Kaggle, where only CPU resources were available, the same code required about 30 minutes to run.

The overall prediction pipeline was structured into a three-stage iterative loop: data preprocessing, model selection, and performance refinement. Among these, I placed particular emphasis on the data preprocessing stage, considering the specific characteristics and constraints of the dataset. My primary objective was to handle missing values in a way that best preserved the underlying structure of the data, especially its linear trends and occasional abrupt transitions.

Note: Since I did not fix the random seed during training, the results may vary slightly with each run. Additionally, the same best-performing prediction was duplicated across all 10 realizations, which does not fully satisfy the task requirement for uncertainty quantification. I sincerely apologize for this oversight. Nevertheless, as seen in my submission history, not every run yielded a high score, but the overall performance showed a gradual improvement over time. When the performance did improve, it often resulted in scores that surpassed those of the top-ranking submissions. 
Since the scores fluctuate due to the seed not being fixed, it might be a good idea to try using seed=0 or seed=37 and 200 epochs to get more consistent results.

I kindly ask that you consider the integrity and originality of the methodology presented in this code when making any further assessments.

# Data Processing

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
    
All necessary libraries were imported, and the train.csv and test.csv files provided by the competition organizers were loaded.
In the train dataset, all realization_1 through realization_9 columns contain identical values. Therefore, only columns 1 through 300 were used for training.

Since there were missing values in the input columns ranging from -299 to 0—which are essential for model training—I implemented an interpolation process to handle them.
For each row, values exist only from a certain point -k up to 0, with all preceding values missing. Therefore, a function was defined to dynamically locate this -k for each row.

What follows reflects the core of my original idea, derived through multiple experiments and observations.
The data exhibits a combination of linear trends and occasional abrupt changes. This characteristic made it difficult to address uncertainty using a single imputation method.
To tackle this, I employed multiple interpolation models (e.g., linear, PCHIP, KNN, spline) and applied them to each row. I then selected the most natural-looking result based on smoothness evaluation (using the variance of the second derivative).
In summary, the preprocessing was designed to harmonize multiple methods, selecting the most appropriate one for each individual row, thus ensuring a more adaptive and reliable handling of missing data.

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

The criterion for selecting the most suitable model was based on how naturally the imputed data aligned with the overall trend of the sequence.

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

After completing the missing value imputation, I reloaded the resulting file to verify whether the data preprocessing reflected my intended approach.

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

# Model selection 
Since the data appeared to exhibit characteristics of a time series, I employed an RNN-based LSTM model. The specific parameters are provided in the code below.

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

Afterward, the results are saved as a CSV file.

# Addition : More elaborate code
This section was created to provide additional insights and is not the exact code that achieved 4th place. However, the original 4th-place solution emerged as part of the iterative process aimed at improving performance. The following explanation presents the version of the code that I personally find most satisfying, based on continued experimentation.

Data Preprocessing Strategy:
Among the models considered during preprocessing, I found that using only three methods—Linear, Spline, and PCHIP—produced the most stable results. Furthermore, we advanced our approach from selecting the most natural model for each row based on evaluation metrics, to selecting the most natural value at the individual cell level — allowing each missing value to be filled by the model that performs best for that specific position.

Initially, I used the variance of curvature (curvature_var) as the sole evaluation metric to assess the smoothness of the data. However, I concluded that relying solely on this metric might overlook local fluctuations or sharp transitions in the data. To better capture such localized variations, I introduced additional evaluation metrics: max_curvature and num_sharp_turns. I determined that assigning weights of 1.0 to curvature_var, 0.5 to max_curvature, and 0.3 to num_sharp_turns led to the most desirable preprocessing outcomes. This is because the weighted combination maintains the overall linear trend of the data while still incorporating significant local variations when necessary.

Model Selection:
Through experimentation, I found that using GRU instead of LSTM, along with tuning the number of hidden units, yielded better results. Additionally, employing a checkpointing strategy that saves the model achieving the lowest validation loss during training proved to be an effective way to identify the best-performing model.

I’ve uploaded the file titled 'More elaborate Code'. Please take a look when you get a chance.
