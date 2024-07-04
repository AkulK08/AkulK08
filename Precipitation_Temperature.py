import pygrib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Read the GRIB file
grbs = pygrib.open('adaptor.mars.internal-1720093204.7159867-9472-15-8365f445-06f7-4899-ac86-86c68f5ffc5f.grib')

# Print available messages and their dimensions
print("Available messages in the GRIB file:")
for grb in grbs:
    print(f"- {grb.shortName}: {grb.name}, Dimensions: {grb.values.shape}")

# Reset the file pointer
grbs.seek(0)

# Initialize dictionaries to store data
data = {}

# Extract data from the GRIB file
for grb in grbs:
    shortName = grb.shortName
    parameterName = grb.parameterName
    if shortName not in data:
        data[shortName] = {
            'times': [],
            'values': [],
            'name': parameterName
        }
    data[shortName]['times'].append(grb.validDate)
    data[shortName]['values'].append(grb.values.flatten())

# Check if we have the required variables (update these based on what's in your file)
required_vars = ['2t', 'tp']  # '2t' is often used for 2m temperature instead of 't2m'
missing_vars = [var for var in required_vars if var not in data]

if missing_vars:
    print(f"Warning: The following required variables are missing: {missing_vars}")
    print("Available variables:", list(data.keys()))
    print("Please update the required_vars list with available variables.")
else:
    # Create DataFrame
    df_list = []
    for var in required_vars:
        df = pd.DataFrame({
            'time': np.repeat(data[var]['times'], len(data[var]['values'][0])),
            var: np.concatenate(data[var]['values'])
        })
        df_list.append(df)

    # Merge dataframes
    df = pd.merge(df_list[0], df_list[1], on='time')

    # Calculate average 2m temperature
    df['avg_2m_temperature'] = df.groupby('time')['2t'].transform('mean')

    # Calculate max and min total precipitation
    df['max_total_precipitation'] = df.groupby('time')['tp'].transform('max')
    df['min_total_precipitation'] = df.groupby('time')['tp'].transform('min')

    # Prepare features (X) and target variable (y)
    X = df[['avg_2m_temperature', 'max_total_precipitation', 'min_total_precipitation']]
    y = df['2t']  # Using 2t as target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define parameter grid for SVM
    svm_param_grid = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.5],
        'kernel': ['rbf', 'linear']
    }

    # Perform GridSearch for SVM
    svm_grid_search = GridSearchCV(SVR(), svm_param_grid, cv=5, n_jobs=-1, verbose=2)
    svm_grid_search.fit(X_train_scaled, y_train)

    print("Best SVM parameters:", svm_grid_search.best_params_)
    svm_best_model = svm_grid_search.best_estimator_

    # Evaluate model
    y_pred = svm_best_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nSVM Results:")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")

    # Feature importance (coefficients for linear kernel, not available for non-linear kernels)
    if svm_best_model.kernel == 'linear':
        feature_importance = pd.DataFrame({'feature': X.columns, 'importance': np.abs(svm_best_model.coef_[0])})
        print(feature_importance.sort_values('importance', ascending=False))
    else:
        print("Feature importance is not available for non-linear kernels in SVM.")

    # Example: Make a prediction for new data
    new_data = np.array([[20, 50, 10]])  # [avg_2m_temp, max_total_precip, min_total_precip]
    new_data_scaled = scaler.transform(new_data)
    svm_prediction = svm_best_model.predict(new_data_scaled)
    print(f"\nPrediction for new data:")
    print(f"SVM: {svm_prediction[0]}")
