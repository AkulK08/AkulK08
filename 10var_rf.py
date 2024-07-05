import pygrib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time

# Start timing
start_time = time.time()

# Open the GRIB file
file_path = '10var.grib'
grbs = pygrib.open(file_path)

# Define variables to extract
variables = ['10u', '10v', '2t', 'mx2t', 'mn2t', 'sp', 'tp']

# Initialize a dictionary to store data
data = {var: [] for var in variables}

# Extract data
for grb in grbs:
    if grb.shortName in variables:
        data[grb.shortName].append(grb.values.flatten())

# Convert to numpy arrays
for var in variables:
    data[var] = np.array(data[var])

# Create a pandas DataFrame
df = pd.DataFrame({var: data[var].flatten() for var in variables})

# Select features (X) and target variable (y)
# Using '2t' (2m temperature) as the target variable
X = df.drop('2t', axis=1)
y = df['2t']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))

# Print execution time
print(f"\nExecution time: {time.time() - start_time:.2f} seconds")
