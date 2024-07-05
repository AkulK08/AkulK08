import pygrib
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import time

# Start timing
start_time = time.time()

# Open the GRIB file
grbs = pygrib.open('adaptor.mars.internal-1720093204.7159867-9472-15-8365f445-06f7-4899-ac86-86c68f5ffc5f.grib')

# Initialize lists to store data
temps = []
precips = []

# Extract data
for grb in grbs:
    if grb.shortName == '2t':
        temps.append(grb.values.flatten())
    elif grb.shortName == 'tp':
        precips.append(grb.values.flatten())

# Convert to numpy arrays for faster processing
temps = np.array(temps)
precips = np.array(precips)

# Create features (X) and target (y)
X = np.column_stack((temps.flatten(), precips.flatten()))
y = temps.flatten()  # Using temperature as target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
svm_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svm_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm_model.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Print execution time
print(f"Execution time: {time.time() - start_time:.2f} seconds")
