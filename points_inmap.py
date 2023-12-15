
import numpy as np
import os
import pandas as pd  # Add import for pandas

# TensorFlow
import tensorflow as tf

def generate_random_coordinates(num_points=657500, radius=1, center_lat=0, center_lon=0):
    pi = np.pi
    theta = np.random.uniform(0, 2 * pi, size=num_points)

    # Generate positive values for the radius
    r_positive = np.abs(radius * np.sqrt(np.random.normal(0, 1, size=num_points)**2))

    # Calculate x and y coordinates
    x_coords = np.cos(theta) * r_positive + center_lon
    y_coords = np.sin(theta) * r_positive + center_lat

    # Adjust the precision of the coordinates
    x_coords = np.round(x_coords, 6)
    y_coords = np.round(y_coords, 6)

    # Create a DataFrame with the coordinates
    df = pd.DataFrame({'latitude': y_coords, 'longitude': x_coords})
    return df


vegas_data = generate_random_coordinates(num_points=100, radius=2, center_lat=-36.1699, center_lon=-115.1398)

london_data = generate_random_coordinates(num_points=100, radius=0.5, center_lat=51.509865, center_lon=-0.118092)


X = np.concatenate([vegas_data, london_data])
X = np.round(X, 6)
y = np.concatenate([np.zeros(800), np.ones(100), np.ones(100)])  # Assign labels (0 for circular data, 1 for London and Las Vegas)

train_end = int(0.6 * len(X))
test_start = int(0.8 * len(X))
X_train, y_train = X[:train_end], y[:train_end]
X_test, y_test = X[test_start:], y[test_start:]
X_val, y_val = X[train_end:test_start], y[train_end:test_start]

tf.keras.backend.clear_session()
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=2, input_shape=[2], activation='relu', name='capa_1'),
    tf.keras.layers.Dense(units=4, activation='relu', name='capa_2'),
    tf.keras.layers.Dense(units=8, activation='relu', name='capa_3'),
    tf.keras.layers.Dense(units=1, activation='sigmoid', name='Output_Layer')
])

model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
print(model.summary())

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200)


# Guarda el modelo entrenado en el directorio 'linear-model/1/'
export_path = 'linear-model/1/'  # Cambia el número del modelo si es necesario
tf.saved_model.save(model, os.path.join('./', export_path))
# Predict labels for all data points
all_predictions = model.predict(X).tolist()

# GPS points for London and Las Vegas
gps_points_london = [[51.5074, -0.1278],  [51.4602, -0.1140], [51.5320, -0.1054],  [51.4715, -0.0696], [51.5532, -0.0921]]
gps_points_vegas = [[36.1699, -115.1398],  [36.0800, -115.1522],  [36.1146, -115.1728],  [36.1420, -115.2432], [-47.7, -15.8]]

# Extract predictions for London and Las Vegas
predictions_london = model.predict(gps_points_london).tolist()
predictions_vegas = model.predict(gps_points_vegas).tolist()

print("\nPredictions for London:")
print(predictions_london)

print("\nPredictions for Las Vegas:")
print(predictions_vegas)
