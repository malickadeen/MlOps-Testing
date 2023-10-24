import pandas as pd
import joblib
import random

# Load the trained model from a file
model = joblib.load('model.pkl')  # Replace 'model.pkl' with the path to your saved model

# Generate random input data for prediction
random_sepal_length = random.uniform(4.0, 7.0)  # Random sepal length between 4.0 and 7.0
random_sepal_width = random.uniform(2.0, 4.5)   # Random sepal width between 2.0 and 4.5
random_petal_length = random.uniform(1.0, 6.0)  # Random petal length between 1.0 and 6.0
random_petal_width = random.uniform(0.1, 2.5)   # Random petal width between 0.1 and 2.5

# Create a DataFrame with the random input data
new_data_point = pd.DataFrame({
    'sepal length (cm)': [random_sepal_length],
    'sepal width (cm)': [random_sepal_width],
    'petal length (cm)': [random_petal_length],
    'petal width (cm)': [random_petal_width]
})

# Make predictions
predictions = model.predict(new_data_point)

# Print the predictions
print(f'Random Input Data: {new_data_point.to_dict(orient="records")}')
print(f'Predicted class: {predictions[0]}')
