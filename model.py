import pandas as pd
import os
import joblib
from sklearn.linear_model import LinearRegression

# CSV file path
file_path = os.path.join(os.getcwd(), "nifty_50.csv")

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found. Please ensure the CSV is in the same folder.")
else:
    data = pd.read_csv(file_path)
    X = data[['Open']]
    y = data['Open'].shift(-1)

    X = X[:-1]
    y = y[:-1]

    model = LinearRegression()
    model.fit(X, y)

    model_path = os.path.join(os.getcwd(), "model.pkl")
    joblib.dump(model, model_path)

    print("Model trained and saved as model.pkl")