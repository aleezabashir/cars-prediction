import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os

# Create model folder
os.makedirs("model", exist_ok=True)

# --- Create dataset ---
data = {
    "ID": [1,2,3,4,5,6,7,8,9,10],
    "Year": [2015,2018,2012,2020,2016,2017,2014,2019,2013,2021],
    "Mileage": [50000,30000,70000,10000,40000,35000,60000,20000,65000,5000],
    "EngineSize": [1.6,2.0,1.4,2.0,1.6,1.8,1.2,2.0,1.6,2.2],
    "Horsepower": [120,150,100,160,130,140,90,155,125,170],
    "FuelType": ["Petrol","Diesel","Petrol","Diesel","Petrol","Diesel","Petrol","Diesel","Petrol","Diesel"],
    "Price": [12000,18000,9000,22000,14000,16000,8000,20000,11000,25000]
}

df = pd.DataFrame(data)

# One-hot encode FuelType
df = pd.get_dummies(df, columns=['FuelType'], drop_first=True)

# Features & target
X = df.drop(['ID','Price'], axis=1)
y = df['Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/car_price_model.pkl")
print("✅ Model trained and saved as model/car_price_model.pkl")
