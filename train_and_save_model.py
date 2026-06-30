import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("household_electricity_data.csv")

# Drop ID column (NOT useful for ML)
df = df.drop(columns=["Household_ID"])

# Target
TARGET = "Monthly_Electricity_Consumption_kWh"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Identify columns
numeric_features = [
    "Number_of_Occupants",
    "Number_of_Appliances",
    "Daily_Usage_Hours_Avg",
    "Has_Fridge",
    "Has_TV",
    "Has_Electric_Iron",
    "Has_Fan",
    "Has_Air_Conditioner",
    "Has_Washing_Machine",
    "Has_Water_Heater",
    "Has_Microwave",
    "Has_Electric_Kettle",
    "Has_Phone_Charger",
    "Has_Radio",
    "Has_Multiple_Lighting_Bulbs"
]

categorical_features = ["Type_of_House"]

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
])

# Model
model = Pipeline([
    ("prep", preprocessor),
    ("lr", LinearRegression())
])

# Train
model.fit(X, y)

# Save model
joblib.dump(model, "linear_regression_model.pkl")

print("✅ Model trained and saved successfully!")