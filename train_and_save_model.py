import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------------------------
# PATHS (must match app.py)
# ------------------------------------------------------------------
DATA_PATH = "data/household_electricity_data.csv"
MODEL_PATH = "linear_regression_model.pkl"
os.makedirs("outputs", exist_ok=True)

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Drop ID column (not useful for ML)
if "Household_ID" in df.columns:
    df = df.drop(columns=["Household_ID"])

# Align target column name with app.py (TARGET = "Monthly_Consumption_kWh")
if "Monthly_Electricity_Consumption_kWh" in df.columns:
    df = df.rename(columns={"Monthly_Electricity_Consumption_kWh": "Monthly_Consumption_kWh"})

TARGET = "Monthly_Consumption_kWh"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# ------------------------------------------------------------------
# FEATURE COLUMNS (must match the columns app.py builds in input_df)
# ------------------------------------------------------------------
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
    "Has_Multiple_Lighting_Bulbs",
]
categorical_features = ["Type_of_House"]

# ------------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# PREPROCESSING + MODEL
# ------------------------------------------------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
])

model = Pipeline([
    ("prep", preprocessor),
    ("lr", LinearRegression()),
])

# ------------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------------
model.fit(X_train, y_train)

# ------------------------------------------------------------------
# EVALUATE
# ------------------------------------------------------------------
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2:   {r2:.4f}")

# ------------------------------------------------------------------
# REFIT ON FULL DATA (best practice before shipping the final model)
# ------------------------------------------------------------------
model.fit(X, y)

# ------------------------------------------------------------------
# SAVE MODEL (matches app.py's MODEL_PATH)
# ------------------------------------------------------------------
joblib.dump(model, MODEL_PATH)
print(f"✅ Model trained and saved to {MODEL_PATH}")

# ------------------------------------------------------------------
# SAVE METRICS (for 🤖 Model Comparison / 📋 Model Performance pages)
# ------------------------------------------------------------------
metrics_df = pd.DataFrame([{
    "Model": "Linear Regression",
    "MAE": round(mae, 3),
    "RMSE": round(rmse, 3),
    "R2_Score": round(r2, 4),
}])
metrics_df.to_csv("outputs/model_metrics.csv", index=False)

metrics_json = {
    "best_model": "Linear Regression",
    "train_rows": int(X_train.shape[0]),
    "test_rows": int(X_test.shape[0]),
    "results": metrics_df.to_dict(orient="records"),
}
with open("outputs/metrics.json", "w") as f:
    json.dump(metrics_json, f, indent=2)

# ------------------------------------------------------------------
# SAVE FEATURE IMPORTANCE (absolute coefficient values)
# ------------------------------------------------------------------
feature_names = model.named_steps["prep"].get_feature_names_out()
coefs = model.named_steps["lr"].coef_

fi_df = pd.DataFrame({"Feature": feature_names, "Importance": np.abs(coefs)})
fi_df["Feature"] = fi_df["Feature"].str.replace("num__", "", regex=False)
fi_df["Feature"] = fi_df["Feature"].str.replace("cat__Type_of_House_", "House Type: ", regex=False)
fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
fi_df.to_csv("outputs/feature_importance.csv", index=False)

print("✅ Metrics, feature importance saved to outputs/")