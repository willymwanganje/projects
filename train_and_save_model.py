import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# ------------------------------------------------------------------
# PATHS (must match app.py)
# ------------------------------------------------------------------
DATA_PATH = "data/household_electricity_data.csv"
MODEL_PATH = "linear_regression_model.pkl"   # app.py loads this filename -> we save the "best" model here
os.makedirs("outputs", exist_ok=True)

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

if "Household_ID" in df.columns:
    df = df.drop(columns=["Household_ID"])

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
# PREPROCESSING
# ------------------------------------------------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
])

# ------------------------------------------------------------------
# MODELS TO COMPARE
# ------------------------------------------------------------------
candidate_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
}

results = []
fitted_pipelines = {}

for name, estimator in candidate_models.items():
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", estimator),
    ])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"[{name}] MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.4f}")

    results.append({
        "Model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2_Score": round(r2, 4),
    })
    fitted_pipelines[name] = pipe

metrics_df = pd.DataFrame(results).sort_values("R2_Score", ascending=False).reset_index(drop=True)
metrics_df.to_csv("outputs/model_metrics.csv", index=False)

# ------------------------------------------------------------------
# SELECT THE BEST MODEL (highest R2 Score)
# ------------------------------------------------------------------
best_model_name = metrics_df.iloc[0]["Model"]
best_pipeline_on_test = fitted_pipelines[best_model_name]
print(f"\n🏆 Best model: {best_model_name} (R2 = {metrics_df.iloc[0]['R2_Score']})")

# ------------------------------------------------------------------
# REFIT THE BEST MODEL ON THE FULL DATASET (best practice before saving)
# ------------------------------------------------------------------
best_estimator = candidate_models[best_model_name]
final_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", best_estimator),
])
final_pipeline.fit(X, y)

# ------------------------------------------------------------------
# SAVE BEST MODEL (filename stays the same so app.py keeps working)
# ------------------------------------------------------------------
joblib.dump(final_pipeline, MODEL_PATH)
print(f"✅ Best model ({best_model_name}) saved to: {MODEL_PATH}")

# ------------------------------------------------------------------
# SAVE METRICS JSON (for the Model Performance page)
# ------------------------------------------------------------------
metrics_json = {
    "best_model": best_model_name,
    "train_rows": int(X_train.shape[0]),
    "test_rows": int(X_test.shape[0]),
    "results": metrics_df.to_dict(orient="records"),
}
with open("outputs/metrics.json", "w") as f:
    json.dump(metrics_json, f, indent=2)

# ------------------------------------------------------------------
# SAVE FEATURE IMPORTANCE
# Linear Regression -> |coefficients|
# Decision Tree / Random Forest -> feature_importances_
# ------------------------------------------------------------------
feature_names = final_pipeline.named_steps["prep"].get_feature_names_out()
fitted_model = final_pipeline.named_steps["model"]

if hasattr(fitted_model, "coef_"):
    importances = np.abs(fitted_model.coef_)
elif hasattr(fitted_model, "feature_importances_"):
    importances = fitted_model.feature_importances_
else:
    importances = np.zeros(len(feature_names))

fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
fi_df["Feature"] = fi_df["Feature"].str.replace("num__", "", regex=False)
fi_df["Feature"] = fi_df["Feature"].str.replace("cat__Type_of_House_", "House Type: ", regex=False)
fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
fi_df.to_csv("outputs/feature_importance.csv", index=False)

print("✅ Metrics (all 3 models) and feature importance saved to outputs/")