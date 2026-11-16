
import numpy as np
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score
)

# Import custom transformer so it is available during loading
from model_components import BrakeFeatureEngineer


# --------------------------------
# 1. Configuration
# --------------------------------

RANDOM_STATE = 42

TARGET = "service_alert_90d"

MODEL_PATH = "models/brake_alert_model.joblib"


# --------------------------------
# 2. Load Saved Model
# --------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}. Run train.py first."
    )

best_model = joblib.load(MODEL_PATH)

print("Trained model loaded successfully!")


# --------------------------------
# 3. Load Dataset
# --------------------------------

df = pd.read_csv("breakpad.csv")

df = df.drop_duplicates().reset_index(drop=True)


# --------------------------------
# 4. Separate Features and Target
# --------------------------------

X = df.drop(
    columns=[
        "customer_id",
        "vehicle_id",
        "estimated_km_to_replacement",
        TARGET
    ]
)

y = df[TARGET]


# --------------------------------
# 5. Recreate Same Train-Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)


# --------------------------------
# 6. Make Predictions
# --------------------------------

y_test_pred = best_model.predict(X_test)

y_test_probs = best_model.predict_proba(X_test)[:, 1]


# --------------------------------
# 7. Evaluate Model
# --------------------------------

print("\n========== TEST RESULTS ==========")

print(
    f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy_score(y_test, y_test_pred):.4f}"
)

print(
    f"F1 Score: {f1_score(y_test, y_test_pred):.4f}"
)

print(
    f"ROC-AUC: {roc_auc_score(y_test, y_test_probs):.4f}"
)


# --------------------------------
# 8. Classification Report
# --------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=["No Alert (0)", "Service Alert (1)"]
    )
)


# --------------------------------
# 9. Confusion Matrix
# --------------------------------

print("\nConfusion Matrix:")

print(
    confusion_matrix(y_test, y_test_pred)
)


print("\nModel testing completed successfully!")
