
# ============================================================
# PORTGUARD AI
# RANDOM FOREST CLASSIFICATION MODEL
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ------------------------------------------------------------
# 1. DEFINE PROJECT DIRECTORIES
# ------------------------------------------------------------

project_directory = Path(__file__).resolve().parent

data_path = project_directory / "data" / "port_congestion_data.csv"

models_directory = project_directory / "models"
models_directory.mkdir(exist_ok=True)

reports_directory = project_directory / "reports"
reports_directory.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print("\n==========================================")
print(" PORTGUARD AI - MODEL TRAINING")
print("==========================================")

df = pd.read_csv(data_path)

print("\nDataset loaded successfully!")

print(f"Total records: {len(df)}")


# ------------------------------------------------------------
# 3. DEFINE INPUT FEATURES AND TARGET
# ------------------------------------------------------------

feature_columns = [
    "port_name",
    "vessel_arrivals",
    "port_capacity",
    "queue_length",
    "weather_severity",
    "historical_avg_delay",
    "hour_of_day",
    "month"
]

target_column = "congestion"

X = df[feature_columns]

y = df[target_column]


# Check whether both classes exist

if y.nunique() != 2:
    raise ValueError(
        "Dataset must contain both congestion classes (0 and 1)."
    )

print("\nInput features:")
print(feature_columns)

print("\nTarget variable:")
print(target_column)

print("\nTarget distribution:")
print(y.value_counts())


# ------------------------------------------------------------
# 4. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ------------------------------------------------------------

categorical_features = ["port_name"]

numerical_features = [
    "vessel_arrivals",
    "port_capacity",
    "queue_length",
    "weather_severity",
    "historical_avg_delay",
    "hour_of_day",
    "month"
]


# ------------------------------------------------------------
# 5. PREPROCESSING
# ------------------------------------------------------------

# Convert port names into numerical features.
# Numeric features are passed through without scaling.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ------------------------------------------------------------
# 6. SPLIT DATA INTO TRAINING AND TESTING SETS
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ------------------------------------------------------------
# 7. CREATE RANDOM FOREST CLASSIFIER
# ------------------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ------------------------------------------------------------
# 8. CREATE COMPLETE MACHINE LEARNING PIPELINE
# ------------------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", random_forest)
    ]
)


# ------------------------------------------------------------
# 9. TRAIN THE MODEL
# ------------------------------------------------------------

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed successfully!")


# ------------------------------------------------------------
# 10. MAKE PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# 11. EVALUATE THE MODEL
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n==========================================")
print(" MODEL EVALUATION RESULTS")
print("==========================================")

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


# Detailed classification report

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Congestion",
            "Congestion"
        ],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 12. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Congestion",
        "Congestion"
    ]
)

display.plot(cmap="Blues")

plt.title("PortGuard AI - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    reports_directory / "confusion_matrix.png"
)

plt.close()


# ------------------------------------------------------------
# 13. FEATURE IMPORTANCE
# ------------------------------------------------------------

trained_preprocessor = model.named_steps["preprocessor"]

trained_classifier = model.named_steps["classifier"]

feature_names = (
    trained_preprocessor.get_feature_names_out()
)

importance_values = trained_classifier.feature_importances_

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_values
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\nFeature Importance:")

print(feature_importance.to_string(index=False))


# Save feature importance chart

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["feature"],
    feature_importance["importance"]
)

plt.xlabel("Importance Score")

plt.ylabel("Features")

plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    reports_directory / "feature_importance.png"
)

plt.close()


# ------------------------------------------------------------
# 14. SAVE THE TRAINED MODEL
# ------------------------------------------------------------

model_path = models_directory / "congestion_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")

print(f"Model location: {model_path}")


# ------------------------------------------------------------
# 15. SAVE EVALUATION RESULTS
# ------------------------------------------------------------

metrics_path = reports_directory / "model_evaluation.txt"

with open(metrics_path, "w", encoding="utf-8") as file:

    file.write("PORTGUARD AI - MODEL EVALUATION\n")

    file.write("================================\n\n")

    file.write(f"Training records: {len(X_train)}\n")

    file.write(f"Testing records: {len(X_test)}\n\n")

    file.write(f"Accuracy: {accuracy:.4f}\n")

    file.write(f"Precision: {precision:.4f}\n")

    file.write(f"Recall: {recall:.4f}\n")

    file.write(f"F1 Score: {f1:.4f}\n")

    file.write(f"ROC-AUC: {roc_auc:.4f}\n\n")

    file.write("Classification Report:\n")

    file.write(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Congestion",
                "Congestion"
            ],
            zero_division=0
        )
    )

    file.write("\nConfusion Matrix:\n")

    file.write(str(cm))

    file.write("\n\nFeature Importance:\n")

    file.write(feature_importance.to_string(index=False))


# ------------------------------------------------------------
# 16. FINAL SUMMARY
# ------------------------------------------------------------

print("\n==========================================")
print(" PORTGUARD AI - TRAINING COMPLETED")
print("==========================================")

print("\nGenerated files:")

print("1. models/congestion_model.pkl")

print("2. reports/confusion_matrix.png")

print("3. reports/feature_importance.png")

print("4. reports/model_evaluation.txt")

print("\n==========================================")
