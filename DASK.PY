import dask.dataframe as dd
from dask_ml.impute import SimpleImputer as DaskImputer
from dask_ml.preprocessing import StandardScaler as DaskScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time

# Start timer
start_time = time.time()

# Load dataset
print("Loading dataset...")
data = dd.read_csv("pdc_dataset_with_target.csv")

# Identify features
categorical_features = data.select_dtypes(include=["object"]).columns.tolist()
numerical_features = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
if "target" in numerical_features:
    numerical_features.remove("target")

# Feature engineering
print("Performing feature engineering...")
data["feat_1_x_4"] = data["feature_1"] * data["feature_4"]
data["feat_1_x_7"] = data["feature_1"] * data["feature_7"]
data["feat_2_x_6"] = data["feature_2"] * data["feature_6"]
data["feat_1_div_2"] = data["feature_1"] / (data["feature_2"] + 1e-5)
data["feat_4_div_6"] = data["feature_4"] / (data["feature_6"] + 1e-5)
data["feat_1_squared"] = data["feature_1"] ** 2
data["feat_2_squared"] = data["feature_2"] ** 2

# Update numerical features
new_features = [col for col in data.columns if col.startswith("feat_")]
numerical_features.extend(new_features)

# Impute and scale numerical features
print("Preprocessing numeric features...")
numeric_imputer = DaskImputer(strategy="median")
data[numerical_features] = numeric_imputer.fit_transform(data[numerical_features])

scaler = DaskScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Impute categorical features
if categorical_features:
    print("Preprocessing categorical features...")
    cat_imputer = DaskImputer(strategy="most_frequent")
    data[categorical_features] = cat_imputer.fit_transform(data[categorical_features])

# Convert Dask to pandas and encode categoricals
print("Converting Dask DataFrame to pandas and encoding categoricals...")
data = data.compute()
data = pd.get_dummies(data, drop_first=True)

# Final split
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train Random Forest
print("Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=26,
    min_samples_split=13,
    min_samples_leaf=2,
    max_features="sqrt",
    bootstrap=False,
    random_state=42,
    n_jobs=-1,
)

train_start = time.time()
model.fit(X_train, y_train)
train_time = time.time() - train_start
print(f"Training completed in {train_time:.2f} seconds")

# Evaluate
print("Evaluating model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["0", "1"],
    yticklabels=["0", "1"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("rf_confusion_matrix_final.png")
print("Saved rf_confusion_matrix_final.png")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("rf_roc_curve_final.png")
print("Saved rf_roc_curve_final.png")

# Total time
total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time:.2f} seconds")
