import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import time
from scipy.stats import randint, uniform

# Start timer
start_time = time.time()

# Load the dataset
print("Loading dataset...")
data = pd.read_csv("pdc_dataset_with_target.csv")

# Display data summary
print(f"\nDataset shape: {data.shape}")
print(f"Target distribution:\n{data['target'].value_counts(normalize=True)}")

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing values in each column:")
print(missing_values[missing_values > 0])

# Identify categorical and numerical features
categorical_features = data.select_dtypes(include=["object"]).columns.tolist()
numerical_features = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
numerical_features.remove("target")  # Remove target from feature list

print(f"\nCategorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")

# Feature engineering
print("\nPerforming feature engineering...")

# Add interaction features between numerical columns
data["feat_1_x_4"] = data["feature_1"] * data["feature_4"]
data["feat_1_x_7"] = data["feature_1"] * data["feature_7"]
data["feat_2_x_6"] = data["feature_2"] * data["feature_6"]

# Add ratio features
data["feat_1_div_2"] = data["feature_1"] / (
    data["feature_2"] + 1e-5
)  # Avoid division by zero
data["feat_4_div_6"] = data["feature_4"] / (data["feature_6"] + 1e-5)

# Create square features
data["feat_1_squared"] = data["feature_1"] ** 2
data["feat_2_squared"] = data["feature_2"] ** 2

# Update numerical features list with new features
new_numerical_features = [col for col in data.columns if col.startswith("feat_")]
numerical_features.extend(new_numerical_features)
print(f"Added {len(new_numerical_features)} engineered features")

# Split the data into features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Create preprocessing pipeline
# For numeric features: impute missing values, then scale
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# For categorical features: impute missing values, then one-hot encode
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

# Combine the transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Define Random Forest parameters for full tuning
param_grid = {
    "classifier__n_estimators": randint(100, 500),
    "classifier__max_depth": randint(5, 30),
    "classifier__min_samples_split": randint(2, 20),
    "classifier__min_samples_leaf": randint(1, 10),
    "classifier__max_features": ["sqrt", "log2", None],
    "classifier__bootstrap": [True, False],
    "classifier__class_weight": ["balanced", "balanced_subsample", None],
}

# Create full pipeline with optimized parameters from hyperparameter tuning
# Best parameters:
# - bootstrap: False
# - class_weight: None
# - max_depth: 26
# - max_features: sqrt
# - min_samples_leaf: 2
# - min_samples_split: 13
# - n_estimators: 393
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=393,
                max_depth=26,
                min_samples_split=13,
                min_samples_leaf=2,
                max_features="sqrt",
                bootstrap=False,
                class_weight=None,
                random_state=42,
                n_jobs=1,
            ),
        ),
    ]
)

# Comment out RandomizedSearchCV section since we're using optimized parameters directly
"""
# Use RandomizedSearchCV for thorough hyperparameter tuning
print("\nBeginning Random Forest hyperparameter tuning with 10 iterations...")
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=10,  # Number of parameter settings sampled
    cv=5,       # 5-fold cross-validation
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,  # Use all CPU cores
    verbose=2   # Show progress
)

# Fit the model
search_start = time.time()
search.fit(X_train, y_train)
search_time = time.time() - search_start

print(f"\nRandom Forest tuning completed in {search_time:.2f} seconds")
print(f"Best parameters: {search.best_params_}")
print(f"Best cross-validation score: {search.best_score_:.4f}")
"""

# Train the model with optimized parameters
print("\nTraining Random Forest with optimized parameters...")
train_start = time.time()
pipeline.fit(X_train, y_train)
train_time = time.time() - train_start
print(f"Training completed in {train_time:.2f} seconds")

# Evaluate on test set
best_model = pipeline  # No need for search.best_estimator_ since we're using optimized params directly
test_start = time.time()
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
test_time = time.time() - test_start

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nTest set performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  ROC AUC: {roc_auc:.4f}")
print(f"  Prediction time: {test_time:.2f} seconds")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
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
plt.savefig("rf_confusion_matrix.png")
print("Saved rf_confusion_matrix.png")

# Plot ROC curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("rf_roc_curve.png")
print("Saved rf_roc_curve.png")

# Feature importance analysis
feature_importances = best_model.named_steps["classifier"].feature_importances_

# Get preprocessed feature names (simplified approach)
# This won't be perfectly accurate but gives rough idea of importance
feature_names = numerical_features.copy()
if categorical_features:
    # Add placeholder names for categorical features
    for cat_feature in categorical_features:
        # Assume each categorical feature expands to multiple columns
        feature_names.extend(
            [f"{cat_feature}_{i}" for i in range(3)]
        )  # Assuming 3 categories per feature

# Truncate or pad feature_names to match feature_importances length
if len(feature_names) > len(feature_importances):
    feature_names = feature_names[: len(feature_importances)]
elif len(feature_names) < len(feature_importances):
    feature_names.extend(
        [f"Unknown_{i}" for i in range(len(feature_importances) - len(feature_names))]
    )

# Get top 15 features
indices = np.argsort(feature_importances)[-15:]
plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), feature_importances[indices], align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.title("Random Forest Feature Importance (Top 15)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("rf_feature_importance.png")
print("Saved rf_feature_importance.png")

# Print total execution time
total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time:.2f} seconds")
