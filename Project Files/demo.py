from imblearn.over_sampling import SMOTE
import sys
print("Running with Python executable:", sys.executable)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
file_path = "HealthCareData.xlsx"
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

# Drop S.NO if present
if 'S.NO' in df.columns:
    df.drop(columns=['S.NO'], inplace=True)

# Convert specific columns to numeric
for col in ['TG', 'LDL', 'Total Bilirubin    (mg/dl)', 'A/G Ratio']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Detect and map target column
target_col = [col for col in df.columns if 'predict' in col.lower() or 'outcome' in col.lower() or 'cirrosis' in col.lower()]
if not target_col:
    raise ValueError("❌ Target column not found.")
target = target_col[0]
df[target] = df[target].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
if df[target].isnull().any():
    raise ValueError("❌ Target column has unknown values that couldn't be mapped to 0/1.")

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Split features and target
X = df.drop(columns=[target])
y = df[target].astype(int)

# Step 1: Train/Test Split BEFORE SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 2: Apply SMOTE on training set only
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 3: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Train models
def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Logistic Regression CV": LogisticRegressionCV(cv=5, max_iter=1000),
        "Ridge Classifier": RidgeClassifier(),
        "KNN Classifier": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "model": model,
            "train_accuracy": model.score(X_train, y_train),
            "test_accuracy": model.score(X_test, y_test),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        }

        print(f"\n🔍 {name}")
        print("-" * 40)
        print("Train Accuracy:", results[name]["train_accuracy"])
        print("Test Accuracy:", results[name]["test_accuracy"])
        print("F1 Score:", results[name]["f1_score"])
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return results

model_results = evaluate_models(X_train_scaled, X_test_scaled, y_train_resampled, y_test)

# Feature Importance using Random Forest
rf_model = model_results["Random Forest"]["model"]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Feature Importances - Random Forest")
plt.tight_layout()
plt.show()

# Save best model
best_model_name = max(model_results, key=lambda x: model_results[x]['test_accuracy'])
best_model = model_results[best_model_name]["model"]

with open("best_model.pkl", "wb") as f:
    pickle.dump((best_model, scaler, label_encoders, X.columns.tolist()), f)

print(f"\n✅ Saved best model: {best_model_name} to 'best_model.pkl'")
print("🔢 Target class distribution:\n", df[target].value_counts())