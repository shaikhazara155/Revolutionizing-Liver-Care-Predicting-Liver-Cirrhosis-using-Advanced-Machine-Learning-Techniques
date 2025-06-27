# performanceTesting.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# 1. Load Excel data
df = pd.read_excel('HealthCareData.xlsx', sheet_name='Sheet1')

# 2. Clean column names
df.columns = df.columns.str.strip().str.replace(r'\s+', '', regex=True).str.replace(r'[^A-Za-z0-9]+', '', regex=True)

# 3. Display all cleaned columns so we can identify the actual target column
print("\n🧾 Cleaned Columns:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

# 4. Try to automatically find the target column with fuzzy matching
target_candidates = [col for col in df.columns if 'cirrhosis' in col.lower() or 'outcome' in col.lower()]
if target_candidates:
    print(f"\n✅ Found candidate for target column: {target_candidates[0]}")
    df.rename(columns={target_candidates[0]: 'Target'}, inplace=True)
else:
    raise KeyError("\n❌ Could not automatically identify the target column. Please check the column names above.")

# 5. Drop rows where target is missing
df = df.dropna(subset=['Target'])

# 6. Encode target column (e.g. YES/NO → 1/0)
df['Target'] = df['Target'].astype(str).str.upper().map({'YES': 1, 'NO': 0})

# 7. Drop rows where mapping failed (invalid labels became NaN)
df = df.dropna(subset=['Target'])

# 8. Drop SNO if it exists
if 'SNO' in df.columns:
    df.drop(columns=['SNO'], inplace=True)

# 9. Encode categorical columns
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# 10. Feature and target split
X = df.drop(['Target'], axis=1)
y = df['Target']

# 11. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 12. Train base model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# 13. Evaluate base model
print("\n📊 Classification Report (Base Model):")
print(classification_report(y_test, y_pred))
print("📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("📈 ROC AUC Score:", roc_auc_score(y_test, y_pred))

# 14. Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# 15. Show best model performance
print("\n✅ Best Parameters from GridSearch:", grid.best_params_)
best_rf = grid.best_estimator_
y_pred_best = best_rf.predict(X_test)

# 16. Final evaluation
print("\n📌 Tuned Model Classification Report:")
print(classification_report(y_test, y_pred_best))