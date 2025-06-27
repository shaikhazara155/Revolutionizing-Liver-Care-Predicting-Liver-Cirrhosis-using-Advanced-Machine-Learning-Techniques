import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Show current Python environment
print("Running with Python executable:", sys.executable)

# === Load Excel File ===
file_path = 'HealthCareData.xlsx'  # 🔁 Replace with your actual file path
df = pd.read_excel(file_path)

# === Clean and Normalize Column Names ===
df.columns = df.columns.str.strip().str.replace('\n', ' ', regex=True).str.replace('\r', '', regex=True)
print("📋 Available columns:", df.columns.tolist())

# === Identify Numeric Columns to Convert ===
numeric_candidates = ['TG', 'LDL', 'Total_Bilirubin_mgdl', 'AG_Ratio']
cols_to_convert = [col for col in numeric_candidates if col in df.columns]
if cols_to_convert:
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

# === Fill Missing Numeric Values with Median ===
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# === Fill Missing Categorical Values with Mode ===
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        print(f"⚠️ Column '{col}' has no mode. Check for uniform or empty values.")

# === Detect or Validate Target Column ===
target_name = 'Predicted ValueOut_ComePatient suffering from liver cirrosis or not'
target_candidates = [col for col in df.columns if 'cirrosis' in col.lower()]
if target_name not in df.columns and target_candidates:
    target_name = target_candidates[0]
    print(f"✅ Target column detected as: '{target_name}'")
elif target_name in df.columns:
    print(f"✅ Using defined target column: '{target_name}'")
else:
    print(f"❌ Target column not found. Please check your column names.")
    target_name = None

# === Drop Rows with Missing Target and Encode ===
if target_name:
    df = df[df[target_name].notna()]
    if df[target_name].dtype == 'object':
        target_le = LabelEncoder()
        df[target_name] = target_le.fit_transform(df[target_name].astype(str))
        joblib.dump(target_le, 'target_encoder.pkl')

# === Encode Categorical Features ===
le_dict = {}
for col in categorical_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# === Save Label Encoders for Later Use ===
joblib.dump(le_dict, 'label_encoders.pkl')

# ✅ Summary
print("✅ Preprocessing complete. Final DataFrame shape:", df.shape)