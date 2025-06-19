import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Load data
df = pd.read_csv("hypertension_dataset.csv")

# Map target variable
df["Hypertension"] = df["Hypertension"].map({"Low": 0, "High": 1})

# Keep only features with physiological relevance
keep_cols = [
    "Age", "BMI", "Cholesterol", "Systolic_BP", "Diastolic_BP", "Glucose",
    "Heart_Rate", "Sleep_Duration", "Gender", "Diabetes", "Alcohol_Intake",
    "Smoking_Status", "Stress_Level", "Salt_Intake", "Physical_Activity_Level",
    "Family_History", "HDL", "LDL", "Triglycerides"
]
df = df[["Hypertension"] + keep_cols].dropna()

# Split features and target
y = df["Hypertension"]
X = df.drop(columns="Hypertension")

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

# Save model and feature columns
joblib.dump(model, "hypertension_model.pkl")
joblib.dump(X.columns.tolist(), "hypertension_columns.pkl")
print("✅ Model and feature columns saved.")

