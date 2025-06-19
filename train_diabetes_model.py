import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\asimw\Downloads\symptrack\diabetes.csv")

# Features and target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Impute missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Cross-validation
cv_acc = cross_val_score(model, X_imputed, y, cv=5, scoring='accuracy').mean()
cv_f1 = cross_val_score(model, X_imputed, y, cv=5, scoring='f1').mean()
print("CV Accuracy:", round(cv_acc, 3))
print("CV F1 Score:", round(cv_f1, 3))

# Save model and feature names
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(df.drop(columns=["Outcome"]).columns.tolist(), "diabetes_columns.pkl")
print("✅ Model and columns saved.")


