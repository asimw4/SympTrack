import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Load the dataset
df = pd.read_csv(r"C:\Users\asimw\Downloads\heart.csv")

# Replace missing values marked with '?' with NaN
df.replace('?', pd.NA, inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Convert all columns to numeric
df = df.apply(pd.to_numeric)

# Convert target: 0 = no disease, 1-4 = disease → binary classification
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Split features and target
X = df.drop(columns=['target'])
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a basic logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict + evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {acc:.2f}")
print(f"F1 Score: {f1:.2f}")

# Save model
joblib.dump(model, "risk_model.pkl")
print("✅ Model saved as risk_model.pkl")