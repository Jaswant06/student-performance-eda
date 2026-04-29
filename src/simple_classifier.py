import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "students.csv"
df = pd.read_csv(DATA_PATH)

df["average_score"] = (
    df["math_score"] + df["reading_score"] + df["writing_score"]
) / 3

df["passed"] = (df["average_score"] >= 70).astype(int)

X = df[["study_hours", "sleep_hours", "attendance"]]
y = df["passed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

results = pd.DataFrame({
    "actual": y_test.values,
    "predicted": predictions
})

results["actual_label"] = results["actual"].map({1: "Pass", 0: "Fail"})
results["predicted_label"] = results["predicted"].map({1: "Pass", 0: "Fail"})

print(results[["actual_label", "predicted_label"]])

print("\nAccuracy:")
print(accuracy_score(y_test, predictions))

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

print("\nConfusion Matrix Meaning:")
print("Rows = actual values, columns = predicted values")
print("0 = Fail, 1 = Pass")
print(f"Actual Fail predicted Fail: {cm[0][0]}")
print(f"Actual Fail predicted Pass: {cm[0][1]}")
print(f"Actual Pass predicted Fail: {cm[1][0]}")
print(f"Actual Pass predicted Pass: {cm[1][1]}")


print("\nClassification Report:")
print(classification_report(y_test, predictions))
