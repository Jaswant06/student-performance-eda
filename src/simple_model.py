import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "students.csv"
df = pd.read_csv(DATA_PATH)

df["average_score"] = (
    df["math_score"] + df["reading_score"] + df["writing_score"]
) / 3

X = df[["study_hours", "sleep_hours", "attendance"]]
y = df["average_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
print("Weights:")
for feature, weight in zip(X.columns, model.coef_):
    print(feature, weight)

print("\nBias:")
print(model.intercept_)


predictions = model.predict(X_test)

print("Predictions:")
print(predictions)

print("\nActual values:")
print(y_test.values)

print("\nMAE:")
print(mean_absolute_error(y_test, predictions))
