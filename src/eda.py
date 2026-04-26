import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(BASE_DIR / "data" / "students.csv")

# Basic inspection
print(df.head())
print("Shape:", df.shape)
print("Columns:", df.columns)

print(df.info())
print(df.describe())
print(df.isnull().sum())

# Feature engineering
df["average_score"] = (
    df["math_score"] + df["reading_score"] + df["writing_score"]
) / 3

print(df[["student_id", "average_score"]])

# Plot 1: Study hours vs score

plt.figure()
sns.scatterplot(data=df, x="study_hours", y="average_score", hue="test_prep")
plt.title("Study Hours vs Average Score")

plt.savefig(PLOTS_DIR / "study_hours_vs_score.png", bbox_inches="tight")
plt.close()

# Plot 2: Test prep vs score

plt.figure()
sns.boxplot(data=df, x="test_prep", y="average_score")
plt.title("Average Score by Test Preparation")

plt.savefig(PLOTS_DIR / "test_prep_vs_score.png", bbox_inches="tight")
plt.close()

# Plot 3: Correlation heatmap

numeric_cols = [
    "study_hours",
    "sleep_hours",
    "attendance",
    "math_score",
    "reading_score",
    "writing_score",
    "average_score"
]

corr = df[numeric_cols].corr()

plt.figure()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")

plt.savefig(PLOTS_DIR / "correlation_heatmap.png", bbox_inches="tight")
plt.close()

# Additional Analysis

# 1. Average score by gender
avg_by_gender = df.groupby("gender")["average_score"].mean()
print("\nAverage Score by Gender:\n", avg_by_gender)

plt.figure()
avg_by_gender.plot(kind="bar", title="Average Score by Gender")
plt.ylabel("Average Score")
plt.savefig(PLOTS_DIR / "avg_score_by_gender.png", bbox_inches="tight")
plt.close()


# 2. Average score by test preparation
avg_by_prep = df.groupby("test_prep")["average_score"].mean()
print("\nAverage Score by Test Prep:\n", avg_by_prep)

plt.figure()
avg_by_prep.plot(kind="bar", title="Average Score by Test Preparation")
plt.ylabel("Average Score")
plt.savefig(PLOTS_DIR / "avg_score_by_test_prep.png", bbox_inches="tight")
plt.close()


# 3. Correlation between attendance and average score
corr_attendance = df["attendance"].corr(df["average_score"])
print("\nCorrelation (Attendance vs Avg Score):", corr_attendance)

plt.figure()
sns.regplot(data=df, x="attendance", y="average_score")
plt.title("Attendance vs Average Score")
plt.savefig(PLOTS_DIR / "attendance_vs_score.png", bbox_inches="tight")
plt.close()


# 4. Top 5 students
top_5 = df.sort_values(by="average_score", ascending=False).head(5)
print("\nTop 5 Students:\n", top_5[["student_id", "average_score"]])

# 5. Bottom 5 students
bottom_5 = df.sort_values(by="average_score", ascending=True).head(5)
print("\nBottom 5 Students:\n", bottom_5[["student_id", "average_score"]])