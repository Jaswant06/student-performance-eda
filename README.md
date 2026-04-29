# Student Performance EDA + Prediction

## Project Goal

This project analyzes a student performance dataset to understand what factors are related to academic performance. It also builds simple machine learning models to predict:

- a student's average score
- whether a student passed or failed

This is my first end-to-end beginner AI/ML project covering EDA, visualization, regression, classification, and model evaluation.

## Dataset

The dataset contains student records with the following columns:

- student_id
- gender
- study_hours
- sleep_hours
- attendance
- test_prep
- math_score
- reading_score
- writing_score

Two new columns were created:

- average_score: average of math, reading, and writing scores
- passed: 1 if average_score is at least 70, otherwise 0

## Tools Used

- Python
- pandas
- matplotlib
- seaborn
- scikit-learn

## Exploratory Data Analysis

The EDA focused on questions such as:

- Do students who study more get higher scores?
- Does test preparation improve average score?
- Is attendance related to performance?
- Are math, reading, and writing scores correlated?
- Which students performed the best and worst?

Plots were saved in the `plots/` folder.

## Key Insights

- Students with higher study hours generally had higher average scores.
- Students who completed test preparation usually performed better.
- Attendance had a strong positive relationship with average score.
- Math, reading, and writing scores were strongly correlated.
- The dataset is small and clean, so model performance is likely better than it would be on real-world data.

## Regression Model

The regression model predicts `average_score`.

Input features:

- study_hours
- sleep_hours
- attendance

Target:

- average_score

Result:

```text
Mean Absolute Error: 1.29
```

This means the model's predictions were off by about `1.29` marks on average.

## Classification Model

The classification model predicts whether a student passed or failed.

Input features:

- study_hours
- sleep_hours
- attendance

Target:

- passed

Label meaning:

```text
1 = Passed
0 = Failed
```

Result:

```text
Accuracy: 1.0
```

This means the model correctly predicted all students in the test set.

## Confusion Matrix

```text
[[ 7  0]
 [ 0 13]]
```

Rows represent actual values. Columns represent predicted values.

Interpretation:

- 7 students actually failed and were correctly predicted as Fail.
- 0 students actually failed but were incorrectly predicted as Pass.
- 0 students actually passed but were incorrectly predicted as Fail.
- 13 students actually passed and were correctly predicted as Pass.

The model made 20 correct predictions out of 20 test examples.

## Important Note

The model achieved very high performance because the dataset is small, clean, and highly patterned.

In real-world datasets, performance is usually lower because data often contains missing values, noise, outliers, and more complex relationships.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run EDA:

```bash
python src/eda.py
```

Run regression model:

```bash
python src/simple_model.py
```

Run classification model:

```bash
python src/simple_classifier.py
```

## Project Structure

```text
student-performance-eda/
+-- data/
|   +-- students.csv
+-- plots/
|   +-- attendance_vs_score.png
|   +-- avg_score_by_gender.png
|   +-- avg_score_by_test_prep.png
|   +-- correlation_heatmap.png
|   +-- study_hours_vs_score.png
|   +-- test_prep_vs_score.png
+-- src/
|   +-- eda.py
|   +-- simple_model.py
|   +-- simple_classifier.py
+-- .gitignore
+-- README.md
+-- requirements.txt
```

## What I Learned

- How to load and inspect data with pandas
- How to create new features from existing columns
- How to visualize relationships in data
- How to build a simple regression model
- How to build a simple classification model
- How to evaluate models using MAE, accuracy, confusion matrix, precision, recall, and F1-score
- Why very high accuracy on a small clean dataset should be interpreted carefully
