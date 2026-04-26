# Student Performance EDA

## 📌 Project Goal
This project performs Exploratory Data Analysis (EDA) on a student performance dataset to understand factors affecting academic scores.

---

## 📊 Dataset Columns
- student_id
- gender
- study_hours
- sleep_hours
- attendance
- test_prep
- math_score
- reading_score
- writing_score

---

## 🛠 Tools Used
- Python
- pandas
- matplotlib
- seaborn

---

## ❓ Key Questions
- Do study hours affect performance?
- Does test preparation improve scores?
- How does attendance relate to performance?
- Are there differences based on gender?
- What factors are most correlated with scores?

---

## 📈 Insights
- Higher study hours generally lead to higher scores
- Students with test preparation perform better
- Attendance shows positive correlation with performance
- Math, reading, and writing scores are strongly correlated
- High-performing students consistently have higher study hours and attendance

---

## 🖼 Sample Plots
Plots are saved in the `plots/` directory:
- Study Hours vs Score
- Test Prep vs Score
- Correlation Heatmap
- Average Score by Gender
- Attendance vs Score

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
python src/eda.py