# Airline Passenger Satisfaction — Binary Classification

A machine learning project that predicts whether an airline passenger is **satisfied** or **neutral/dissatisfied** based on survey data covering flight experience, in-flight services, and passenger demographics.

---

## Project Overview

This is a supervised binary classification problem built on a dataset of ~26,000 airline passenger surveys. Three classifiers are trained, tuned, and compared to find the best-performing model.

**Target variable:** `satisfaction` → `satisfied` or `neutral or dissatisfied`

---

## Dataset

| Property | Value |
|---|---|
| Rows | 25,977 passengers |
| Features | 24 (after dropping ID/unnamed columns) |
| Task | Binary Classification |
| Class Balance | 56.1% neutral/dissatisfied · 43.9% satisfied |

**Feature types:**
- **Categorical:** Gender, Customer Type, Type of Travel, Class
- **Numerical:** Age, Flight Distance, Departure Delay, Arrival Delay
- **Ordinal (0–5 ratings):** Inflight wifi, Online boarding, Seat comfort, Food & drink, Inflight entertainment, Cleanliness, and 8 more service ratings

The dataset is reasonably balanced, so no resampling was needed.

---

## Models

Three classifiers were trained and tuned with hyperparameter search:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| KNN | **91.4%** | **92.0%** | 87.8% | **89.9%** |
| Decision Tree | 91.3% | 90.8% | **88.9%** | 89.9% |
| Logistic Regression | 85.6% | 83.8% | 82.7% | 83.2% |

KNN and Decision Tree perform comparably, both significantly outperforming Logistic Regression. ROC curves are included in the notebook for a visual comparison of all three models.

---

## Tech Stack

- Python 3.12
- pandas, NumPy
- scikit-learn
- matplotlib
- Jupyter Notebook

---

## Getting Started

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib jupyter
   ```

3. **Launch the notebook**
   ```bash
   jupyter notebook Classification.ipynb
   ```

---


## Key Findings

- In-flight service ratings (wifi, entertainment, seat comfort, online boarding) are among the strongest predictors of satisfaction.
- Business travelers and loyal customers tend to report higher satisfaction.
- KNN achieved the highest precision, making it the most conservative in predicting satisfaction — useful when false positives are costly.
