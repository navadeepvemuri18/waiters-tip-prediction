# 🍽️ Waiter's Tip Prediction using Machine Learning

Predict the tip amount a customer will leave at a restaurant based on features like total bill, party size, day of visit, and more.

---

## 📁 Project Structure

```
waiters-tip-prediction/
├── tips.csv              # Dataset (244 rows, 7 columns)
├── tip_prediction.py     # Full ML pipeline script
├── requirements.txt      # Python dependencies
├── plots/                # Auto-generated EDA & result plots
└── README.md
```

---

## 📊 Dataset

The **Tips dataset** contains 244 records collected from a restaurant. Each row represents a single dining visit.

| Column       | Description                          |
|--------------|--------------------------------------|
| total_bill   | Total bill amount (USD)              |
| tip          | Tip amount (USD) — **target**        |
| sex          | Gender of the bill payer             |
| smoker       | Whether the party included a smoker  |
| day          | Day of the week                      |
| time         | Meal time (Lunch / Dinner)           |
| size         | Number of people in the party        |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/waiters-tip-prediction.git
cd waiters-tip-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the script

```bash
python tip_prediction.py
```

All EDA plots will be saved to the `plots/` folder.

---

## 🔍 Key Findings (EDA)

- The distribution of `total_bill` and `tip` is **positively skewed**.
- Weekend footfall is higher than weekdays.
- Dinner visits attract higher tips than lunch.
- Tip amount is **directly proportional** to party size.

---

## 🤖 Models Compared

| Model                  | Train MAE | Validation MAE |
|------------------------|-----------|----------------|
| Linear Regression      | ~0.69     | ~0.79          |
| XGBoost Regressor      | ~0.44     | ~0.74          |
| **Random Forest**      | **~0.28** | **~0.72**      |
| AdaBoost Regressor     | ~0.66     | ~0.85          |

✅ **Random Forest Regressor** achieves the lowest validation MAE.

---

## 🛠️ Tech Stack

- Python 3.x
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- XGBoost

---

## 📚 Reference

- [GeeksforGeeks — Waiter's Tip Prediction](https://www.geeksforgeeks.org/machine-learning/waiters-tip-prediction-using-machine-learning/)
