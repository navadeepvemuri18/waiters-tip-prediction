# =============================================================
# Waiter's Tip Prediction using Machine Learning
# Dataset: tips.csv (244 rows, 7 features)
# =============================================================

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------------
df = pd.read_csv('tips.csv')
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe().T)

# ---------------------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# ---------------------------------------------------------------

# Check for null values
print("\nNull values:\n", df.isnull().sum())

# Distribution plots for continuous columns
plt.subplots(figsize=(15, 8))
for i, col in enumerate(['total_bill', 'tip']):
    plt.subplot(2, 3, i + 1)
    sb.histplot(df[col], kde=True)
plt.tight_layout()
plt.savefig('plots/distribution_plots.png')
plt.close()

# Boxplots to check for outliers
plt.subplots(figsize=(15, 8))
for i, col in enumerate(['total_bill', 'tip']):
    plt.subplot(2, 3, i + 1)
    sb.boxplot(df[col])
plt.tight_layout()
plt.savefig('plots/boxplots.png')
plt.close()

# Check how many rows we'd lose if we remove outliers
print("\nFull shape:", df.shape,
      "| After outlier removal:", df[(df['total_bill'] < 45) & (df['tip'] < 7)].shape)

# Remove outliers
df = df[(df['total_bill'] < 45) & (df['tip'] < 7)]

# Count plots for categorical columns
import os
os.makedirs('plots', exist_ok=True)

feat = df.loc[:, 'sex':'size'].columns
plt.subplots(figsize=(15, 8))
for i, col in enumerate(feat):
    plt.subplot(2, 3, i + 1)
    sb.countplot(data=df, x=col)
plt.tight_layout()
plt.savefig('plots/countplots.png')
plt.close()

# Scatter: total_bill vs tip
plt.scatter(df['total_bill'], df['tip'])
plt.title('Total Bill vs Total Tip')
plt.xlabel('Total Bill')
plt.ylabel('Total Tip')
plt.savefig('plots/scatter_bill_tip.png')
plt.close()

# Group-by analysis
print("\nMean tip by size:\n", df.groupby('size').mean(numeric_only=True))
print("\nMean tip by time:\n", df.groupby('time').mean(numeric_only=True))
print("\nMean tip by day:\n",  df.groupby('day').mean(numeric_only=True))

# ---------------------------------------------------------------
# 3. Feature Engineering — Label Encoding
# ---------------------------------------------------------------
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == object:
        df[col] = le.fit_transform(df[col])

print("\nEncoded dataset:\n", df.head())

# Correlation heatmap
plt.figure(figsize=(7, 7))
sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png')
plt.close()

# ---------------------------------------------------------------
# 4. Model Development
# ---------------------------------------------------------------
features = df.drop('tip', axis=1)
target   = df['tip']

X_train, X_val, Y_train, Y_val = train_test_split(
    features, target, test_size=0.2, random_state=22
)
print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")

# Standardise
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# Train and evaluate four models
models = [
    LinearRegression(),
    XGBRegressor(),
    RandomForestRegressor(),
    AdaBoostRegressor()
]

print("\n--- Model Evaluation (MAE) ---")
for model in models:
    model.fit(X_train, Y_train)
    train_mae = mae(Y_train, model.predict(X_train))
    val_mae   = mae(Y_val,   model.predict(X_val))
    print(f"{model.__class__.__name__:<30} Train MAE: {train_mae:.4f}  |  Val MAE: {val_mae:.4f}")

# ---------------------------------------------------------------
# 5. Best Model — RandomForest — Feature Importances
# ---------------------------------------------------------------
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, Y_train)

importances = pd.Series(rf.feature_importances_, index=features.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sb.barplot(x=importances.values, y=importances.index)
plt.title('Feature Importances — Random Forest')
plt.tight_layout()
plt.savefig('plots/feature_importances.png')
plt.close()

print("\nBest Model — RandomForestRegressor")
print(f"Validation MAE: {mae(Y_val, rf.predict(X_val)):.4f}")
print("\nAll plots saved to plots/ directory.")
