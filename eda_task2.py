
# eda_task2.py
"""
Exploratory Data Analysis (EDA) script for the Titanic dataset.

Saves:
 - fig_histograms.png
 - fig_boxplots.png
 - fig_corr.png
 - fig_age_vs_fare_scatter.png
 - fig_age_by_survival_hist.png

Outputs some summary text to stdout and saves figures to the current folder.

(Brackets []) explain difficult words briefly where needed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import warnings
warnings.filterwarnings("ignore")

# ---- Config ----
DATA_PATH = "tit-data.csv"        # change if your filename differs
OUT_DIR = "."                     # change to e.g. "figures" to save elsewhere
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load data ----
df = pd.read_csv(DATA_PATH)
print("Loaded:", DATA_PATH, "shape =", df.shape)

# ---- Basic info ----
print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Missing values ---")
print(df.isnull().sum())

print("\n--- Statistical summary (numeric) ---")
print(df.describe().T)

# ---- Numeric columns list ----
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumeric columns:", numeric_cols)

# ---- Histograms for numeric features ----
plt.figure(figsize=(12, 10))
df[numeric_cols].hist(bins=20, edgecolor="black", layout=(int(np.ceil(len(numeric_cols)/3)), 3))
plt.suptitle("Histograms for numeric features", fontsize=16)
hist_path = os.path.join(OUT_DIR, "fig_histograms.png")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(hist_path)
plt.close()
print("Saved histogram figure:", hist_path)

# ---- Boxplots to check outliers ----
# Create a grid of boxplots (will skip if no numeric cols)
if numeric_cols:
    cols = numeric_cols
    n = len(cols)
    cols_per_row = 3
    rows = int(np.ceil(n / cols_per_row))
    plt.figure(figsize=(cols_per_row*4, rows*3))
    for i, col in enumerate(cols, 1):
        plt.subplot(rows, cols_per_row, i)
        sns.boxplot(y=df[col], color="skyblue")
        plt.title(col)
    plt.tight_layout()
    box_path = os.path.join(OUT_DIR, "fig_boxplots.png")
    plt.savefig(box_path)
    plt.close()
    print("Saved boxplots figure:", box_path)

# ---- Correlation matrix (numeric only) ----
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=False)
    plt.title("Correlation matrix (numeric features only)")
    corr_path = os.path.join(OUT_DIR, "fig_corr.png")
    plt.tight_layout()
    plt.savefig(corr_path)
    plt.close()
    print("Saved correlation heatmap:", corr_path)
else:
    print("Not enough numeric columns for correlation heatmap.")

# ---- Pairplot / limited pair relationships ----
# Pairplot can be slow; pick a small subset if many numeric cols.
pair_cols = [c for c in ["Survived", "Age", "Fare", "Pclass"] if c in df.columns]
if len(pair_cols) >= 2:
    try:
        sns.pairplot(df[pair_cols], hue="Survived", corner=True)
        pair_path = os.path.join(OUT_DIR, "fig_pairplot.png")
        plt.savefig(pair_path)
        plt.close()
        print("Saved pairplot (limited):", pair_path)
    except Exception as e:
        print("Pairplot skipped (too slow or failed):", e)

# ---- Categorical breakdowns (counts) ----
import matplotlib
matplotlib.use('Agg')  # ensure non-interactive backend for scripts

def save_countplot(col, title=None, fname=None):
    if col in df.columns:
        plt.figure(figsize=(6,4))
        sns.countplot(x=col, hue="Survived", data=df)
        plt.title(title or f"Counts by {col} (by Survived)")
        plt.tight_layout()
        path = os.path.join(OUT_DIR, fname or f"fig_count_{col}.png")
        plt.savefig(path)
        plt.close()
        print("Saved countplot:", path)

save_countplot("Sex", title="Survival Count by Sex", fname="fig_count_sex.png")
save_countplot("Pclass", title="Survival Count by Pclass", fname="fig_count_pclass.png")
if "Embarked" in df.columns:
    save_countplot("Embarked", title="Survival Count by Embarked", fname="fig_count_embarked.png")

# ---- Scatter: Age vs Fare colored by Survived (interactive & png fallback) ----
if ("Age" in df.columns) and ("Fare" in df.columns) and ("Survived" in df.columns):
    try:
        fig = px.scatter(df, x="Age", y="Fare", color=df["Survived"].astype(str),
                         labels={"color":"Survived"}, title="Age vs Fare (interactive)")
        scatter_html = os.path.join(OUT_DIR, "fig_age_vs_fare_interactive.html")
        fig.write_html(scatter_html)
        print("Saved interactive scatter (html):", scatter_html)
    except Exception as e:
        print("Plotly interactive scatter skipped:", e)

    # Save static png using matplotlib
    plt.figure(figsize=(6,5))
    sns.scatterplot(x="Age", y="Fare", hue="Survived", data=df, palette="Set1", alpha=0.7)
    plt.title("Age vs Fare (colored by Survived)")
    scatter_png = os.path.join(OUT_DIR, "fig_age_vs_fare_scatter.png")
    plt.tight_layout()
    plt.savefig(scatter_png)
    plt.close()
    print("Saved static scatter png:", scatter_png)

# ---- Distribution of Age by Survived (histogram) ----
if "Age" in df.columns and "Survived" in df.columns:
    plt.figure(figsize=(8,4))
    sns.histplot(data=df, x="Age", hue="Survived", bins=30, kde=False, multiple="layer")
    plt.title("Distribution of Age by Survival")
    age_hist = os.path.join(OUT_DIR, "fig_age_by_survival_hist.png")
    plt.tight_layout()
    plt.savefig(age_hist)
    plt.close()
    print("Saved Age-by-Survived histogram:", age_hist)

# ---- Basic textual inferences (sample) ----
print("\n--- Quick Inferences (manual check recommended) ---")
# Female vs Male survival (if available)
if set(["Sex","Survived"]).issubset(df.columns):
    try:
        survival_by_sex = df.groupby("Sex")["Survived"].mean()
        print("Survival rate by Sex (0=male,1=female):")
        print(survival_by_sex)
    except Exception:
        pass

# Pclass vs survival
if set(["Pclass","Survived"]).issubset(df.columns):
    print("\nSurvival rate by Pclass:")
    print(df.groupby("Pclass")["Survived"].mean())

# Fare stats by survival
if "Fare" in df.columns:
    print("\nFare statistics (survived / not):")
    print(df.groupby("Survived")["Fare"].describe())

print("\nEDA complete. Figures and summary saved to current folder.")
