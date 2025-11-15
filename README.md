# Titanic — Exploratory Data Analysis (EDA)
**Task 2**

This repository contains the Exploratory Data Analysis performed on the Titanic dataset as part of the AIML internship by **Elevate Labs**.

## Files
- `tit-data.csv` — Raw dataset used for EDA.
- `eda_task2.ipynb` — Jupyter/Colab notebook containing code, plots, and analysis.
- `figures/` — (optional) saved plot images used in the report.
- `EDA_summary.txt` — Short conclusions and observations.

## Objective
Understand the dataset using summary statistics and visualizations (histograms, boxplots, correlation matrix, pairplots, and categorical analysis).

## Key steps performed
1. Summary statistics (mean, median, std, min/max, count).  
2. Visualized distributions (histograms) for numeric features.  
3. Boxplots to detect outliers for Age and Fare.  
4. Correlation matrix (numeric features only) to inspect relationships.  
5. Pairplots and scatter plots to visualize relationships between `Age`, `Fare`, `Pclass`, and `Survived`.  
6. Categorical breakdowns (Survived by Sex, Pclass, Embarked).  
7. Interactive plots (Plotly) for Age vs Fare colored by survival (optional).

## Major observations / inferences
- Females had a higher survival rate than males.  
- First-class passengers (Pclass=1) had a noticeably higher survival rate than 2nd/3rd class.  
- Higher Fare generally correlates with higher survival.  
- Children (lower Age) tended to have better survival in several visualizations.  
- There are outliers in `Fare` and a few extreme ages — noted during boxplot inspection.

## How to run the notebook locally or in Colab
- **Colab**: Upload `eda_task2.ipynb` to Colab (or open via Drive) and run cells.
- **Local**:
  - Create and activate Python environment
  - Install: `pip install pandas numpy matplotlib seaborn plotly`
  - Run the notebook with Jupyter: `jupyter notebook eda_task2.ipynb`

## Author
**Thrishool M S**
Elevate Labs — Task 1
