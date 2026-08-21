# Kahunda: Machine Learning Pipelines

Welcome to the Kahunda repository. This repository showcases end-to-end predictive analytics and machine learning pipelines utilizing `Scikit-Learn` to solve real-world industry forecasting challenges.

---

## 🛠️ Tech stack & Tools

* **Language:** Python
* **ML Libraries:** Scikit-Learn, Pandas, Numpy
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab/ Jupyter Notebooks

---

## 🚀 Projects Included

### 1. Boston Rideshare Forecasting (`boston_rideshare_forecasting.ipynb`)
* **Objective:** Analyze and predict rideshare pricing dynamics and demand trends for Uber and Lyft services in Boston, MA.
* **Key Focus:** Exploratory Data Analysis (EDA), feature engineering around ride types, distance, and real-time environmental factors, and model optimization.
* **Key Results:** Built and compared linear baselines, decision trees and random forests to minimize validation errors through systematic feature engineering and hyperparameter tuning.

### 2. Walmart Sales Forecasting (`walmart_sales_forecasting.ipynb`)
* **Objective:** Predict seasonal sales spikes and holiday demand fluctuations across 81 distinct Walmart departments.
* **Key Focus:** Exploratory Data Analysis (EDA), feature engineering around markdown columns, IsHoliday and model optimization.
* **Key Results:** Evaluated multiple regression baselines. The **`RandomForestRegressor`** was selected as the optimal model, delivering the lowest validation error score by effectively capturing non-linear feature interactions and holiday patterns.


### 3. Rossman Sales Prediction (`rossmandataset.ipynb`)
## Project Title & Overview

**Rossmann Store Sales Prediction**

This project was based on the popular Kaggle competition with the objective of predicting daily sales across 1,115 Rossmann Stores in Germany. Prediction accuracy was evaluated using **Root Mean Square Percentage Error (RMSPE)** across train, validation, and test splits. RMSPE evaluates percentage deviation relative to actual sales rather than absolute dollar values. This ensures fair evaluation by accounting for small stores equally without giving disproportionate weight to high-volume flagship locations.

I evaluated three distinct models: **Linear Regression**, **Decision Trees**, and **Random Forest**. Random Forest emerged as the optimal architecture by effectively mitigating overfitting between training and validation sets. As a practitioner four months into machine learning, I referenced top community submissions and utilized Gemini as an educational guide. My final submission yielded a **Private Leaderboard RMSPE of 0.1337**, placing in the top ~1,900 range.

### Evaluation Metric

$$\text{RMSPE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( \frac{y_i - \hat{y}_i}{y_i} \right)^2}$$

---

## Methodology

### 1. Data Preprocessing
* **Dataset Merging:** Combined `store.csv` metadata into `train.csv` (`merged_df`) and `test.csv` (`merged_test_df`).
* **Imputation Strategy:**
  * Imputed missing `CompetitionDistance` values using the **median** rather than the mean to avoid sensitivity to right-skewed revenue outliers.
  * Filled missing `Promo2SinceYear` and `Promo2SinceWeek` with `0` (indicating no active Promo2 program).
  * Categorized missing `PromoInterval` entries under a dedicated `'Unknown'` category.
  * Imputed missing `CompetitionOpenSinceMonth` with `1` (January) and `CompetitionOpenSinceYear` with a baseline year of `1990`.
  * Filled missing values in the test set `Open` column with `1` (assuming stores remained open unless specified).
* **Zero-Sales Filtering:** Dropped all records where `Sales == 0` (or `Open == 0`). Since closed stores strictly generate zero revenue, excluding them prevents division-by-zero errors in RMSPE calculations and keeps models focused on active commercial operations.

### 2. Feature Engineering & Data Leakage Prevention
* **Temporal Features:** Parsed date values to extract `Day`, `Month`, `Year`, and `WeekOfYear`.
* **External Weather Integration:** Ingested historic weather metrics (temperature, rainfall, snowfall) for Frankfurt via the OpenMeteo API as a central proxy for regional German weather conditions.
* **Target & Feature Transformation:**
  * Applied a log transformation to the target variable ($y \to \log(1+y)$) to stabilize variance across peak sales days and align loss optimization with percentage-based metrics.
  * Log-transformed skewed numerical features (`CompetitionDistance`) to improve numerical stability and linear model convergence.
* **Domain Indicators:** Constructed `CompetitionStartDate`, `CompetitionActive`, `CompetitionMonthsOpen`, `Log_CompDistance`, `Public_Holiday`, and `IsPromo2Active` (via custom month mappings).
* **Target Encoding & Leakage Prevention:** Aggregated statistics (`Salesmean`, `Salesstd`, `StoreGroup`) were computed **strictly after** splitting the dataset into `train_df` and `val_df`. Computing target stats prior to splitting would leak target distribution information into the validation set, artificially deflating validation RMSPE.
* **Feature Pruning:**
  * Dropped `Customers` from model features. Because customer counts are unavailable at inference time in the test set, including this feature during training would induce severe feature leakage.
  * Evaluated feature encodings: raw integer IDs for `Store` work natively with Decision Trees and Random Forests due to ordinal threshold splitting, whereas keeping Store as integer provided a consistent baseline feature set across models without introducing a 1,000+ column sparse matrix through One-Hot Encoding.

### 3. Model Architecture & Hyperparameter Tuning
* **Baseline Setup:** Defined custom evaluation functions (`RETURN_MEAN` and `RMSPE`) to compute training and validation baselines as comparative benchmarks.
* **Linear Regressor:** Trained on log-transformed sales, converting predictions back to raw values prior to RMSPE calculation. Outperformed the naive baseline.
* **Decision Tree Regressor:** Captured non-linear patterns and complex interactions, but exhibited severe overfitting between train and validation splits.
* **Random Forest Regressor:** Reduced variance and controlled tree-level overfitting compared to single Decision Trees.
* **Hyperparameter Tuning & Reproducibility:**
  * Performed hyperparameter tuning to optimize Random Forest generalization.
  * Enforced strict experiment reproducibility across all random sampling and model definitions by standardizing `random_state=42` and `np.random.seed(42)`.
  * Configured parameter testing sequentially (`n_jobs=1`) during validation checks, transitioning to full multi-core parallel execution (`n_jobs=-1`) for fitting the final production model.

---

## 🛠️ Environment & Setup

To replicate these workflows and run the Jupyter Notebooks locally, ensure you have Python installed and run the following commands:

```bash
# Clone the repository
git clone https://github.com/kennethuchegbu22-GOD/Kahunda.git

# Navigate into the project folder
cd Kahunda

# Install the required data science and ML dependencies
pip install -r requirements.txt
