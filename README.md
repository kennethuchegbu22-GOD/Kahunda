# Kahunda: Machine Learning Pipelines

Welcome to the Kahunda repository. This repository showcases end-to-end predictive analytics and machine learning pipelines utilizing `scikit-learn` to solve real-world industry forecasting challenges.

---

## 🚀 Projects Included

### 1. Boston Rideshare Forecasting (`boston_rideshare_forecasting.ipynb`)
* **Objective:** Analyze and predict rideshare pricing dynamics and demand trends for Uber and Lyft services in Boston, MA.
* **Key Focus:** Exploratory Data Analysis (EDA), feature engineering around ride types, distance, and real-time environmental factors, and model optimization.

### 2. Walmart Sales Forecasting (`walmart_sales_forecasting.ipynb`)
* **Objective:** Predict seasonal sales spikes and holiday demand fluctuations across 81 distinct Walmart departments.
* **Model Selection:** Evaluated multiple regression baselines. The **`RandomForestRegressor`** was selected as the optimal model, delivering the lowest validation error score by effectively capturing non-linear feature interactions and holiday patterns.

---

## 🛠️ Environment & Setup

To replicate these workflows and run the Jupyter Notebooks locally, ensure you have Python installed and run the following commands:

```bash
# Clone the repository
git clone [https://github.com/kennethuchegbu22-GOD/Kahunda.git](https://github.com/kennethuchegbu22-GOD/Kahunda.git)

# Navigate into the project folder
cd Kahunda

# Install the required data science and ML dependencies
pip install -r requirements.txt
