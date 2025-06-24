# Machine Learning Project 2

A Jupyter notebook–based pipeline for regression on tabular data, extended with advanced preprocessing, model tuning, and evaluation.

## Project Structure

- **`MachineLearning_project-2.ipynb`**  
  The sole notebook containing the complete data pipeline, from EDA through final model evaluation.

## Code Overview

1. **Exploratory Data Analysis (EDA)**  
   - Display first rows, summary statistics, and missing-value counts  
   - Feature distributions with histograms and boxplots  
   - Correlation heatmap (numeric features only)

2. **Preprocessing**  
   - One-hot encoding of categorical columns  
   - Imputation of missing values with `SimpleImputer`  
   - Min–max scaling of all features (excluding target)

3. **Train/Test Split**  
   - 80/20 random split with a fixed seed for reproducibility

4. **Baseline Metrics**  
   - Dummy regressor (mean strategy) as a performance floor  
   - Evaluated via RMSE, MAE, and R²

5. **Random Forest Regressor**  
   - **RandomizedSearchCV** over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`  
   - **GridSearchCV** refinement around the best randomized results  
   - Final evaluation on the hold-out set

6. **XGBoost Regressor**  
   - **RandomizedSearchCV** on common hyperparameters  
   - **True early stopping** via the native `xgboost.train()` API and `DMatrix`, halting after 30 rounds without improvement  
   - **GridSearchCV** refinement around the best randomized results  
   - Evaluation of each stage (randomized, early-stop, grid) with RMSE, MAE, and R²

7. **Model Comparison & Diagnostics**  
   - Tabulated comparison of all models and tuning stages  
   - Learning curves to diagnose under- or over-fitting  
   - Validation curves for key hyperparameters  
   - Top-20 feature-importance plots  
   - Residuals vs. predicted scatter plots

## Improvements

This repository was originally developed by [@anitatehrani](https://github.com/anitatehrani). The following features have been added to enhance the workflow:

- Comprehensive **EDA** with visualizations and correlation analysis  
- Robust **preprocessing**: encoding, imputation, scaling  
- Standardized **train/test splitting** with reproducible random seed  
- **Baseline** DummyRegressor metrics for context  
- Dual-stage **Random Forest** tuning: broad randomized search + focused grid search  
- Full **XGBoost** workflow:  
  - Randomized search  
  - Early stopping via native API  
  - Local grid search  
- Uniform **evaluation metrics** (RMSE, MAE, R²) for all models  
- Diagnostic **learning & validation curves**  
- **Feature-importance** and **residuals** plots for interpretability  
- Clear **results table** summarizing model performance

## Usage

1. Clone the repository and install dependencies.  
2. Open `MachineLearning_project-2.ipynb` in Jupyter.  
3. Provide your own dataset by updating the CSV path and target column.  
4. Run all cells to reproduce the full analysis and tuning pipeline.  
5. Inspect the final performance table and diagnostic plots for insights.

## Dependencies

- Python 3.8+  
- pandas  
- numpy  
- scikit-learn  
- xgboost  
- matplotlib  
- seaborn

## Report

- **Canva Design:**  
  [View on Canva](https://www.canva.com/design/DAF_gFQy9cU/QWdyZUKmBciXM0MElTQTUg/view?utm_content=DAF_gFQy9cU&utm_campaign=designshare&utm_medium=link&utm_source=editor)

## Contributors

- This project is being implemented by [@anitatehrani](https://github.com/anitatehrani)  
- Implementation enhancements by **Parsa Besharat** (this fork)

