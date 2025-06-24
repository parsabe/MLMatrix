# Random Forest Regressor Optimization


A Jupyter notebook–based pipeline for regression on tabular data, extended with advanced preprocessing, model tuning, and evaluation.

## Project Structure

- **`MachineLearning_project-2.ipynb`**  
  The sole notebook containing the complete data pipeline, from EDA through final model evaluation.

## Code Overview


This repository was originally developed by [@anitatehrani](https://github.com/anitatehrani). The following features have been added to enhance the workflow:

- Comprehensive **EDA** with visualizations and correlation analysis  
- Robust **preprocessing**: encoding, imputation, scaling  
- Standardized **train/test splitting** with reproducible random seed  
- **Baseline** DummyRegressor metrics for context  
- Dual-stage **Random Forest** tuning: broad **randomized search** + focused **grid search**  
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
  
## Contributors

- This project is mainly being implemented by [@anitatehrani](https://github.com/anitatehrani)  

