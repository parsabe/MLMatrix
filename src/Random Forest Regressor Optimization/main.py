# In[1]: IMPORTS & DATA LOAD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV,
    learning_curve,
    validation_curve
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
data = pd.read_csv("winequalityN.csv")
print("Loaded data shape:", data.shape)

# In[2]: EDA — Overview
print("\nFirst 5 rows:")
display(data.head())

print("\nSummary statistics:")
display(data.describe())

print("\nMissing values per column:")
print(data.isnull().sum())

print("\nQuality value counts:")
print(data['quality'].value_counts())

# In[3]: EDA — Distributions & Correlations
# Histograms
data.hist(figsize=(12,10))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

# Boxplot example: alcohol vs quality
plt.figure(figsize=(8,5))
sns.boxplot(x='quality', y='alcohol', data=data)
plt.title("Alcohol by Quality")
plt.show()

# Correlation heatmap on numeric cols
num_df = data.select_dtypes(include=[np.number])
plt.figure(figsize=(12,10))
sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Numeric Feature Correlations")
plt.show()

# In[4]: PREPROCESSING
# 1) One-hot encode 'type'
df = pd.get_dummies(data, columns=['type'], prefix='type')

# 2) Impute any missing
imputer = SimpleImputer()
df[df.columns] = imputer.fit_transform(df)

# 3) Scale features (except target)
features = df.columns.drop('quality')
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Ready X, y
X = df.drop('quality', axis=1)
y = df['quality']
print("\nPreprocessed feature matrix shape:", X.shape)

# In[5]: TRAIN / VALID SPLIT
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train shape:", X_train.shape, "Validation shape:", X_val.shape)

# In[6]: BASELINE METRICS
def evaluate(model, X, y, name):
    preds = model.predict(X)
    return {
        'model': name,
        'RMSE': np.sqrt(mean_squared_error(y, preds)),
        'MAE': mean_absolute_error(y, preds),
        'R2': r2_score(y, preds)
    }

results = []
dummy = DummyRegressor(strategy='mean').fit(X_train, y_train)
results.append(evaluate(dummy, X_val, y_val, 'DummyRegressor'))
print("\nBaseline (Dummy) metrics:", results[-1])

# In[7]: RANDOM FOREST — RANDOMIZED SEARCH
rf = RandomForestRegressor(random_state=42)
rf_rand_params = {
    'n_estimators': [100, 300, 500, 800, 1200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}
rf_rand = RandomizedSearchCV(
    rf, rf_rand_params,
    n_iter=40, cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1, verbose=1, random_state=42
)
rf_rand.fit(X_train, y_train)
best_rf_rand = rf_rand.best_estimator_
print("\nRF Randomized best params:", rf_rand.best_params_)
results.append(evaluate(best_rf_rand, X_val, y_val, 'RF_Randomized'))

# In[8]: RANDOM FOREST — GRID SEARCH (REFINED)
rf_grid_params = {
    'n_estimators': sorted({best_rf_rand.n_estimators-200, best_rf_rand.n_estimators, best_rf_rand.n_estimators+200}),
    'max_depth': sorted({best_rf_rand.max_depth, (best_rf_rand.max_depth or 10)+5, (best_rf_rand.max_depth or 10)-5}),
    'min_samples_split': sorted({best_rf_rand.min_samples_split, best_rf_rand.min_samples_split+2}),
    'min_samples_leaf': sorted({best_rf_rand.min_samples_leaf, best_rf_rand.min_samples_leaf+1}),
    'max_features': [best_rf_rand.max_features]
}
rf_grid = GridSearchCV(
    rf, rf_grid_params,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1, verbose=1
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
print("\nRF Grid best params:", rf_grid.best_params_)
results.append(evaluate(best_rf, X_val, y_val, 'RF_Grid'))

# In[9]: XGBOOST — RANDOMIZED SEARCH
xgb = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    eval_metric='rmse',
    use_label_encoder=False
)
xgb_rand_params = {
    'n_estimators': [100, 300, 500, 800],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
xgb_rand = RandomizedSearchCV(
    xgb, xgb_rand_params,
    n_iter=30, cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1, verbose=1, random_state=42
)
xgb_rand.fit(X_train, y_train)
best_xgb_rand = xgb_rand.best_estimator_
print("\nXGB Randomized best params:", xgb_rand.best_params_)
results.append(evaluate(best_xgb_rand, X_val, y_val, 'XGB_Randomized'))

# In[10]: XGBOOST — EARLY STOPPING (FINAL FIT)
best_xgb_rand.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=30,
    verbose=False
)
results.append(evaluate(best_xgb_rand, X_val, y_val, 'XGB_EarlyStop'))
print("\nXGB with early stopping metrics:", results[-1])

# In[11]: XGBOOST — GRID SEARCH (REFINED)
xgb_grid_params = {
    'n_estimators': sorted({best_xgb_rand.n_estimators-100, best_xgb_rand.n_estimators, best_xgb_rand.n_estimators+100}),
    'max_depth': sorted({best_xgb_rand.max_depth, best_xgb_rand.max_depth+2}),
    'learning_rate': sorted({best_xgb_rand.learning_rate, best_xgb_rand.learning_rate/2}),
    'subsample': [best_xgb_rand.subsample],
    'colsample_bytree': [best_xgb_rand.colsample_bytree]
}
xgb_grid = GridSearchCV(
    xgb, xgb_grid_params,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1, verbose=1
)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_
print("\nXGB Grid best params:", xgb_grid.best_params_)
results.append(evaluate(best_xgb, X_val, y_val, 'XGB_Grid'))

# In[12]: RESULTS SUMMARY
df_results = pd.DataFrame(results).set_index('model')
print("\nModel comparison:")
display(df_results)

# In[13]: LEARNING CURVES
def plot_learning(est, name):
    ts, tr_sc, val_sc = learning_curve(
        est, X_train, y_train,
        cv=5, scoring='neg_root_mean_squared_error',
        train_sizes=np.linspace(0.1,1.0,5), n_jobs=-1
    )
    train_rmse = -tr_sc.mean(axis=1)
    val_rmse   = -val_sc.mean(axis=1)

    plt.figure()
    plt.plot(ts, train_rmse, 'o-', label='Train RMSE')
    plt.plot(ts, val_rmse, 'o-', label='CV RMSE')
    plt.title(f"Learning Curve: {name}")
    plt.xlabel("Train Size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid()
    plt.show()

plot_learning(best_rf, 'RF_Grid')
plot_learning(best_xgb, 'XGB_Grid')

# In[14]: VALIDATION CURVE (RF n_estimators)
param_range = [50, 100, 200, 400, 800]
tr_sc, val_sc = validation_curve(
    best_rf, X_train, y_train,
    param_name='n_estimators', param_range=param_range,
    cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1
)
plt.figure()
plt.plot(param_range, -tr_sc.mean(axis=1), 'o-', label='Train RMSE')
plt.plot(param_range, -val_sc.mean(axis=1), 'o-', label='CV RMSE')
plt.title('RF Validation Curve: n_estimators')
plt.xlabel('n_estimators')
plt.ylabel('RMSE')
plt.legend()
plt.grid()
plt.show()

# In[15]: FEATURE IMPORTANCES
def plot_importances(mdl, name):
    imp = pd.Series(mdl.feature_importances_, index=X_train.columns)
    imp = imp.sort_values(ascending=False).head(20)
    plt.figure(figsize=(8,6))
    imp.plot(kind='barh')
    plt.title(f"Top 20 Feature Importances: {name}")
    plt.gca().invert_yaxis()
    plt.show()

plot_importances(best_rf, 'RF_Grid')
plot_importances(best_xgb, 'XGB_Grid')

# In[16]: RESIDUAL PLOTS
def plot_residuals(mdl, X, y, name):
    preds = mdl.predict(X)
    res   = y - preds
    plt.figure()
    plt.scatter(preds, res, alpha=0.3)
    plt.axhline(0, linestyle='--', color='red')
    plt.title(f"Residuals vs Predicted: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.show()

plot_residuals(best_rf, X_val, y_val, 'RF_Grid')
plot_residuals(best_xgb, X_val, y_val, 'XGB_Grid')

