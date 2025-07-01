

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

# Load the dataset
df = pd.read_csv('static/data/esg_factors_data.csv')

# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 0', 'company_id'])

# --- 1. Data Preparation ---

# Define feature and target columns
feature_cols = ['carbon_emissions', 'renewable_energy_ratio', 'waste_management_score', 'water_usage_intensity', 'biodiversity_score', 'excluded_sector', 'employee_satisfaction', 'diversity_index', 'turnover_rate', 'philanthropy_spend', 'supply_chain_ethics_score', 'data_privacy_compliance', 'independent_board_ratio', 'executive_pay_ratio', 'proxy_voting_score', 'risk_mgmt_score', 'audit_transparency', 'governance_framework_score']
target_cols = ['financial_return', 'roe', 'profit_margin']

# Create the ESG score
env_cols = ['carbon_emissions', 'renewable_energy_ratio', 'waste_management_score', 'water_usage_intensity', 'biodiversity_score']
soc_cols = ['employee_satisfaction', 'diversity_index', 'turnover_rate', 'philanthropy_spend', 'supply_chain_ethics_score', 'data_privacy_compliance']
gov_cols = ['independent_board_ratio', 'executive_pay_ratio', 'proxy_voting_score', 'risk_mgmt_score', 'audit_transparency', 'governance_framework_score']

scaler_features = MinMaxScaler()
df[feature_cols] = scaler_features.fit_transform(df[feature_cols])

df['esg_score'] = (df[env_cols].mean(axis=1) + df[soc_cols].mean(axis=1) + df[gov_cols].mean(axis=1)) / 3 * 100
target_cols.append('esg_score')

# --- 2. Model Training and Selection ---

# Define models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'SVM': SVR(),
    'MLP': MLPRegressor(random_state=42, max_iter=500)
}

X = df[feature_cols]

# Store min/max for percentage scaling
min_max_values = {}

for target in target_cols:
    print(f"--- Training models for {target} ---")
    y = df[target]
    
    # Store min/max for scaling the final prediction
    min_max_values[target] = {'min': y.min(), 'max': y.max()}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_model_name = ''
    best_model_score = -np.inf
    best_model = None
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"{name}: R2 = {r2:.4f}, MSE = {mse:.4f}")
        
        if r2 > best_model_score:
            best_model_score = r2
            best_model_name = name
            best_model = model
            
    print(f"\nBest model for {target} is {best_model_name}\n")
    joblib.dump(best_model, f'{target}_best_model.pkl')

# --- 3. Save Supporting Files ---

joblib.dump(scaler_features, 'feature_scaler.pkl')

with open('min_max_values.json', 'w') as f:
    json.dump(min_max_values, f)

print("Advanced model training complete. Best models and scalers saved.")

