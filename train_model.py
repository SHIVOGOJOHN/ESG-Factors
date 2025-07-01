import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
df = pd.read_csv('static/data/esg_factors_data.csv')

# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 0', 'company_id'])

# Define ESG columns and financial targets
env_cols = ['carbon_emissions', 'renewable_energy_ratio', 'waste_management_score', 'water_usage_intensity', 'biodiversity_score']
soc_cols = ['employee_satisfaction', 'diversity_index', 'turnover_rate', 'philanthropy_spend', 'supply_chain_ethics_score', 'data_privacy_compliance']
gov_cols = ['independent_board_ratio', 'executive_pay_ratio', 'proxy_voting_score', 'risk_mgmt_score', 'audit_transparency', 'governance_framework_score']
financial_cols = ['financial_return', 'roe', 'profit_margin']

# Normalize the ESG feature columns
scaler = MinMaxScaler()
features_to_scale = env_cols + soc_cols + gov_cols
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# --- 1. ESG Score Model --- 

# Calculate E, S, and G scores
df['E_score'] = df[env_cols].mean(axis=1)
df['S_score'] = df[soc_cols].mean(axis=1)
df['G_score'] = df[gov_cols].mean(axis=1)

# Calculate the final ESG score (target variable)
df['ESG_score'] = (df['E_score'] + df['S_score'] + df['G_score']) / 3 * 100

# Define features (X) and target (y) for ESG model
X_esg = df[features_to_scale]
y_esg = df['ESG_score']

# Train and save the ESG model
esg_model = LinearRegression()
esg_model.fit(X_esg, y_esg)
joblib.dump(esg_model, 'esg_model.pkl')

# --- 2. Financial Metrics Models --- 

# Use the same features (X_esg) to predict financial metrics
X_financial = df[features_to_scale]

for target in financial_cols:
    y_financial = df[target]
    
    # Train a model for each financial target
    model = LinearRegression()
    model.fit(X_financial, y_financial)
    
    # Save each model
    joblib.dump(model, f'{target}_model.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

print("All models and the scaler have been saved successfully.")