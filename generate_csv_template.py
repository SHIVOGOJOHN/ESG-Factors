import pandas as pd

feature_names = [
    'carbon_emissions', 'renewable_energy_ratio', 'waste_management_score',
    'water_usage_intensity', 'biodiversity_score', 'excluded_sector',
    'employee_satisfaction', 'diversity_index', 'turnover_rate',
    'philanthropy_spend', 'supply_chain_ethics_score', 'data_privacy_compliance',
    'independent_board_ratio', 'executive_pay_ratio', 'proxy_voting_score',
    'risk_mgmt_score', 'audit_transparency', 'governance_framework_score'
]

df = pd.DataFrame(columns=feature_names)
df.to_csv('C:/Users/A/.vscode/Flask/ESG Investing System/static/data/esg_factors_data.csv', index=False)
