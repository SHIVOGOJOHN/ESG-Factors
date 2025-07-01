
import pandas as pd

# Load the dataset
df = pd.read_csv('static/data/esg_factors_data.csv')

# Display basic information about the dataset
print("Dataset Info:")
df.info()

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())
