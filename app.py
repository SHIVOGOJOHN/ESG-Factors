
from flask import Flask, request, jsonify, render_template, make_response
import io
import joblib
import pandas as pd
import json
import mysql.connector
import psycopg2

app = Flask(__name__)

# --- 1. Load Models and Supporting Files ---

# Load the best models
model_esg = joblib.load('esg_score_best_model.pkl')
model_financial_return = joblib.load('financial_return_best_model.pkl')
model_roe = joblib.load('roe_best_model.pkl')
model_profit_margin = joblib.load('profit_margin_best_model.pkl')

# Load the feature scaler and min/max values
scaler = joblib.load('feature_scaler.pkl')
with open('min_max_values.json', 'r') as f:
    min_max_values = json.load(f)

# Define the feature names in the correct order
feature_names = ['carbon_emissions', 'renewable_energy_ratio', 'waste_management_score', 'water_usage_intensity', 'biodiversity_score', 'excluded_sector', 'employee_satisfaction', 'diversity_index', 'turnover_rate', 'philanthropy_spend', 'supply_chain_ethics_score', 'data_privacy_compliance', 'independent_board_ratio', 'executive_pay_ratio', 'proxy_voting_score', 'risk_mgmt_score', 'audit_transparency', 'governance_framework_score']

# --- 2. Prediction Logic ---

def scale_to_percentage(value, min_val, max_val):
    """Scales a value to a 0-100 percentage."""
    return ((value - min_val) / (max_val - min_val)) * 100

@app.route('/download_template')
def download_template():
    df_template = pd.DataFrame(columns=feature_names)
    output = io.StringIO()
    df_template.to_csv(output, index=False)
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=esg_data_template.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data and scale features
        data = request.get_json(force=True)
        data_df = pd.DataFrame([data])[feature_names]
        scaled_features = scaler.transform(data_df)
        
        # --- Make Predictions ---
        pred_esg = model_esg.predict(scaled_features)[0]
        pred_financial_return = model_financial_return.predict(scaled_features)[0]
        pred_roe = model_roe.predict(scaled_features)[0]
        pred_profit_margin = model_profit_margin.predict(scaled_features)[0]
        
        # --- Scale to Percentages ---
        pct_esg = scale_to_percentage(pred_esg, min_max_values['esg_score']['min'], min_max_values['esg_score']['max'])
        pct_financial_return = scale_to_percentage(pred_financial_return, min_max_values['financial_return']['min'], min_max_values['financial_return']['max'])
        pct_roe = scale_to_percentage(pred_roe, min_max_values['roe']['min'], min_max_values['roe']['max'])
        pct_profit_margin = scale_to_percentage(pred_profit_margin, min_max_values['profit_margin']['min'], min_max_values['profit_margin']['max'])
        
        # Clamp ESG score to be between 0 and 100
        pct_esg = max(0, min(100, pct_esg))

        return jsonify({
            'esg_score': pct_esg,
            'financial_return': pct_financial_return,
            'roe': pct_roe,
            'profit_margin': pct_profit_margin
        })
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({'error': f'Prediction error: {str(e)}. Please check your input data.'}), 500

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if 'csvFile' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['csvFile']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            
            # Ensure all required feature_names are in the uploaded CSV
            if not all(feature in df.columns for feature in feature_names):
                missing_features = [feature for feature in feature_names if feature not in df.columns]
                return jsonify({'error': f'Missing columns in CSV: {", ".join(missing_features)}'}), 400

            # Process each row
            predictions = []
            for index, row in df.iterrows():
                data_df = pd.DataFrame([row[feature_names]])
                scaled_features = scaler.transform(data_df)
                
                pred_esg = model_esg.predict(scaled_features)[0]
                pred_financial_return = model_financial_return.predict(scaled_features)[0]
                pred_roe = model_roe.predict(scaled_features)[0]
                pred_profit_margin = model_profit_margin.predict(scaled_features)[0]
                
                pct_esg = scale_to_percentage(pred_esg, min_max_values['esg_score']['min'], min_max_values['esg_score']['max'])
                pct_financial_return = scale_to_percentage(pred_financial_return, min_max_values['financial_return']['min'], min_max_values['financial_return']['max'])
                pct_roe = scale_to_percentage(pred_roe, min_max_values['roe']['min'], min_max_values['roe']['max'])
                pct_profit_margin = scale_to_percentage(pred_profit_margin, min_max_values['profit_margin']['min'], min_max_values['profit_margin']['max'])
                
                pct_esg = max(0, min(100, pct_esg))

                feature_ranges = {feature: min_max_values[feature] for feature in feature_names}
                predictions.append({
                    'esg_score': pct_esg,
                    'financial_return': pct_financial_return,
                    'roe': pct_roe,
                    'profit_margin': pct_profit_margin,
                    'input_features': row[feature_names].astype(float).to_dict(), # Include input features
                    'feature_ranges': feature_ranges # Include feature ranges
                })
            
            return jsonify(predictions)

        except ValueError as ve:
            print(f"ValueError in /predict_csv: {ve}")
            return jsonify({'error': f'Data type error in CSV. Please ensure all feature columns contain numeric values. Specific error: {str(ve)}'}), 400
        except Exception as e:
            print(f"Error in /predict_csv: {e}")
            return jsonify({'error': f'Error processing CSV: {str(e)}. Please check your CSV format and data.'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict_from_db', methods=['POST'])
def predict_from_db():
    data = request.get_json(force=True)
    db_type = data.get('db_type')
    db_host = data.get('db_host')
    db_port = data.get('db_port')
    db_user = data.get('db_user')
    db_password = data.get('db_password')
    db_name = data.get('db_name')
    query_type = data.get('query_type')
    user_input = data.get('user_input')

    conn = None
    cursor = None
    try:
        if db_type == 'mysql':
            conn = mysql.connector.connect(
                host=db_host,
                port=db_port,
                user=db_user,
                password=db_password,
                database=db_name
            )
        elif db_type == 'postgresql':
            conn = psycopg2.connect(
                host=db_host,
                port=db_port,
                user=db_user,
                password=db_password,
                dbname=db_name
            )
        else:
            return jsonify({'error': 'Invalid database type selected.'}), 400

        cursor = conn.cursor()

        if query_type == 'custom_query':
            sql_query = user_input
        elif query_type == 'table_name':
            sql_query = f"SELECT * FROM {user_input}"
        else:
            return jsonify({'error': 'Invalid query type selected.'}), 400

        cursor.execute(sql_query)
        columns = [desc[0] for desc in cursor.description]
        db_data = cursor.fetchall()

        if not db_data:
            return jsonify({'error': 'No data fetched from the database.'}), 400

        df = pd.DataFrame(db_data, columns=columns)

        # Ensure all required feature_names are in the fetched data
        if not all(feature in df.columns for feature in feature_names):
            missing_features = [feature for feature in feature_names if feature not in df.columns]
            return jsonify({'error': f'Missing columns in database data: {", ".join(missing_features)}. Please ensure your query or table contains all required features.'}), 400

        predictions = []
        for index, row in df.iterrows():
            data_df = pd.DataFrame([row[feature_names]])
            scaled_features = scaler.transform(data_df)
            
            pred_esg = model_esg.predict(scaled_features)[0]
            pred_financial_return = model_financial_return.predict(scaled_features)[0]
            pred_roe = model_roe.predict(scaled_features)[0]
            pred_profit_margin = model_profit_margin.predict(scaled_features)[0]
            
            pct_esg = scale_to_percentage(pred_esg, min_max_values['esg_score']['min'], min_max_values['esg_score']['max'])
            pct_financial_return = scale_to_percentage(pred_financial_return, min_max_values['financial_return']['min'], min_max_values['financial_return']['max'])
            pct_roe = scale_to_percentage(pred_roe, min_max_values['roe']['min'], min_max_values['roe']['max'])
            pct_profit_margin = scale_to_percentage(pred_profit_margin, min_max_values['profit_margin']['min'], min_max_values['profit_margin']['max'])
            
            pct_esg = max(0, min(100, pct_esg))

            feature_ranges = {feature: min_max_values[feature] for feature in feature_names}
            predictions.append({
                'esg_score': pct_esg,
                'financial_return': pct_financial_return,
                'roe': pct_roe,
                'profit_margin': pct_profit_margin,
                'input_features': row[feature_names].astype(float).to_dict(),
                'feature_ranges': feature_ranges
            })
        
        return jsonify(predictions)

    except Exception as e:
        print(f"Error in /predict_from_db: {e}")
        return jsonify({'error': f'Database error: {str(e)}. Please check your connection details and query.'}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == '__main__':
    app.run(debug=True)
