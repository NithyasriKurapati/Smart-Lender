import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.pkl')
COMPARISON_CSV_PATH = os.path.join(PROJECT_ROOT, 'notebook', 'model_comparison.csv')

# Global variables for model and preprocessor
model = None
preprocessor = None

def load_assets():
    global model, preprocessor
    if os.path.exists(BEST_MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        try:
            with open(BEST_MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(PREPROCESSOR_PATH, 'rb') as f:
                preprocessor = pickle.load(f)
            print("Successfully loaded model and preprocessor.")
        except Exception as e:
            print(f"Error loading model assets: {e}")
    else:
        print("Warning: Model files not found. Please run train_model.py first.")

# Initial load
load_assets()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    # Load model comparisons from CSV if available, to render dynamically
    comparison_data = []
    if os.path.exists(COMPARISON_CSV_PATH):
        try:
            df_comp = pd.read_csv(COMPARISON_CSV_PATH)
            # Round numeric columns for display
            for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']:
                if col in df_comp.columns:
                    df_comp[col] = df_comp[col].round(4)
            comparison_data = df_comp.to_dict(orient='records')
        except Exception as e:
            print(f"Could not load comparisons: {e}")
            
    return render_template('about.html', comparisons=comparison_data)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html', scroll_to_form=True)
        
    global model, preprocessor
    # Reload models if not loaded yet
    if model is None or preprocessor is None:
        load_assets()
        if model is None or preprocessor is None:
            return render_template('result.html', error="Model assets are not available. Please run model training first.")

    try:
        # Extract inputs from form
        gender = request.form.get('Gender', 'Male')
        married = request.form.get('Married', 'No')
        dependents = request.form.get('Dependents', '0')
        education = request.form.get('Education', 'Graduate')
        self_employed = request.form.get('Self_Employed', 'No')
        
        try:
            applicant_income = float(request.form.get('ApplicantIncome', 0))
            coapplicant_income = float(request.form.get('CoapplicantIncome', 0))
            loan_amount = float(request.form.get('LoanAmount', 0))
            loan_term = float(request.form.get('Loan_Amount_Term', 360))
            credit_history = float(request.form.get('Credit_History', 1.0))
        except ValueError:
            return render_template('index.html', error="Invalid numerical values provided.", scroll_to_form=True)
            
        property_area = request.form.get('Property_Area', 'Semiurban')

        # Create input DataFrame
        input_data = pd.DataFrame([{
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history,
            'Property_Area': property_area
        }])

        # Process input
        input_processed = preprocessor.transform(input_data)
        
        # Predict status and probabilities
        pred = model.predict(input_processed)[0]
        prob_approve = model.predict_proba(input_processed)[0][1]
        
        # Map output
        status = "Approved" if pred == 1 else "Rejected"
        prob_percent = round(prob_approve * 100, 1)
        
        # Risk assessment logic
        # 1. Bad Credit History represents high risk
        # 2. High Loan-to-Income ratios represent high risk
        total_income = applicant_income + coapplicant_income
        loan_to_income = (loan_amount * 1000) / total_income if total_income > 0 else 999
        
        if credit_history == 0.0:
            risk_level = "High"
            risk_class = "danger"
        elif prob_approve < 0.4 or loan_to_income > 4.5:
            risk_level = "High"
            risk_class = "danger"
        elif prob_approve < 0.72 or loan_to_income > 3.0:
            risk_level = "Medium"
            risk_class = "warning"
        else:
            risk_level = "Low"
            risk_class = "success"

        # Summary of details to show in results
        applicant_summary = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self Employed': self_employed,
            'Applicant Income': f"${applicant_income:,.2f}",
            'Coapplicant Income': f"${coapplicant_income:,.2f}",
            'Loan Amount': f"${loan_amount * 1000:,.2f} (${loan_amount}k)",
            'Term': f"{int(loan_term)} months",
            'Credit History': "Good / Cleared" if credit_history == 1.0 else "Poor / Delinquent",
            'Property Area': property_area
        }

        return render_template(
            'result.html',
            status=status,
            probability=prob_percent,
            risk_level=risk_level,
            risk_class=risk_class,
            applicant=applicant_summary
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('result.html', error=f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
