import os
import urllib.request
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Configure directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
STATIC_IMG_DIR = os.path.join(PROJECT_ROOT, 'static', 'images')
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, 'notebook')

for d in [DATASET_DIR, MODEL_DIR, STATIC_IMG_DIR, NOTEBOOK_DIR]:
    os.makedirs(d, exist_ok=True)

DATASET_PATH = os.path.join(DATASET_DIR, 'loan_data.csv')
DATA_URL = "https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv"

def get_or_generate_dataset():
    """
    Downloads the standard Loan Prediction dataset from a GitHub mirror,
    or generates a highly realistic synthetic dataset if offline.
    """
    if os.path.exists(DATASET_PATH):
        print(f"Dataset already exists at {DATASET_PATH}")
        return pd.read_csv(DATASET_PATH)

    print("Attempting to download dataset...")
    try:
        urllib.request.urlretrieve(DATA_URL, DATASET_PATH)
        print(f"Dataset successfully downloaded and saved to {DATASET_PATH}")
        return pd.read_csv(DATASET_PATH)
    except Exception as e:
        print(f"Could not download dataset ({e}). Generating realistic synthetic dataset...")
        
        # Seed for reproducibility
        np.random.seed(42)
        n_samples = 614
        
        # Generating synthetic values with similar distribution to the original dataset
        genders = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.8, 0.2])
        married = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.65, 0.35])
        dependents = np.random.choice(['0', '1', '2', '3+'], size=n_samples, p=[0.57, 0.17, 0.16, 0.10])
        education = np.random.choice(['Graduate', 'Not Graduate'], size=n_samples, p=[0.78, 0.22])
        self_employed = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.14, 0.86])
        
        # Incomes follow a log-normal distribution roughly
        applicant_income = np.random.lognormal(mean=8.4, sigma=0.6, size=n_samples).astype(int)
        # Shift lower
        applicant_income = np.clip(applicant_income, 150, 81000)
        
        coapplicant_income_chance = np.random.choice([0, 1], size=n_samples, p=[0.45, 0.55])
        coapplicant_income = (coapplicant_income_chance * np.random.lognormal(mean=7.3, sigma=0.8, size=n_samples)).astype(int)
        
        # Loan amounts (in thousands) correlated with total income
        total_income = applicant_income + coapplicant_income
        loan_amount = (total_income * np.random.normal(0.022, 0.005, size=n_samples)).astype(int)
        loan_amount = np.clip(loan_amount, 9, 700)
        
        loan_term = np.random.choice([360.0, 180.0, 240.0, 300.0, 84.0, 120.0], size=n_samples, p=[0.85, 0.07, 0.03, 0.02, 0.01, 0.02])
        credit_history = np.random.choice([1.0, 0.0], size=n_samples, p=[0.84, 0.16])
        property_area = np.random.choice(['Semiurban', 'Urban', 'Rural'], size=n_samples, p=[0.38, 0.33, 0.29])
        
        # Deterministic rules + noise for Loan Status
        # Credit history is the strongest predictor
        loan_status_prob = np.zeros(n_samples)
        for i in range(n_samples):
            prob = 0.15  # Baseline approval chance
            if credit_history[i] == 1.0:
                prob += 0.60
            if education[i] == 'Graduate':
                prob += 0.05
            if married[i] == 'Yes':
                prob += 0.05
            if property_area[i] == 'Semiurban':
                prob += 0.05
            
            # Debt-to-income penalty
            income_ratio = (loan_amount[i] * 1000) / (total_income[i] + 1e-5)
            if income_ratio > 3.0:
                prob -= 0.15
            
            loan_status_prob[i] = np.clip(prob, 0.02, 0.98)
            
        loan_status = np.where(np.random.rand(n_samples) < loan_status_prob, 'Y', 'N')
        
        # Inject missing values (approx 5% per column, matching real dataset)
        def inject_nan(arr, p=0.05):
            arr = arr.astype(object)
            mask = np.random.rand(len(arr)) < p
            arr[mask] = np.nan
            return arr
            
        genders = inject_nan(genders, 0.02)
        married = inject_nan(married, 0.01)
        dependents = inject_nan(dependents, 0.025)
        self_employed = inject_nan(self_employed, 0.05)
        loan_amount = inject_nan(loan_amount, 0.035).astype(float)
        loan_term = inject_nan(loan_term, 0.023).astype(float)
        credit_history = inject_nan(credit_history, 0.08).astype(float)
        
        df = pd.DataFrame({
            'Loan_ID': [f"LP{1000+i:04d}" for i in range(n_samples)],
            'Gender': genders,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history,
            'Property_Area': property_area,
            'Loan_Status': loan_status
        })
        
        df.to_csv(DATASET_PATH, index=False)
        print(f"Synthetic dataset written to {DATASET_PATH}")
        return df

def run_model_pipeline():
    df = get_or_generate_dataset()
    
    # 1. Split Data
    X = df.drop(columns=['Loan_ID', 'Loan_Status'])
    y = df['Loan_Status'].map({'Y': 1, 'N': 0})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # Define Numerical & Categorical Columns
    num_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    cat_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
    
    # 2. Setup Transformers
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])
    
    # Preprocess training features
    print("Fitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the fitted preprocessor
    preprocessor_path = os.path.join(MODEL_DIR, 'preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Saved preprocessor to {preprocessor_path}")
    
    # 3. Model Configurations
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'XGBoost': XGBClassifier(
            random_state=42, 
            n_estimators=100, 
            max_depth=4, 
            learning_rate=0.05, 
            use_label_encoder=False, 
            eval_metric='logloss'
        )
    }
    
    results = []
    trained_models = {}
    
    # Plot configurations for ROC curves
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_processed, y_train)
        trained_models[name] = model
        
        # Predictions
        preds = model.predict(X_test_processed)
        probs = model.predict_proba(X_test_processed)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'ROC-AUC': auc,
            'Confusion Matrix': cm.tolist()
        })
        
        # ROC plot line
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    
    # Finalize and save ROC plot
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    roc_plot_path = os.path.join(STATIC_IMG_DIR, 'roc_curve.png')
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC comparison plot saved to {roc_plot_path}")
    
    # Convert results list to DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n--- Model Evaluation Results Summary ---")
    print(results_df.drop(columns=['Confusion Matrix']).to_string(index=False))
    
    # Save comparison to CSV
    comparison_csv = os.path.join(NOTEBOOK_DIR, 'model_comparison.csv')
    results_df.to_csv(comparison_csv, index=False)
    print(f"Comparison metrics CSV saved to {comparison_csv}")
    
    # Plot model metrics comparison bar chart
    metrics_melted = pd.melt(
        results_df.drop(columns=['Confusion Matrix']), 
        id_vars=['Model'], 
        value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
        var_name='Metric', 
        value_name='Value'
    )
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x='Metric', y='Value', hue='Model', data=metrics_melted, palette='viridis')
    plt.title('Comparison of Machine Learning Models', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.ylabel('Score')
    plt.xlabel('Evaluation Metric')
    plt.legend(title='Classifier', loc='lower right')
    comparison_plot_path = os.path.join(STATIC_IMG_DIR, 'model_comparison.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison bar plot saved to {comparison_plot_path}")
    
    # Select Best Model automatically (by F1 Score, fallback to Accuracy)
    best_idx = results_df['Accuracy'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_model = trained_models[best_model_name]
    
    print(f"\nWinner Model: {best_model_name} with Accuracy = {results_df.loc[best_idx, 'Accuracy']:.4f}")
    
    # Save the winning model
    best_model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Saved best model to {best_model_path}")

if __name__ == '__main__':
    run_model_pipeline()
