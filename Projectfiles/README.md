# Smart Lender – AI-Powered Loan Approval Prediction System

Smart Lender is a complete end-to-end Machine Learning web application designed to predict retail loan eligibility. Using custom preprocessing pipelines and modern ensemble models, the system assesses applicant default probabilities in real-time, providing bank officers with a responsive, visual dashboard.

---

## Technical Stack

*   **Backend Framework:** Python 3.10+, Flask, Jinja2 Templates
*   **Machine Learning:** Scikit-learn, XGBoost, Pandas, NumPy, SciPy
*   **Visualizations:** Matplotlib, Seaborn
*   **Frontend UI:** HTML5, CSS3 (Custom Glassmorphism), Bootstrap 5, Font Awesome Icons, JavaScript
*   **Deployment:** IBM Cloud Foundry / Gunicorn ready

---

## Directory Structure

```
SmartLender/
│
├── app.py                  # Flask web application entrypoint
├── requirements.txt        # Package dependencies list
├── Procfile                # Heroku/IBM Web server process config
├── manifest.yml            # IBM Cloud deployment manifest
├── README.md               # Setup and deployment documentation
│
├── dataset/
│   └── loan_data.csv       # Training & testing dataset
│
├── model/
│   ├── best_model.pkl      # Pickled champion model (Random Forest)
│   └── preprocessor.pkl    # Pickled preprocessing pipeline
│
├── templates/
│   ├── index.html          # Main application dashboard & submission form
│   ├── result.html         # Loan approval & risk assessment output
│   └── about.html          # Performance benchmarks & ROC curves
│
├── static/
│   ├── css/
│   │   └── style.css       # Core custom animations, dark navy theme styling
│   ├── js/
│   │   └── script.js       # Dynamic SVG gauge rendering & validation
│   └── images/
│       ├── roc_curve.png   # ROC comparison evaluation plot
│       └── model_comparison.png # Metrics validation bar plot
│
└── notebook/
    ├── SmartLender.ipynb   # Interactive Jupyter Notebook for EDA & model training
    └── model_comparison.csv # Model performance CSV data
```

---

## Installation & Local Execution

### 1. Set up a virtual environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Models
This script fetches the classic Analytics Vidhya Loan Prediction dataset, imputes null parameters, fits scaling & encoding pipelines, trains the 4 classifiers, generates evaluation plots, and serializes the champion classifier.
```bash
python train_model.py
```

### 4. Run the Flask Web Application
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000`.

---

## Model Evaluation Benchmarks

The models are automatically trained, scored, and compared. Below are the validation metrics computed during the pipeline execution:

| Model / Classifier | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** (Winner) | **85.37%** | **83.17%** | **98.82%** | **90.32%** | **0.8173** |
| XGBoost | 84.55% | 83.67% | 96.47% | 89.62% | 0.7536 |
| Decision Tree | 82.11% | 82.47% | 94.12% | 87.91% | 0.7259 |
| K-Nearest Neighbors | 80.49% | 79.05% | 97.65% | 87.37% | 0.7751 |

---

## IBM Cloud Deployment

The project is structured for seamless deployment onto IBM Cloud. Follow these instructions:

### Prerequisites
1. Install the [IBM Cloud CLI](https://cloud.ibm.com/docs/cli).
2. Install the Cloud Foundry plugin: `ibmcloud cf install`.

### Deployment Steps
1. **Login to IBM Cloud:**
   ```bash
   ibmcloud login --sso
   ```
2. **Target Cloud Foundry Org and Space:**
   ```bash
   ibmcloud target --cf
   ```
3. **Deploy Web App:**
   ```bash
   ibmcloud cf push
   ```
   *The deployment container reads `manifest.yml` for allocation limits, uses `Procfile` to start Gunicorn, and loads python dependencies from `requirements.txt` using the standard Python buildpack.*

---

## Screenshots Placeholders

* **Dashboard Landing Page & Underwriting Form:** Place screenshot showing the landing sections and loan underwriting parameters inputs cards.
* **Eligible Approval (Green Gauge):** Place screenshot showing the positive approval status, risk metrics badge (Low Risk), and applicant summaries grid.
* **Ineligible Rejection (Red Gauge):** Place screenshot showing the negative status warning, risk level (High Risk), and applicant particulars.
* **Models Performance Diagnostics:** Place screenshot showing the about page comparison tables, ROC Curves, and comparison bar graphs.
