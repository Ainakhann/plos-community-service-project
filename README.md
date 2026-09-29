# Predictive Model for Patient Length of Stay in Rehabilitation Hospitals (PLOS)

> **BCA Final Year College Academic Project**  
> A Machine Learning & Full-Stack Web Application for Clinical Decision Support and Rehabilitation Hospital Bed Capacity Forecasting.

---

## 1. Project Title
**Predictive Model for Patient Length of Stay in Rehabilitation Hospitals (PLOS)**

---

## 2. Problem Description
Patient Length of Stay (PLOS) represents the total number of days a patient is expected to remain in a specialized rehabilitation hospital following acute hospitalization or surgical procedure. Inefficient bed utilization, unexpected prolonged stays, and sudden readmissions lead to hospital overcrowding, increased waiting times, staff burnout, and delayed rehabilitation access for emergency patients.

Predicting PLOS at the time of admission using machine learning provides hospital administrators and clinical care teams with actionable forecasting intelligence to plan therapy schedules, manage bed turnover, optimize nurse-to-patient ratios, and initiate post-discharge arrangements early.

---

## 3. Objectives
- **Accurate Stay Duration Prediction**: Predict continuous length of stay in days with low error (MAE < 1.5 days).
- **Clinical Categorization**: Stratify stay duration into **Short Stay (1–7 days)**, **Medium Stay (8–14 days)**, and **Long Stay (15+ days)**.
- **Risk Level Stratification**: Assign clinical complexity risk levels (**Low**, **Moderate**, **High**).
- **Hospital Workflow Optimization**: Assist hospital staff with resource allocation, bed management, staff planning, discharge planning, and reduced waiting times.
- **Interactive Healthcare Dashboard**: Provide hospital leadership with visual analytics, subgroup comparisons, and model performance metrics.

---

## 4. Proposed Solution
A full-stack Python web application featuring:
1. **Machine Learning Pipeline**: Trained regression and classification models (`scikit-learn`) incorporating 17 patient clinical features and 6 engineered domain risk indicators.
2. **Flask Web Engine**: Modular web application providing interactive patient prediction forms, analytics views, model performance metrics, and REST API.
3. **Database Integration**: Persistence using **MongoDB** with automatic local JSON fallback for offline environments.
4. **Modern UI/UX**: Healthcare light-theme interface built with Bootstrap 5 and Chart.js.

---

## 5. Features
- 🏥 **Interactive Hospital Dashboard**: Real-time KPI summary widgets and stay distribution charts.
- 📋 **Patient Prediction Form**: Pre-filled sample patient option ("Load Sample Patient" for Stroke Demo) and real-time range sliders.
- 📊 **Visual Analytics**: Interactive charts showing stay duration by diagnosis, severity, rehabilitation setting, and age group.
- 🎯 **High Accuracy ML Model**: Achieves **>95% category classification accuracy** and **R² > 0.95**.
- 🔌 **REST API (`POST /predict`)**: Standardized JSON endpoint for third-party integration.
- 💾 **Dual Storage Engine**: MongoDB integration with transparent local fallback.
- ⚠️ **Academic Disclaimer & Guidance**: Clear clinical disclaimer banners.

---

## 6. Technology Stack
- **Programming Language**: Python 3.9+ (Tested on Python 3.13)
- **Web Framework**: Flask 2.0+ / Flask 3.1
- **Machine Learning Engine**: `scikit-learn`, `pandas`, `numpy`, `joblib`
- **Database Engine**: MongoDB (`pymongo`) with local JSON fallback
- **Frontend UI/UX**: HTML5, CSS3, JavaScript (ES6+), Bootstrap 5.3, FontAwesome 6
- **Data Visualizations**: Chart.js

---

## 7. System Architecture

```
[ Patient Data Input / Web Form / REST API ]
                    │
                    ▼
          [ Flask Web Server (app.py) ]
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
[ Feature Engineering ]  [ Data Preprocessing ]
(Age group, Risk score)  (Imputer, OneHotEncoder, Scaler)
        │                       │
        └───────────┬───────────┘
                    ▼
      [ ML Model Inference (src/prediction.py) ]
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
[ Length of Stay (Days) ] [ Stay Category & Risk ]
(e.g., 18 Days)          (Long Stay, High Risk)
        │                       │
        └───────────┬───────────┘
                    ▼
     [ Database Storage (MongoDB / Fallback JSON) ]
                    │
                    ▼
  [ HTML Dashboard & Chart.js Analytics Visualization ]
```

---

## 8. Machine Learning Workflow
1. **Synthetic Dataset Generation** (`src/generate_data.py`): 1,000 realistic clinical patient records.
2. **Data Cleaning & Missing Value Handling** (`src/preprocessing.py`): Median imputation for numerical fields, mode for categorical fields.
3. **Feature Engineering** (`src/feature_engineering.py`): Creation of domain risk scores, comorbidity burden index, mobility risk indicators.
4. **Encoding & Scaling**: One-Hot Encoding for categorical variables and StandardScaler for numerical features.
5. **Model Training & Comparison** (`train_model.py`): Comparison of Random Forest, Gradient Boosting, Decision Tree, and Linear Regression algorithms.
6. **Model Serialization**: Saving trained model to `models/plos_model.pkl` and preprocessor to `models/preprocessor.pkl`.

---

## 9. Dataset Description
- **File Location**: `data/patient_data.csv`
- **Record Count**: 1,000 synthetic patient records (No real patient data used).
- **Features Included**:
  - `patient_id`: Unique identifier (e.g., PAT-1001)
  - `age`: Age in years (18–88)
  - `gender`: Male, Female
  - `diagnosis`: Stroke, Spinal Cord Injury, Traumatic Brain Injury, Joint Replacement, Amputation, Cardiac Rehab, Neurological Disorder, General Rehab
  - `rehab_type`: Neurological, Orthopedic, Cardiac, General Physical, Pulmonary
  - `admission_type`: Emergency, Elective, Referral, Transfer
  - `previous_hospitalization`: Yes, No
  - `previous_surgery`: Yes, No
  - `severity`: Low, Medium, High, Severe
  - `mobility_score`: Mobility score (0–100)
  - `comorbidities`: Co-existing condition count (0–4)
  - `diabetes`: Yes, No
  - `hypertension`: Yes, No
  - `insurance_type`: Medicare, Medicaid, Private, Uninsured
  - `functional_score`: Functional Independence Measure (0–100 FIM score)
  - `pain_level`: Pain score (0–10)
  - `therapy_frequency`: Sessions per week (1–7)
  - `previous_admissions`: Prior hospital admissions count (0–3)
  - `length_of_stay`: Target continuous variable in days (1–45)

---

## 10. Feature Engineering
The feature engineering module (`src/feature_engineering.py`) derives:
- **`age_group`**: Categorizes age into Young (<40), Adult (40–60), Senior (60–75), and Elderly (>75).
- **`comorbidity_burden`**: Composite score summing raw comorbidities, diabetes flag, and hypertension flag.
- **`mobility_risk`**: Binary indicator (1 if mobility score < 40).
- **`functional_recovery_risk`**: Binary indicator (1 if functional score < 40).
- **`prev_hosp_flag`**: Binary flag indicating prior hospitalization.
- **`overall_risk_score`**: Normalized composite index (0–100) combining clinical severity, mobility impairment, functional dependence, and chronic disease burden.

---

## 11. Model Training
Run model training script:
```bash
python train_model.py
```
This evaluates four model candidates and automatically selects the best performing model.

---

## 12. Evaluation Metrics
Current Model Evaluation Results on Test Set (200 records):
- **Continuous Regression Metrics**:
  - **MAE (Mean Absolute Error)**: ~0.87 Days
  - **MSE (Mean Squared Error)**: ~1.14
  - **RMSE (Root Mean Squared Error)**: ~1.07 Days
  - **R² Score**: ~0.976
- **Stay Category Classification Metrics**:
  - **Stay Category Accuracy**: **96.5%** (Exceeds >80% target)
  - **Weighted Precision**: 96.5%
  - **Weighted Recall**: 96.5%
  - **Weighted F1 Score**: 96.2%

---

## 13. MongoDB Setup
MongoDB stores prediction logs in the `predictions` collection of `plos_database`.

1. Install MongoDB Community Server locally.
2. Start MongoDB service:
   - Windows Service: Starts automatically on port `27017`.
3. Configure `MONGO_URI` in `.env`:
   ```env
   MONGO_URI=mongodb://localhost:27017/plos_database
   ```
4. **Fallback Mechanism**: If MongoDB is not running, the application seamlessly logs predictions to `data/local_predictions_fallback.json` without throwing errors or crashing.

---

## 14. Installation Instructions

1. Clone or open the repository folder in VS Code / Antigravity:
   ```bash
   cd "PLOS_Project"
   ```

2. Create a Python virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate virtual environment:
   - **Windows**:
     ```cmd
     venv\Scripts\activate
     ```
   - **Linux / macOS**:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 15. How to Run the Project

1. **Train Model & Generate Dataset**:
   ```bash
   python train_model.py
   ```

2. **Start Flask Application**:
   ```bash
   python app.py
   ```

3. **Access Application**:
   Open web browser at:
   `http://127.0.0.1:5000`

---

## 16. API Documentation

### Endpoint: `POST /predict`
Accepts patient JSON payload and returns stay duration prediction.

#### Request Body Example:
```json
{
  "age": 65,
  "gender": "Male",
  "diagnosis": "Stroke",
  "rehab_type": "Neurological",
  "admission_type": "Emergency",
  "severity": "High",
  "mobility_score": 35,
  "comorbidities": 3,
  "diabetes": "Yes",
  "hypertension": "Yes",
  "insurance_type": "Private",
  "functional_score": 40,
  "pain_level": 7,
  "therapy_frequency": 5,
  "previous_admissions": 2
}
```

#### Response Example:
```json
{
  "status": "success",
  "predicted_days": 18,
  "stay_category": "Long Stay",
  "risk_level": "High",
  "factors": [
    "High initial severity level ('High')",
    "Low physical mobility score (35.0/100)",
    "Impaired functional independence score (40.0/100)",
    "Multiple co-existing medical conditions (3 comorbidities)",
    "Complex primary diagnosis: Stroke",
    "High pain score (7.0/10)"
  ],
  "disclaimer": "This system is an academic demonstration and does not provide medical advice."
}
```

---

## 17. Folder Structure
```
PLOS_Project/
│
├── app.py                     # Flask Web Application entry point
├── train_model.py             # Dataset generation, training, and evaluation script
├── requirements.txt           # Python dependencies
├── README.md                  # Detailed project documentation
├── .env.example               # Environment variables configuration
│
├── data/
│   ├── patient_data.csv       # Synthetic dataset (1,000 records)
│   └── local_predictions_fallback.json # Local storage fallback file
│
├── models/
│   ├── plos_model.pkl         # Exported trained ML model
│   ├── preprocessor.pkl       # Exported data preprocessor pipeline
│   └── model_metrics.json     # Saved evaluation metrics JSON
│
├── src/
│   ├── __init__.py
│   ├── generate_data.py       # Synthetic patient dataset generator
│   ├── preprocessing.py        # Data cleaning, encoding, scaling pipeline
│   ├── feature_engineering.py # Derived feature calculation module
│   ├── prediction.py          # Inference engine & risk mapping
│   ├── evaluation.py          # Metric calculation functions
│   └── database.py            # MongoDB connection & local fallback manager
│
├── templates/
│   ├── base.html              # Base layout template with navbar & footer
│   ├── index.html             # Main hospital analytics dashboard
│   ├── prediction.html        # Patient input prediction form
│   ├── result.html            # Prediction result card
│   ├── analytics.html         # Interactive analytics charts page
│   ├── performance.html       # ML performance evaluation page
│   └── about.html             # Project details & SDG 3 impact
│
└── static/
    ├── css/
    │   └── style.css          # Healthcare dashboard custom CSS
    └── js/
        └── script.js          # Interactive JavaScript UI functions
```

---

## 18. Future Enhancements
- Integration with electronic health record (EHR) FHIR standards.
- Deep Learning / LSTM temporal recovery trajectory modeling.
- Real-time hospital bed allocation map and notification system.

---

## 19. SDG & Community Impact
Aligned with **UN Sustainable Development Goal 3: Good Health and Well-Being**:
- **Optimized Rehabilitation Services**: Improves post-acute rehabilitation scheduling.
- **Enhanced Bed Availability**: Reduces bottlenecks in hospital emergency departments.
- **Efficient Resource Delivery**: Minimizes patient waiting times.

---

## 20. Academic Disclaimer
> **IMPORTANT:** This application is strictly an academic demonstration developed for BCA college project evaluation. The predictions generated by machine learning models are for educational purposes and **must not** be used as medical advice or replace professional clinical decision-making by qualified healthcare professionals.
