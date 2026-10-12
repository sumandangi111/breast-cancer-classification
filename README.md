# Breast Cancer Classification Using Machine Learning

A machine learning project that classifies breast cancer tumors as **Benign** or **Malignant** using the Wisconsin Diagnostic Breast Cancer dataset and Logistic Regression.

## Project Overview

This project builds an end-to-end machine learning workflow for breast cancer classification. It includes data inspection, exploratory data analysis, preprocessing, model training, evaluation, prediction, and automated testing.

The model uses 30 numeric features computed from digitized images of fine needle aspirates (FNA) of breast masses.

> **Disclaimer:** This project is intended for educational purposes only. It is not a medical diagnostic tool and must not be used to make healthcare decisions.

## Features

- Dataset inspection and validation
- Exploratory data analysis (EDA)
- Feature scaling using `StandardScaler`
- Logistic Regression classification
- Stratified train-test split
- Evaluation using accuracy, precision, recall, F1-score, and confusion matrix
- Saved scikit-learn pipeline for predictions
- Automated tests using `pytest`
- Git version control and GitHub hosting

## Technology Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Model persistence:** Joblib
- **Testing:** Pytest
- **Version control:** Git and GitHub

## Dataset

The project uses the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

- **Records:** 569
- **Features:** 30 numeric features
- **Target:** Diagnosis
- **Classes:** Benign (`B`) and Malignant (`M`)
- **Missing values:** None in the inspected dataset

The diagnosis labels are mapped as follows:

| Label | Meaning |
|---|---|
| `B` / `0` | Benign |
| `M` / `1` | Malignant |

The features describe characteristics of cell nuclei, including radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

## Project Structure

```text
Disease-Classification/
├── data/
│   ├── wdbc.data
│   ├── wdbc.names
│   └── processed/
├── src/
│   ├── inspect_data.py
│   ├── eda.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
├── reports/
├── tests/
│   ├── test_data.py
│   └── test_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

Generated model files, processed data, and reports may be created when you run the scripts. Some generated artifacts are excluded from Git by `.gitignore`.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sumandangi111/breast-cancer-classification.git
cd Disease-Classification
```


### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, use an appropriate local execution-policy solution or run the virtual environment's Python executable directly.

## How to Run

Run these commands from the project root.

### Inspect the dataset

```powershell
python src/inspect_data.py
```

### Perform exploratory data analysis

```powershell
python src/eda.py
```

### Run preprocessing

```powershell
python src/preprocess.py
```

### Train the model

```powershell
python src/train.py
```

The training script saves the fitted machine learning pipeline to:

```text
models/breast_cancer_pipeline.joblib
```

### Evaluate the model

```powershell
python src/evaluate.py
```

### Run a prediction demo

```powershell
python src/predict.py
```

Run the training script before evaluation or prediction if the saved model is missing.

## Model Performance

Using the current stratified 80/20 train-test split with `random_state=42`, the Logistic Regression pipeline achieved the following results:

| Metric | Result |
|---|---:|
| Test records | 114 |
| Accuracy | 96.49% |
| Benign precision | 96% |
| Benign recall | 99% |
| Malignant precision | 97% |
| Malignant recall | 92.86% |
| Malignant F1-score | 95% |

### Confusion Matrix

| | Predicted Benign | Predicted Malignant |
|---|---:|---:|
| Actual Benign | 71 | 1 |
| Actual Malignant | 3 | 39 |

The test split contained three malignant cases classified as benign. In medical classification, false negatives are particularly important, so accuracy alone is not sufficient to assess model suitability.

These are results from one held-out test split, not evidence of clinical performance or independent validation.

## Automated Testing

Run the tests with:

```powershell
python -m pytest tests -v
```

The current test suite contains seven tests covering dataset structure, valid labels, numeric features, missing values, saved pipeline availability, predictions, and predicted probabilities.

## Future Improvements

- Add ROC and precision-recall curves
- Compare Logistic Regression with Random Forest and Support Vector Machine models
- Perform cross-validation and hyperparameter tuning
- Add a simple Flask web interface
- Improve input validation and model monitoring
- Add reproducible evaluation reports and screenshots

## Responsible Use

This project is a learning exercise in machine learning and binary classification. It has not been validated for clinical use. Predictions and probability estimates must not be interpreted as medical diagnoses or individual health-risk estimates.

## Author

**Suman Dangi**

- GitHub: https://github.com/sumandangi111
