import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ================================
# Clinical Feature Processing
# ================================
features = pd.Series([
    'Blood_Test ALBUMIN', 'Blood_Test ALK.PHOSPHATASE', 'Blood_Test ALT', 'Blood_Test BASOPHILS',
    'Blood_Test BILIRUBIN', 'Blood_Test CREATININE', 'Blood_Test CRP', 'Blood_Test EOSINOPHILS',
    'Blood_Test HAEMATOCRIT', 'Blood_Test HAEMOGLOBIN', 'Blood_Test LYMPHOCYTES',
    'Blood_Test MEAN CELL VOL.', 'Blood_Test MONOCYTES', 'Blood_Test NEUTROPHILS',
    'Blood_Test PLATELETS', 'Blood_Test POTASSIUM', 'Blood_Test SODIUM', 'Blood_Test UREA',
    'Blood_Test WHITE CELLS', 'Blood_Test eGFR', 'Vital_Sign Respiratory Rate',
    'Vital_Sign Heart Rate', 'Vital_Sign Systolic Blood Pressure', 'Vital_Sign Temperature Tympanic',
    'Vital_Sign Oxygen Saturation', 'Vital_Sign Delivery device used', 'Vital_Sign Diastolic Blood Pressure'
])

def process_clinical_data(file_path):
    df = pd.read_csv(file_path)
    labels = df['Covid-19 Positive'].fillna(0)
    df = df[features]

    # Split dataset
    X_temp, X_test, y_temp, y_test = train_test_split(df, labels, test_size=0.2, random_state=25)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=25)

    # Median imputation (based on train only)
    median_values = X_train.median()
    X_train = X_train.fillna(median_values)
    X_val = X_val.fillna(median_values)
    X_test = X_test.fillna(median_values)

    # Standard scaling (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ================================
# Proteomics Data Processing
# ================================
def process_proteomics_data():
    data = pd.read_csv('./final_data/data/sorted_proteomics.csv')
    labels = pd.read_csv('./final_data/data/y_proteomics.csv')

    data = data.iloc[:, 1:1000]  # Select relevant features

    # Stratified split for label balance
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Mean imputation
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    # Standard scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (
        (X_train, y_train.to_numpy().flatten()),
        (X_val, y_val.to_numpy().flatten()),
        (X_test, y_test.to_numpy().flatten())
    )
