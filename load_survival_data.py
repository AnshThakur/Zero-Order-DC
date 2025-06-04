import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_seer(csv_path="./data/SEER Breast Cancer Dataset .csv"):
    """
    Loads SEER dataset for models that require standardized inputs (e.g., CoxPH, DeepSurv).
    One-hot encodes categoricals and standardizes all features.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[col for col in df.columns if df[col].isna().all()], errors="ignore")
    df["event"] = df["Status"].map({"Dead": 1, "Alive": 0})
    df = df.rename(columns={"Survival Months": "duration"}).drop(columns=["Status"])
    df = df.dropna(subset=["duration", "event"])
    df["duration"] = df["duration"].astype(float)
    df["event"] = df["event"].astype(int)

    features = df.drop(columns=["duration", "event"])
    features_encoded = pd.get_dummies(features)
    valid_idx = features_encoded.dropna().index
    feature_names = features_encoded.columns.tolist()
    X = features_encoded.loc[valid_idx].reset_index(drop=True).values
    duration = df.loc[valid_idx, "duration"].reset_index(drop=True).to_numpy()
    event = df.loc[valid_idx, "event"].reset_index(drop=True).to_numpy()

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_idx, temp_idx in sss1.split(X, event):
        X_train, X_temp = X[train_idx], X[temp_idx]
        dur_train, dur_temp = duration[train_idx], duration[temp_idx]
        evt_train, evt_temp = event[train_idx], event[temp_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.6, random_state=42)
    for val_idx, test_idx in sss2.split(X_temp, evt_temp):
        X_val, X_test = X_temp[val_idx], X_temp[test_idx]
        dur_val, dur_test = dur_temp[val_idx], dur_temp[test_idx]
        evt_val, evt_test = evt_temp[val_idx], evt_temp[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, (dur_train, evt_train)), \
           (X_val, (dur_val, evt_val)), \
           (X_test, (dur_test, evt_test)),feature_names




def load_diabetes(filepath="./data/diabetes_anshul.csv"):
    
    df = pd.read_csv(filepath)
    # Drop target NA rows and HDL column
    df = df.dropna(subset=["inc_diabetes", "cox_time"])
    df = df.drop(columns=["BBC_HDL_Result"], errors="ignore")

    # Extract survival labels
    event = df["inc_diabetes"].astype(int).values
    duration = df["cox_time"].astype(float).values
    df = df.drop(columns=["inc_diabetes", "cox_time"])

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    # Impute missing values
    df[numeric_cols] = SimpleImputer(strategy="mean").fit_transform(df[numeric_cols])
    df[categorical_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[categorical_cols])

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df)

    # Final feature matrix
    X = df_encoded.values

    
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_idx, temp_idx in sss1.split(X, event):
        X_train, X_temp = X[train_idx], X[temp_idx]
        dur_train, dur_temp = duration[train_idx], duration[temp_idx]
        evt_train, evt_temp = event[train_idx], event[temp_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.6, random_state=42)
    for val_idx, test_idx in sss2.split(X_temp, evt_temp):
        X_val, X_test = X_temp[val_idx], X_temp[test_idx]
        dur_val, dur_test = dur_temp[val_idx], dur_temp[test_idx]
        evt_val, evt_test = evt_temp[val_idx], evt_temp[test_idx]

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, (dur_train, evt_train)), \
           (X_val, (dur_val, evt_val)), \
           (X_test, (dur_test, evt_test))
