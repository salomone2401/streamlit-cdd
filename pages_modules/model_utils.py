import os
import joblib
import pickle
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

from huggingface_hub import hf_hub_download
import pickle
import xgboost as xgb

REPO = "gilmagali4/spotify-popularity-models"   

def load_all_models():
    # Logistic Regression
    logreg_path = hf_hub_download(REPO, "logistic_regression_model.pkl")
    with open(logreg_path, "rb") as f:
        logreg = pickle.load(f)

    # Random Forest
    rf_path = hf_hub_download(REPO, "random_forest_model.pkl")
    with open(rf_path, "rb") as f:
        rf = pickle.load(f)

    # XGBoost
    xgb_path = hf_hub_download(REPO, "xgboost_model.pkl")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_path)

    # Scaler
    scaler_path = hf_hub_download(REPO, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Feature names
    feat_path = hf_hub_download(REPO, "feature_names.pkl")
    with open(feat_path, "rb") as f:
        feature_names = pickle.load(f)
        
    # Feature names
    X_train_path = hf_hub_download(REPO, "X_train.pkl")
    with open(X_train_path, "rb") as f:
        X_train = pickle.load(f)
  
    # Feature names
    y_train_path = hf_hub_download(REPO, "y_train.pkl")
    with open(y_train_path, "rb") as f:
        y_train = pickle.load(f)

    return logreg, rf, xgb_model, scaler, feature_names, X_train, y_train