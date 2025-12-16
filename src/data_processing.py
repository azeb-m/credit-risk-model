"""
data_processing.py

This module contains all data preprocessing and feature engineering logic.
It transforms raw transaction-level data into model-ready customer-level features.

Used by:
- train.py
- predict.py
- FastAPI inference
"""

# ======================================================
# Imports
# ======================================================

import os
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Optional (WoE)
try:
    from xverse.transformer import WOE
    XVERSE_AVAILABLE = True
except ImportError:
    XVERSE_AVAILABLE = False


# ======================================================
# 1. Date Feature Extraction
# ======================================================

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts time-based features from TransactionStartTime.
    """

    def __init__(self, datetime_col="TransactionStartTime"):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])

        df["TransactionHour"] = df[self.datetime_col].dt.hour
        df["TransactionDay"] = df[self.datetime_col].dt.day
        df["TransactionMonth"] = df[self.datetime_col].dt.month
        df["TransactionYear"] = df[self.datetime_col].dt.year

        return df


# ======================================================
# 2. Customer-Level Aggregation
# ======================================================

class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates transaction-level data into customer-level features.
    """

    def __init__(self, customer_id_col="CustomerId"):
        self.customer_id_col = customer_id_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        agg_df = df.groupby(self.customer_id_col).agg(
            # Monetary behavior
            total_amount=("Amount", "sum"),
            avg_amount=("Amount", "mean"),
            std_amount=("Amount", "std"),
            transaction_count=("Amount", "count"),

            total_value=("Value", "sum"),
            avg_value=("Value", "mean"),

            # Behavioral diversity
            unique_products=("ProductId", "nunique"),
            unique_channels=("ChannelId", "nunique"),

            # Temporal behavior
            most_common_hour=("TransactionHour", lambda x: x.mode()[0]),
            most_common_day=("TransactionDay", lambda x: x.mode()[0]),

            # Target-related (proxy support)
            fraud_count=("FraudResult", "sum")
        ).reset_index()

        # Replace NaN std (single transaction customers)
        agg_df["std_amount"] = agg_df["std_amount"].fillna(0)

        return agg_df
# ======================================================
# Categorical Feature Encoder
# ======================================================

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    One-Hot Encodes categorical features.
    """

    def __init__(self, categorical_cols=None):
        self.categorical_cols = categorical_cols

    def fit(self, X, y=None):
        self.encoder_ = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )

        self.encoder_.fit(X[self.categorical_cols])
        return self

    def transform(self, X):
        df = X.copy()

        encoded_array = self.encoder_.transform(df[self.categorical_cols])
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=self.encoder_.get_feature_names_out(self.categorical_cols),
            index=df.index
        )

        # Drop original categorical columns
        df = df.drop(columns=self.categorical_cols)

        # Concatenate encoded columns
        df = pd.concat([df, encoded_df], axis=1)

        return df


# ======================================================
# 3. Missing Value Handling
# ======================================================

class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Handles missing values for numerical features using median imputation.
    """

    def fit(self, X, y=None):
        self.num_cols_ = X.select_dtypes(include=["int64", "float64"]).columns
        self.imputer_ = SimpleImputer(strategy="median")
        self.imputer_.fit(X[self.num_cols_])
        return self

    def transform(self, X):
        df = X.copy()
        df[self.num_cols_] = self.imputer_.transform(df[self.num_cols_])
        return df


# ======================================================
# 4. Scaling Numerical Features
# ======================================================

class NumericalScaler(BaseEstimator, TransformerMixin):
    """
    Standardizes numerical features.
    """

    def fit(self, X, y=None):
        self.num_cols_ = X.select_dtypes(include=["int64", "float64"]).columns
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X[self.num_cols_])
        return self

    def transform(self, X):
        df = X.copy()
        df[self.num_cols_] = self.scaler_.transform(df[self.num_cols_])
        return df


# ======================================================
# 5. Optional WoE Transformation
# ======================================================

class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Applies Weight of Evidence transformation.
    Requires xverse.
    """

    def __init__(self, target_col="fraud_count"):
        self.target_col = target_col
        self.woe_ = None

    def fit(self, X, y=None):
        if not XVERSE_AVAILABLE:
            raise ImportError("xverse is not installed.")

        if self.target_col not in X.columns:
            raise KeyError(f"Target column '{self.target_col}' not found in input DataFrame.")

        # Separate features and target
        self.features_ = X.drop(columns=[self.target_col])
        self.target_ = X[self.target_col]

        # Initialize WOE without unsupported args
        self.woe_ = WOE()
        self.woe_.fit(self.features_, self.target_)

        return self

    def transform(self, X):
        if self.woe_ is None:
            raise RuntimeError("WoETransformer must be fitted before calling transform.")

        if self.target_col not in X.columns:
            raise KeyError(f"Target column '{self.target_col}' not found in input DataFrame.")

        df = X.copy()
        X_woe = self.woe_.transform(df.drop(columns=[self.target_col]))
        X_woe[self.target_col] = df[self.target_col].values

        return X_woe


# ======================================================
# 6. Pipeline Builder
# ======================================================

def build_feature_pipeline():

    steps = [
       ("date_features", DateFeatureExtractor()),
         ("customer_aggregation", CustomerAggregator()),
           ("missing_values", MissingValueHandler()), 
           ("categorical_encoding", CategoricalEncoder( categorical_cols=get_categorical_columns() )),
            # ("scaling", NumericalScaler()), ("woe", WoETransformer(target_col="FraudResult")), 
        ]

    return Pipeline(steps=steps)



# ======================================================
# 7. Public API Function
# ======================================================

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point for feature engineering.
    """

    pipeline = build_feature_pipeline()
    processed_df = pipeline.fit_transform(df)

    return processed_df

def get_categorical_columns():
    """
    Returns categorical columns to be encoded after aggregation.
    """
    return [
        "most_common_hour",
        "most_common_day"
    ]

# ======================================================
# 8. Script Execution
# ======================================================

if __name__ == "__main__":

    RAW_DATA_PATH = os.path.join("data", "raw", "data.csv")
    OUTPUT_PATH = os.path.join("data", "processed", "customer_features.csv")

    print(" Loading raw data...")
    raw_df = pd.read_csv(RAW_DATA_PATH)

    print(" Generating features...")
    processed_df = generate_features(raw_df)

    print(" Saving processed data...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    processed_df.to_csv(OUTPUT_PATH, index=False)

    print(f" Feature engineering completed: {OUTPUT_PATH}")
