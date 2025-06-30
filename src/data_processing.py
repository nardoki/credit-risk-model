import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xverse.transformer import WOE


# ----------------------
# Custom Transformers
# ----------------------
def add_feature_sum(df, col1, col2, new_col):
    df[new_col] = df[col1] + df[col2]
    return df

class TransactionAggregates(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        agg_df = df.groupby("CustomerId").agg({
            "Amount": ["sum", "mean", "std", "count"]
        })
        agg_df.columns = [
            "total_amount", "avg_amount", "std_amount", "transaction_count"
        ]
        agg_df.reset_index(inplace=True)
        return agg_df


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_column="TransactionStartTime"):
        self.datetime_column = datetime_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.datetime_column] = pd.to_datetime(df[self.datetime_column])
        df["transaction_hour"] = df[self.datetime_column].dt.hour
        df["transaction_day"] = df[self.datetime_column].dt.day
        df["transaction_month"] = df[self.datetime_column].dt.month
        df["transaction_year"] = df[self.datetime_column].dt.year
        return df


class SelectFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

# ----------------------
# Pipeline Function
# ----------------------

def build_feature_pipeline():
    numeric_features = ["total_amount", "avg_amount", "std_amount", "transaction_count"]
    date_features = ["transaction_hour", "transaction_day", "transaction_month", "transaction_year"]
    categorical_features = ["ProductCategory", "ChannelId", "PricingStrategy", "CurrencyCode"]

    # Preprocessing steps
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Column transformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features + date_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Full pipeline
    pipeline = Pipeline(steps=[
        ('aggregates', TransactionAggregates()),
        ('dates', DateFeatureExtractor()),
        ('select', SelectFeatures(
            columns=numeric_features + date_features + categorical_features)),
        ('preprocess', preprocessor)
    ])

    return pipeline
