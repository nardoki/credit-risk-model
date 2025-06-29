import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src import data_processing # import the helper function

def test_add_feature_sum():
    # Setup
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    # Act
    result = data_processing.add_feature_sum(df.copy(), 'A', 'B', 'C')
    # Assert
    assert 'C' in result.columns
    assert all(result['C'] == df['A'] + df['B'])

def test_preprocessing_pipeline_fit_transform():
    import numpy as np

    # Sample data with missing values
    X_sample = pd.DataFrame({
        'feat1': [1.0, 2.0, None, 4.0],
        'feat2': [4.0, None, 6.0, 8.0]
    })

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, X_sample.columns)
    ])

    # Fit transform should run without errors
    X_transformed = preprocessor.fit_transform(X_sample)

    # Output shape should be same rows and columns
    assert X_transformed.shape == X_sample.shape

    # Check no missing values after transform
    assert not pd.isnull(X_transformed).any()

