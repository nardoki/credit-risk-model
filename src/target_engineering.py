import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# 1. Calculate RFM metrics
def calculate_rfm(df, snapshot_date=None):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).reset_index()

    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
    return rfm

# 2. Cluster customers into 3 groups using RFM
def cluster_rfm(rfm_df, n_clusters=3, random_state=42):
    features = ['Recency', 'Frequency', 'Monetary']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_df[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_df['Cluster'] = kmeans.fit_predict(X_scaled)
    return rfm_df

# 3. Assign "is_high_risk" label
def assign_high_risk(rfm_df):
    summary = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    high_risk_cluster = summary.sort_values(by=['Recency', 'Frequency', 'Monetary'],
                                            ascending=[False, True, True]).index[0]
    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
    return rfm_df

# 4. Merge label back to original data
def merge_labels(df, rfm_df):
    return df.merge(rfm_df[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

# 5. Main function
if __name__ == "__main__":
    RAW_DATA_PATH = '../data/raw/data.csv'
    PROCESSED_PATH = "data/processed/with_labels.csv"

    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)

    print("Calculating RFM metrics...")
    rfm = calculate_rfm(df)

    print("Clustering customers...")
    rfm = cluster_rfm(rfm)

    print("Assigning high-risk labels...")
    rfm = assign_high_risk(rfm)

    print("Merging labels to full dataset...")
    labeled_df = merge_labels(df, rfm)

    print(f"Saving to {PROCESSED_PATH}...")
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    labeled_df.to_csv(PROCESSED_PATH, index=False)

    print("✅ Task 4 completed. Labeled data ready.")
