import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ======================================================
# Step 1: Calculate RFM Metrics
# ======================================================

def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Recency, Frequency, and Monetary values per customer.
    """

    # Ensure datetime format
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    # Define snapshot date (one day after last transaction)
    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerId")
        .agg(
            recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
            frequency=("TransactionId", "count"),
            monetary=("Amount", "sum"),
        )
        .reset_index()
    )

    return rfm


# ======================================================
# Step 2: Scale RFM Features
# ======================================================

def scale_rfm(rfm_df: pd.DataFrame):
    """
    Scale RFM values for clustering.
    """

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm_df[["recency", "frequency", "monetary"]]
    )

    return rfm_scaled


# ======================================================
# Step 3: Cluster Customers using K-Means
# ======================================================

def cluster_customers(rfm_scaled, n_clusters=3):
    """
    Apply K-Means clustering.
    """

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(rfm_scaled)

    return clusters


# ======================================================
# Step 4: Identify High-Risk Cluster
# ======================================================

def identify_high_risk_cluster(rfm_df: pd.DataFrame) -> int:
    """
    Identify the least engaged (high-risk) cluster.
    """

    cluster_summary = (
        rfm_df.groupby("cluster")
        .agg(
            recency=("recency", "mean"),
            frequency=("frequency", "mean"),
            monetary=("monetary", "mean"),
        )
    )

    # High risk: high recency, low frequency, low monetary
    high_risk_cluster = (
        cluster_summary
        .sort_values(
            by=["recency", "frequency", "monetary"],
            ascending=[False, True, True]
        )
        .index[0]
    )

    return high_risk_cluster


# ======================================================
# Step 5: Create Proxy Target Variable
# ======================================================

def create_proxy_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate binary proxy target variable (is_high_risk).
    """

    rfm = calculate_rfm(df)

    rfm_scaled = scale_rfm(rfm)

    rfm["cluster"] = cluster_customers(rfm_scaled)

    high_risk_cluster = identify_high_risk_cluster(rfm)

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    return rfm[["CustomerId", "is_high_risk"]]


# ======================================================
# Step 6: Integrate Target Variable
# ======================================================

def merge_target(processed_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge proxy target back into processed dataset.
    """

    final_df = processed_df.merge(
        target_df,
        on="CustomerId",
        how="left"
    )

    return final_df


# ======================================================
# Script Execution
# ======================================================

if __name__ == "__main__":

    print("Loading raw data...")
    raw_df = pd.read_csv("data/raw/data.csv")
    print("Loading processed data...")
    processed_df = pd.read_csv("data/processed/customer_features.csv")

    print("Creating proxy target variable...")
    proxy_target = create_proxy_target(raw_df)

    print("Merging target with processed dataset...")
    final_df = merge_target(processed_df, proxy_target)

    print("Saving final dataset...")
    final_df.to_csv(
        "data/processed/model_input.csv",
        index=False
    )

    print("Task 4 completed successfully.")
