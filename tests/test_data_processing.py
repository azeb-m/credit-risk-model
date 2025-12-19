import pandas as pd
from src.data_processing import generate_features


def test_generate_features_returns_expected_columns():
    """
    Test that feature engineering returns expected customer-level columns.
    """

    # Sample raw transaction data
    df = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "Amount": [100, 200, 150],
        "Value": [100, 200, 150],
        "TransactionStartTime": [
            "2024-01-01 10:00:00",
            "2024-01-02 12:00:00",
            "2024-01-03 09:00:00"
        ],
        "ProductId": [101, 102, 103],
        "ChannelId": [1, 1, 2],
        "FraudResult": [0, 1, 0]
    })

    processed = generate_features(df)

    expected_columns = [
        "CustomerId",
        "total_amount",
        "avg_amount",
        "std_amount",
        "transaction_count",
        "total_value",
        "avg_value",
        "unique_products",
        "unique_channels",
        "fraud_count"
    ]

    for col in expected_columns:
        assert col in processed.columns
def test_std_amount_is_zero_for_single_transaction_customer():
    """
    Test that std_amount is set to 0 for customers with only one transaction.
    """

    df = pd.DataFrame({
        "CustomerId": [1],
        "Amount": [100],
        "Value": [100],
        "TransactionStartTime": ["2024-01-01 08:00:00"],
        "ProductId": [101],
        "ChannelId": [1],
        "FraudResult": [0]
    })

    processed = generate_features(df)

    assert processed.loc[0, "std_amount"] == 0
