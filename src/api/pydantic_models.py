from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    total_amount: float
    avg_amount: float
    std_amount: float
    transaction_count: float
    total_value: float
    avg_value: float
    unique_products: float
    unique_channels: float
    fraud_count: float
    most_common_hour_0: float = 0
    most_common_hour_1: float = 0
    most_common_hour_2: float = 0
    most_common_hour_3: float = 0
    most_common_hour_4: float = 0
    most_common_hour_5: float = 0
    most_common_hour_6: float = 0
    most_common_hour_7: float = 0
    most_common_hour_8: float = 0
    most_common_hour_9: float = 0
    most_common_hour_10: float = 0
    most_common_hour_11: float = 0
    most_common_hour_12: float = 0
    most_common_hour_13: float = 0
    most_common_hour_14: float = 0
    most_common_hour_15: float = 0
    most_common_hour_16: float = 0
    most_common_hour_17: float = 0
    most_common_hour_18: float = 0
    most_common_hour_19: float = 0
    most_common_hour_20: float = 0
    most_common_hour_21: float = 0
    most_common_hour_22: float = 0
    most_common_hour_23: float = 0
    most_common_day_1: float = 0
    most_common_day_2: float = 0
    most_common_day_3: float = 0
    most_common_day_4: float = 0
    most_common_day_5: float = 0
    most_common_day_6: float = 0
    most_common_day_7: float = 0
    most_common_day_8: float = 0
    most_common_day_9: float = 0
    most_common_day_10: float = 0
    most_common_day_11: float = 0
    most_common_day_12: float = 0
    most_common_day_13: float = 0
    most_common_day_14: float = 0
    most_common_day_15: float = 0
    most_common_day_16: float = 0
    most_common_day_17: float = 0
    most_common_day_18: float = 0
    most_common_day_19: float = 0
    most_common_day_20: float = 0
    most_common_day_21: float = 0
    most_common_day_22: float = 0
    most_common_day_23: float = 0
    most_common_day_24: float = 0
    most_common_day_25: float = 0
    most_common_day_26: float = 0
    most_common_day_27: float = 0
    most_common_day_28: float = 0
    most_common_day_29: float = 0
    most_common_day_30: float = 0
    most_common_day_31: float = 0


class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: int
