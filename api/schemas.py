from pydantic import BaseModel
from typing import List

class StockFeatures(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: int
    ma5: float
    ma10: float
    ma20: float
    rsi: float
    macd: float
    macd_signal: float
    volume_ma5: float
    volume_ratio: float

class PredictionResponse(BaseModel):
    direction: int  # 1 = up, 0 = down
    probability: float