from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, func

from api.src.db import Base


class CreditPredictionApiLog(Base):
    __tablename__ = "credit_predictions_api_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(10), nullable=False)
    features = Column(JSON, nullable=False)
    prediction = Column(String(10), nullable=False)
    confidence = Column(Float, nullable=False)
    elapsed_ms = Column(Integer, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
