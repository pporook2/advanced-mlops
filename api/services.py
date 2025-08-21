import os
import time
import warnings

import bentoml
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from api.src.db import SessionLocal
from api.src.models import CreditPredictionApiLog
from api.src.schemas import Features, MetadataResponse, Response
from utils.dates import DateValues

warnings.filterwarnings(action="ignore")

# .env 파일 로드
load_dotenv()

MODEL_NAME = "credit_score_classification"
BASE_DT = DateValues.get_current_date()

artifacts_path = os.getenv("ARTIFACTS_PATH")
encoder_path = os.path.join(
    artifacts_path, "preprocessing", MODEL_NAME, BASE_DT, "encoders"
)


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class CreditScoreClassifier:
    """
    신용 점수 분류를 위한 BentoML 서비스입니다.

    이 서비스는 전처리된 데이터를 입력받아 신용 등급을 예측하고,
    예측 로그를 데이터베이스에 기록합니다.
    """

    def __init__(self) -> None:
        """
        서비스를 초기화하고 필요한 모델과 인코더를 로드합니다.
        """
        self.session_maker = None
        self.bento_model = bentoml.models.get("credit_score_classifier:latest")
        self.robust_scalers = joblib.load(
            os.path.join(encoder_path, "robust_scaler.joblib")
        )
        self.model = bentoml.catboost.load_model(self.bento_model)

    @bentoml.on_startup
    def initialize(self):
        self.session_maker = SessionLocal

    @bentoml.api
    def predict(self, data: Features) -> Response:
        """
        입력된 고객 특징(features)을 기반으로 신용 등급을 예측합니다.

        Args:
            data (Features): 예측에 사용할 고객 특징 데이터입니다.
            ctx (Context): 요청 컨텍스트로, DB 세션에 접근하는 데 사용됩니다.

        Returns:
            Response: 예측된 신용 등급과 신뢰도 점수를 포함하는 응답입니다.
        """
        start_time = time.time()
        df = pd.DataFrame([data.model_dump()])
        customer_id = df.pop("customer_id").item()

        for col, scaler in self.robust_scalers.items():
            df[col] = scaler.transform(df[[col]])

        prob = np.max(self.model.predict(df, prediction_type="Probability"))
        label = self.model.predict(df, prediction_type="Class").item()
        elapsed_ms = (time.time() - start_time) * 1000

        record = CreditPredictionApiLog(
            customer_id=customer_id,
            features=data.model_dump(),
            prediction=label,
            confidence=prob,
            elapsed_ms=elapsed_ms,
        )
        with self.session_maker() as db:
            with db.begin():
                db.add(record)

        return Response(customer_id=customer_id, predict=label, confidence=prob)

    @bentoml.api(route="/metadata", output_spec=MetadataResponse)
    def metadata(self) -> MetadataResponse:
        """현재 컨테이너에서 서빙 중인 모델의 메타데이터를 반환합니다."""
        return MetadataResponse(
            model_name=self.bento_model.tag.name,
            model_version=self.bento_model.tag.version,
            params=self.bento_model.info.metadata,
            creation_time=self.bento_model.info.creation_time,
        )
