import os
import time
import warnings
from typing import Any, Awaitable, Callable, MutableMapping

import bentoml
import joblib
import numpy as np
import pandas as pd
from bentoml import Context
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from api.src.db import get_db
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


class DBSessionMiddleware:
    """
    ASGI 미들웨어로, 각 요청에 대한 데이터베이스 세션을 관리합니다.

    이 미들웨어는 요청이 들어올 때마다 새로운 DB 세션을 생성하여
    ASGI scope에 추가하고, 요청 처리가 완료되면 세션을 닫습니다.
    """

    def __init__(
        self,
        app: Callable[
            [MutableMapping[str, Any], Callable, Callable], Awaitable[None]
        ],
    ):
        """
        미들웨어를 초기화합니다.

        Args:
            app: 다음 처리 단계로 전달될 ASGI 애플리케이션입니다.
        """
        self.app = app

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[object]],
        send: Callable[[object], Awaitable[None]],
    ) -> None:
        """
        미들웨어를 호출하여 요청을 처리합니다.

        Args:
            scope: 요청에 대한 정보를 담고 있는 ASGI scope입니다.
            receive: 서버로부터 이벤트를 받기 위한 awaitable입니다.
            send: 클라이언트에게 이벤트를 보내기 위한 awaitable입니다.
        """
        session_generator = get_db()
        session: Session = next(session_generator)
        scope["db_session"] = session
        try:
            await self.app(scope, receive, send)
        finally:
            session.close()


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
@bentoml.asgi_app(DBSessionMiddleware)
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
        self.bento_model = bentoml.models.get("credit_score_classifier:latest")
        self.robust_scalers = joblib.load(
            os.path.join(encoder_path, "robust_scaler.joblib")
        )
        self.model = bentoml.catboost.load_model(self.bento_model)

    @bentoml.api
    def predict(self, data: Features, ctx: Context) -> Response:
        """
        입력된 고객 특징(features)을 기반으로 신용 등급을 예측합니다.

        Args:
            data (Features): 예측에 사용할 고객 특징 데이터입니다.
            ctx (Context): 요청 컨텍스트로, DB 세션에 접근하는 데 사용됩니다.

        Returns:
            Response: 예측된 신용 등급과 신뢰도 점수를 포함하는 응답입니다.
        """
        db: Session = ctx.request.scope["db_session"]
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
