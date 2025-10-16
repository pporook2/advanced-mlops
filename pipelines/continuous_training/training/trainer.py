import argparse
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple

import bentoml
import mlflow
import numpy.typing as npt
import pandas as pd
from catboost import CatBoostClassifier, Pool
from dotenv import load_dotenv
from mlflow.entities import Run
from mlflow.models.signature import infer_signature
from tqdm.auto import tqdm

from utils.dates import DateValues

# .env 파일에서 환경 변수 로드
load_dotenv()


@dataclass
class TrainingConfig:
    """학습 파이프라인에 필요한 모든 설정을 관리하는 데이터 클래스."""

    model_name: str
    base_dt: str
    artifacts_path: str = os.getenv("ARTIFACTS_PATH", "")
    target_name: str = "credit_score"
    
    drop_cols: List[str] = field(
        default_factory=lambda: ["base_dt", "id", "customer_id", "date"]
    )
    text_cols: List[str] = field(
        default_factory=lambda: ["type_of_loan", "payment_behaviour"]
    )
    categorical_cols: List[str] = field(
        default_factory=lambda: ["occupation", "credit_mix", "payment_of_min_amount"]
    )
    params_candidates: Dict[str, List] = field(
        default_factory=lambda: {
            "depth": [7, 8, 9],
            "rsm": [0.8, 0.9, 1.0],
            "l2_leaf_reg": [3, 5, 7],
        }
    )


class Trainer:
    """모델 학습 파이프라인을 관리하고 실행하는 클래스."""

    def __init__(self, config: TrainingConfig):
        """
        Trainer 클래스를 초기화합니다.

        Args:
            config (TrainingConfig): 학습 설정 객체
        """
        self._config = config
        self._preprocessing_path = os.path.join(
            self._config.artifacts_path,
            "preprocessing",
            self._config.model_name,
            self._config.base_dt,
        )
        self._model_path = os.path.join(
            self._config.artifacts_path,
            "models",
            self._config.model_name,
            self._config.base_dt,
        )
        self.is_trained = False

    def run(self) -> None:
        """전체 학습 파이프라인을 실행합니다."""
        self._setup_environment()
        train_pool, val_pool, x_train = self._prepare_data()
        experiment_id = self._tune_hyperparameters(
            train_pool, val_pool, x_train
        )
        best_run = self._get_best_run(experiment_id)
        self._save_model_to_bentoml(best_run)
        print(f"모델 학습 및 저장 완료: {self._config.model_name}")

    def _setup_environment(self) -> None:
        """학습 환경을 설정합니다. 기존 모델 경로가 존재하면 제거하고, 새로운 경로를 생성합니다."""
        if os.path.exists(self._model_path):
            shutil.rmtree(self._model_path)
        os.makedirs(self._model_path)
        print(f"학습 환경 설정 완료: {self._model_path}")

    def _prepare_data(self) -> Tuple[Pool, Pool, pd.DataFrame]:
        """
        데이터를 로드하고 CatBoost 학습에 사용할 Pool 객체를 생성합니다.

        Returns:
            Tuple[Pool, Pool, pd.DataFrame]: 학습용 Pool, 검증용 Pool, 학습 피처 데이터프레임
        """
        train_df = pd.read_csv(
            os.path.join(
                self._preprocessing_path, f"{self._config.model_name}_train.csv"
            )
        )
        val_df = pd.read_csv(
            os.path.join(
                self._preprocessing_path, f"{self._config.model_name}_val.csv"
            )
        )

        x_train = train_df.drop(
            [self._config.target_name] + self._config.drop_cols, axis=1
        )
        y_train = train_df[self._config.target_name].to_numpy()
        x_val = val_df.drop(
            [self._config.target_name] + self._config.drop_cols, axis=1
        )
        y_val = val_df[self._config.target_name].to_numpy()

        train_pool = self._create_pool(x_train, y_train)
        val_pool = self._create_pool(x_val, y_val)

        print("데이터 준비 완료.")
        return train_pool, val_pool, x_train

    def _create_pool(self, x: pd.DataFrame, y: npt.NDArray) -> Pool:
        """CatBoost Pool 객체를 생성합니다."""
        return Pool(
            data=x,
            label=y,
            cat_features=self._config.categorical_cols,
            text_features=self._config.text_cols,
        )

    def _tune_hyperparameters(
        self, train_pool: Pool, val_pool: Pool, x_train: pd.DataFrame
    ) -> str:
        """
        하이퍼파라미터 튜닝을 수행하고 최적의 모델을 찾습니다.

        Args:
            train_pool (Pool): 학습용 데이터 Pool
            val_pool (Pool): 검증용 데이터 Pool
            x_train (pd.DataFrame): 모델 서명 추론에 사용할 학습 피처

        Returns:
            str: 생성된 MLflow 실험 ID
        """
        # 파이프라인 실행 시마다 고유한 실험 이름을 생성하여 실행 결과를 명확히 구분합니다.
        # 이를 통해 동일한 모델에 대한 다른 학습 시도들을 쉽게 비교하고 추적할 수 있습니다.
        experiment_name = (
            f"training-{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
        )
        experiment = mlflow.set_experiment(experiment_name)

        param_set = self._get_params_set(self._config.params_candidates)
        catboost_static_params = self._get_static_params()

        print(f"하이퍼파라미터 튜닝 시작. 실험 이름: {experiment_name}")
        for i, params in enumerate(
            tqdm(param_set, desc="Hyperparameter Tuning")
        ):
            run_name = f"Run {i}"
            with mlflow.start_run(run_name=run_name):  # type: ignore
                cls = CatBoostClassifier(**params, **catboost_static_params)
                cls.fit(
                    train_pool,
                    eval_set=val_pool,
                    early_stopping_rounds=50,
                )

                self._log_to_mlflow(cls, params, x_train)

        self.is_trained = True
        print("하이퍼파라미터 튜닝 완료.")
        return experiment.experiment_id

    def _log_to_mlflow(
        self, model: CatBoostClassifier, params: dict, x_train: pd.DataFrame
    ) -> None:
        """학습된 모델의 정보와 결과를 MLflow에 로깅합니다."""
        mlflow.set_tag("estimator_name", model.__class__.__name__)  # type: ignore
        mlflow.log_params({key: model.get_params()[key] for key in params})  # type: ignore
        mlflow.log_param("iterations", model.best_iteration_)  # type: ignore
        mlflow.log_metrics(  # type: ignore
            self._parse_score_dict(model.get_best_score().get("validation"))
        )

        # `infer_signature`는 모델의 입력 및 출력 스키마를 자동으로 추론합니다.
        # 이는 모델을 사용할 때 데이터 유효성 검사를 활성화하고,
        # MLflow UI에서 모델의 입출력 형식을 명확하게 보여주는 중요한 역할을 합니다.
        mlflow.catboost.log_model(
            model,
            "CatBoostClassifier",
            signature=infer_signature(x_train, model.predict(x_train)),
        )

    def _get_best_run(self, experiment_id: str) -> Run:
        """
        지정된 실험에서 가장 성능이 좋은 실행(Run) 정보를 반환합니다.

        Args:
            experiment_id (str): MLflow 실험 ID

        Raises:
            AttributeError: 학습이 진행되지 않았거나 실험 정보를 찾을 수 없는 경우

        Returns:
            Run: 가장 성능이 좋은 모델의 실행 정보
        """
        if not self.is_trained:
            raise AttributeError(
                "학습이 진행되지 않았습니다. 실험 결과를 가져올 수 없습니다."
            )

        # 현재 비즈니스 요구사항에서는 모델의 '정확도(Accuracy)'를 가장 중요한 성능 지표로 판단합니다.
        # 따라서 Accuracy를 기준으로 최적의 모델을 선택합니다.
        best_run_df = mlflow.search_runs(  # type: ignore
            experiment_ids=[experiment_id],
            order_by=["metrics.Accuracy DESC"],
            max_results=1,
        )

        if len(best_run_df) == 0:
            raise AttributeError(
                f"실험 '{experiment_id}'에서 실행 정보를 찾을 수 없습니다."
            )

        # Ensure best_run_df is a DataFrame and access the first run_id correctly
        if isinstance(best_run_df, pd.DataFrame):
            best_run_id = best_run_df.iloc[0]["run_id"]
        else:
            # Fallback for unexpected types
            best_run_id = best_run_df[0].info.run_id
        print(f"최적 실행 찾음: {best_run_id}")
        return mlflow.get_run(best_run_id)  # type: ignore

    def _save_model_to_bentoml(self, best_run: Run) -> None:
        """최적 모델을 BentoML 형식으로 저장합니다."""
        run_id = best_run.info.run_id
        model_uri = f"runs:/{run_id}/CatBoostClassifier"
        model = mlflow.catboost.load_model(model_uri)

        # BentoML로 모델을 저장할 때 `signatures`를 정의하면,
        # API 서버에서 이 모델의 `predict` 함수를 어떻게 호출할지 명시할 수 있습니다.
        # `batchable=True`는 여러 입력을 한 번에 처리할 수 있도록 하여 처리량을 높입니다.
        # `metadata`에는 모델 학습에 사용된 파라미터를 저장하여, 나중에 모델을 추적하고 재현하는 데 사용합니다.
        bentoml.catboost.save_model(
            name=self._config.model_name,
            model=model,
            signatures={"predict": {"batchable": True, "batch_dim": 0}},
            metadata=best_run.data.params,
        )

        print(f"최적 모델을 BentoML에 저장 완료: {self._config.model_name}")

    @staticmethod
    def _get_params_set(params: dict) -> list:
        """하이퍼파라미터 후보 딕셔너리를 모든 조합의 리스트로 변환합니다."""
        # `itertools.product`를 사용하여 파라미터 후보들의 데카르트 곱(Cartesian product)을 생성합니다.
        # 이를 통해 모든 하이퍼파라미터 조합에 대한 탐색을 자동화할 수 있습니다.
        keys, values = zip(
            *[(k, v if isinstance(v, list) else [v]) for k, v in params.items()]
        )
        return [dict(zip(keys, v)) for v in product(*values)]

    @staticmethod
    def _parse_score_dict(score_dict: dict) -> dict:
        """MLflow 로깅을 위해 CatBoost 점수 딕셔너리의 키를 변환합니다."""
        # MLflow는 메트릭 키에 '=' 문자를 허용하지 않으므로,
        # CatBoost에서 생성된 'F1:class=X'와 같은 키를 'F1:class X'로 변경합니다.
        return {k.replace("=", " "): v for k, v in score_dict.items()}

    @staticmethod
    def _get_static_params() -> dict:
        """CatBoost 학습을 위한 정적 파라미터를 반환합니다."""
        return {
            "loss_function": "MultiClassOneVsAll",
            "learning_rate": 0.3,
            "iterations": 2000,
            "thread_count": -1,
            "random_seed": 42,
            "verbose": 50,
            "custom_metric": ["F1", "Accuracy"],
            # `text_processing`은 CatBoost가 텍스트 특성을 내부적으로 처리하는 방법을 정의합니다.
            # 별도의 TF-IDF나 임베딩 전처리 없이, 모델이 직접 텍스트의 의미를 학습하도록 돕습니다.
            # 여기서는 공백, 쉼표, 밑줄을 기준으로 단어를 분리(토큰화)하고,
            # Bi-gram과 Word 사전을 구축하여 텍스트 특성을 분석합니다.
            "text_processing": {
                "tokenizers": [
                    {
                        "tokenizer_id": "Space",
                        "separator_type": "ByDelimiter",
                        "delimiter": " ",
                    },
                    {
                        "tokenizer_id": "Comma",
                        "separator_type": "ByDelimiter",
                        "delimiter": ",",
                    },
                    {
                        "tokenizer_id": "Underscore",
                        "separator_type": "ByDelimiter",
                        "delimiter": "_",
                    },
                ],
                "dictionaries": [
                    {
                        "dictionary_id": "BiGram",
                        "occurence_lower_bound": 1,
                    },
                    {
                        "dictionary_id": "Word",
                        "occurence_lower_bound": 1,
                    },
                ],
            },
        }


def main():
    """스크립트의 메인 실행 함수."""
    parser = argparse.ArgumentParser(
        description="모델 학습 파이프라인을 위한 인자 파서"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="credit_score_classification",
        help="학습할 모델의 이름",
        dest="model_name"
    )
    parser.add_argument(
        "--base_dt",
        type=str,
        default=DateValues.get_current_date(),
        help="데이터의 기준 날짜 (YYYY-MM-DD 형식)",
        dest="base_dt"
    )
    args = parser.parse_args()

    config = TrainingConfig(model_name=args.model_name, base_dt=args.base_dt)
    trainer = Trainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
