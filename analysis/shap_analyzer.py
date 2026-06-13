"""
analysis/shap_analyzer.py
---------------------------
XGBoost 모델에 대한 SHAP 해석 모듈.

base_score 파싱 버그를 자동으로 수정하고,
TreeExplainer 실패 시 KernelExplainer로 폴백합니다.
"""

import json
import logging

import numpy as np
import shap
import xgboost as xgb

logger = logging.getLogger(__name__)


class ShapAnalyzer:
    """
    XGBoost 모델의 SHAP 값을 계산하고 해석 요약을 제공하는 클래스.

    Args:
        xgb_model: XGBoost 학습된 모델 (XGBClassifier)
    """

    def __init__(self, xgb_model) -> None:
        self.xgb_model = xgb_model
        self._explainer = None
        self._use_kernel = False
        self._use_native = False        # XGBoost 네이티브 pred_contribs 사용 여부
        self._booster = None
        self._native_expected_value = None

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _fix_base_score(self) -> xgb.Booster:
        """
        XGBoost booster의 base_score가 문자열 배열 형식("[0.5]")으로
        저장된 버그를 수정하고 booster를 반환합니다.
        """
        booster = self.xgb_model.get_booster()
        try:
            config = json.loads(booster.save_config())
            b_score = config["learner"]["learner_model_param"]["base_score"]
            if isinstance(b_score, str):
                fixed = float(b_score.strip("[]"))
                config["learner"]["learner_model_param"]["base_score"] = str(fixed)
                booster.load_config(json.dumps(config))
        except Exception as exc:
            logger.warning("base_score 수정 중 오류 (무시): %s", exc)
        return booster

    # ------------------------------------------------------------------
    # 공개 메서드
    # ------------------------------------------------------------------

    def build_explainer(self, X_input: np.ndarray) -> None:
        """
        SHAP 설명기를 준비합니다. 우선순위:

        1. **XGBoost 네이티브 pred_contribs** (TreeSHAP) — 권장.
           shap 라이브러리의 base_score 파싱 버그를 우회합니다.
           (XGBoost 3.x 는 base_score 를 배열 '[5E-1]' 로 저장하는데
            shap 0.49.x 의 TreeExplainer 는 이를 float 로 못 읽어 실패함.
            load_config 로도 교정 불가하므로 동일 알고리즘인 네이티브 경로 사용)
        2. shap.TreeExplainer (네이티브 실패 시)
        3. shap.KernelExplainer (최후 폴백)

        Args:
            X_input (np.ndarray): 설명에 사용할 입력 데이터 [n_samples, n_features]
        """
        # 1순위: XGBoost 네이티브 TreeSHAP
        try:
            booster = self.xgb_model.get_booster()
            dm = xgb.DMatrix(np.asarray(X_input, dtype=np.float32))
            contribs = booster.predict(dm, pred_contribs=True, validate_features=False)
            self._booster = booster
            self._native_expected_value = float(contribs[0, -1])  # bias = expected value
            self._use_native = True
            self._use_kernel = False
            logger.info("XGBoost 네이티브 pred_contribs 사용")
            return
        except Exception as exc:
            logger.warning("네이티브 pred_contribs 실패, shap TreeExplainer 시도: %s", exc)

        # 2순위: shap TreeExplainer
        try:
            booster = self._fix_base_score()
            self._explainer = shap.TreeExplainer(booster)
            self._use_kernel = False
            logger.info("TreeExplainer 초기화 성공")
        except Exception as exc:
            logger.warning("TreeExplainer 실패, KernelExplainer로 전환: %s", exc)
            self._explainer = shap.KernelExplainer(
                self.xgb_model.predict_proba, X_input
            )
            self._use_kernel = True

    def compute_shap_values(self, X_input: np.ndarray) -> np.ndarray:
        """
        주어진 입력에 대한 SHAP 값을 계산합니다.

        Args:
            X_input (np.ndarray): 입력 배열 [1, n_features]

        Returns:
            np.ndarray: SHAP 값 배열 [1, n_features]

        Raises:
            RuntimeError: build_explainer()를 먼저 호출하지 않은 경우
        """
        if self._use_native:
            dm = xgb.DMatrix(np.asarray(X_input, dtype=np.float32))
            contribs = self._booster.predict(
                dm, pred_contribs=True, validate_features=False
            )
            # 마지막 열(bias) 제거 → SHAP 값 [n_samples, n_features]
            return np.asarray(contribs)[:, :-1]
        if self._explainer is None:
            raise RuntimeError("build_explainer()를 먼저 호출하세요.")
        return self._explainer.shap_values(X_input)

    @property
    def expected_value(self) -> float:
        """SHAP baseline (expected value)을 반환합니다."""
        if self._use_native:
            return self._native_expected_value
        if self._explainer is None:
            raise RuntimeError("build_explainer()를 먼저 호출하세요.")
        return self._explainer.expected_value

    @property
    def is_kernel(self) -> bool:
        """현재 KernelExplainer를 사용 중이면 True를 반환합니다."""
        return self._use_kernel

    # ------------------------------------------------------------------
    # 분석 요약
    # ------------------------------------------------------------------

    def top_features(
        self,
        shap_values: np.ndarray,
        feature_names: list,
        n: int = 10,
    ) -> list:
        """
        SHAP 절대값 기준 상위 n개 피처 정보를 반환합니다.

        Returns:
            list[dict]: [{"feature": str, "shap_value": float, "direction": str}, ...]
        """
        indices = np.argsort(np.abs(shap_values[0]))[-n:][::-1]
        result = []
        for idx in indices:
            sv = float(shap_values[0][idx])
            result.append(
                {
                    "feature": feature_names[idx],
                    "shap_value": round(sv, 6),
                    "direction": "위험 증가" if sv > 0 else "위험 감소",
                }
            )
        return result

    def most_influential_feature(
        self,
        shap_values: np.ndarray,
        feature_names: list,
    ) -> str:
        """가장 영향력이 큰 피처명을 반환합니다."""
        idx = int(np.argmax(np.abs(shap_values[0])))
        return feature_names[idx]
