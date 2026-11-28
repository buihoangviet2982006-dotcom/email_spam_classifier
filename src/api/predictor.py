"""Model loading, inference, and keyword explainability service."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.api.schemas import KeywordContribution, PredictRequest, PredictResponse
from src.preprocess.text_cleaner import combine_subject_and_body

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "best_model.pkl"
DEFAULT_METADATA_PATH = PROJECT_DIR / "models" / "model_metadata.json"


class SpamClassifierService:
    """Loads the selected best scikit-learn pipeline and serves predictions."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.model: Any = None
        self.metadata: dict[str, Any] = {}
        self._feature_names: np.ndarray | None = None
        self._linear_coefs: np.ndarray | None = None
        self._tfidf: Any = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("best_model_name", "Unknown Model"))

    def load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Best model not found at {self.model_path}. Run `python -m src.evaluate.compare_models` first."
            )
        with self.model_path.open("rb") as f:
            self.model = pickle.load(f)

        if self.metadata_path.is_file():
            with self.metadata_path.open("r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"best_model_name": type(self.model).__name__}

        self._init_explainer()

    def _init_explainer(self) -> None:
        """Extract TF-IDF vocabulary and linear weights (if available) for explainability."""
        if not hasattr(self.model, "named_steps"):
            return

        self._tfidf = self.model.named_steps.get("tfidf")
        clf = self.model.named_steps.get("clf") or self.model.named_steps.get("tree")
        if self._tfidf is None or clf is None:
            return

        try:
            self._feature_names = np.array(self._tfidf.get_feature_names_out())
        except Exception:
            self._feature_names = None
            return

        if hasattr(clf, "coef_"):
            self._linear_coefs = np.asarray(clf.coef_[0], dtype=float)
        elif hasattr(clf, "calibrated_classifiers_"):
            coef_list = []
            for cc in clf.calibrated_classifiers_:
                est = getattr(cc, "estimator", None)
                if est is not None and hasattr(est, "coef_"):
                    coef_list.append(np.asarray(est.coef_[0], dtype=float))
            if coef_list:
                self._linear_coefs = np.mean(coef_list, axis=0)

    def _explain(self, cleaned_text: str, top_k: int = 5) -> list[KeywordContribution]:
        if self._tfidf is None or self._feature_names is None or self._linear_coefs is None:
            return []

        vec = self._tfidf.transform([cleaned_text])
        if vec.nnz == 0:
            return []

        indices = vec.indices
        values = vec.data
        contributions = values * self._linear_coefs[indices]
        order = np.argsort(np.abs(contributions))[::-1][:top_k]

        keywords: list[KeywordContribution] = []
        for pos in order:
            weight = float(contributions[pos])
            if abs(weight) < 1e-6:
                continue
            token = str(self._feature_names[indices[pos]])
            keywords.append(
                KeywordContribution(
                    keyword=token,
                    weight=round(weight, 4),
                    direction="SPAM" if weight > 0 else "HAM",
                )
            )
        return keywords

    def predict_one(self, request: PredictRequest) -> PredictResponse:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        t0 = time.perf_counter()
        cleaned = combine_subject_and_body(request.subject, request.text)
        prob_spam = float(self.model.predict_proba([cleaned])[0][1])
        prob_ham = 1.0 - prob_spam

        is_spam = prob_spam >= request.threshold
        label = "SPAM" if is_spam else "HAM"
        label_id = 1 if is_spam else 0
        confidence = prob_spam if is_spam else prob_ham
        top_keywords = self._explain(cleaned, top_k=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return PredictResponse(
            label=label,
            label_id=label_id,
            spam_probability=round(prob_spam, 6),
            ham_probability=round(prob_ham, 6),
            confidence=round(confidence, 6),
            threshold=request.threshold,
            model_name=self.model_name,
            processing_time_ms=round(elapsed_ms, 2),
            top_keywords=top_keywords,
        )

    def predict_batch(self, requests: list[PredictRequest]) -> tuple[list[PredictResponse], float]:
        t0 = time.perf_counter()
        results = [self.predict_one(req) for req in requests]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return results, round(elapsed_ms, 2)
