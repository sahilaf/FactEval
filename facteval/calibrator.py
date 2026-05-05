"""
Calibrator – Transforms raw NLI scores into calibrated probabilities.

Uses isotonic regression models (fitted in Week 0) to produce trustworthy
confidence scores and calibration error estimates.

Falls back gracefully to raw scores if no calibrator file is available.
"""

import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class Calibrator:
    """Apply isotonic regression calibration to raw NLI probabilities."""

    def __init__(self, calibrator_path: str | Path | None = None):
        """
        Load a pre-fitted calibrator from a pickle file.

        Args:
            calibrator_path: Path to the pickle file containing a dict of
                {label_name: IsotonicRegression} objects.
                If None or file doesn't exist, falls back to raw scores.
        """
        self._calibrators: dict | None = None

        if calibrator_path is not None:
            path = Path(calibrator_path)
            if path.exists():
                with open(path, "rb") as f:
                    self._calibrators = pickle.load(f)
                logger.info(
                    "Loaded calibrator from %s (labels: %s)",
                    path, list(self._calibrators.keys()),
                )
            else:
                logger.warning("Calibrator file not found: %s. Using raw scores.", path)

    @property
    def is_calibrated(self) -> bool:
        """Whether a calibrator is loaded."""
        return self._calibrators is not None

    def calibrate(self, raw_scores: dict[str, float]) -> tuple[float, float]:
        """
        Calibrate raw NLI probabilities.

        Args:
            raw_scores: Dict mapping label names to raw probabilities
                        (e.g. {"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02}).

        Returns:
            (calibrated_confidence, calibration_error)
            - calibrated_confidence: The calibrated probability for the predicted label.
            - calibration_error: Absolute difference between raw and calibrated confidence.
        """
        if not raw_scores:
            return 0.0, 0.0

        # Find the predicted label (highest raw score)
        predicted_label = max(raw_scores, key=raw_scores.get)
        raw_confidence = raw_scores[predicted_label]

        if not self.is_calibrated:
            # Fallback: return raw confidence with an estimated error
            return raw_confidence, self._estimate_error(raw_confidence)

        # Apply isotonic regression for each label
        calibrated_scores = {}
        for label, raw_prob in raw_scores.items():
            if label in self._calibrators:
                cal_prob = float(self._calibrators[label].predict([[raw_prob]])[0])
                calibrated_scores[label] = max(0.0, min(1.0, cal_prob))
            else:
                calibrated_scores[label] = raw_prob

        calibrated_confidence = calibrated_scores.get(predicted_label, raw_confidence)
        calibration_error = abs(raw_confidence - calibrated_confidence)

        return round(calibrated_confidence, 4), round(calibration_error, 4)

    @staticmethod
    def _estimate_error(raw_confidence: float) -> float:
        """Rough error estimate when no calibrator is available."""
        # Higher confidence → lower estimated error, but never zero
        return round(max(0.02, (1.0 - raw_confidence) * 0.3), 4)
