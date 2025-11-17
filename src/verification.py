"""
Verification Module
External "Temporal Sense" validation - verifies LLM outputs against numerical computations
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .preprocessing import TimeSeriesData


@dataclass
class VerificationResult:
    """Result of verification check"""
    is_valid: bool
    confidence: float  # 0.0 to 1.0
    expected_value: Optional[any] = None
    actual_value: Optional[any] = None
    discrepancy: Optional[float] = None
    message: str = ""


class TimeSeriesVerifier:
    """
    Verifies LLM outputs against ground truth numerical computations
    Implements external "temporal sense" checking
    """

    def __init__(self, tolerance: float = 0.1):
        """
        Args:
            tolerance: Relative tolerance for numerical comparisons (10% default)
        """
        self.tolerance = tolerance

    def verify_extreme(
        self,
        series: TimeSeriesData,
        predicted_value: float,
        predicted_index: int,
        extremum_type: str = "maximum"
    ) -> VerificationResult:
        """
        Verify extreme value (min/max) prediction

        Args:
            series: Time series data
            predicted_value: Value predicted by LLM
            predicted_index: Index predicted by LLM
            extremum_type: "maximum" or "minimum"

        Returns:
            VerificationResult
        """
        values = series.values

        # Compute actual extreme
        if extremum_type == "maximum":
            actual_value = np.max(values)
            actual_index = int(np.argmax(values))
        else:
            actual_value = np.min(values)
            actual_index = int(np.argmin(values))

        # Check index match
        index_match = predicted_index == actual_index

        # Check value match (with tolerance)
        value_discrepancy = abs(predicted_value - actual_value)
        relative_error = value_discrepancy / (abs(actual_value) + 1e-8)
        value_match = relative_error <= self.tolerance

        # Calculate confidence
        if index_match and value_match:
            confidence = 1.0
            is_valid = True
            message = f"✓ Correct {extremum_type} identified"
        elif index_match:
            confidence = 0.7
            is_valid = True
            message = f"✓ Correct index, but value off by {relative_error:.1%}"
        elif value_match:
            confidence = 0.5
            is_valid = False
            message = f"✗ Correct value but wrong index (expected {actual_index}, got {predicted_index})"
        else:
            confidence = 0.0
            is_valid = False
            message = f"✗ Both index and value incorrect"

        return VerificationResult(
            is_valid=is_valid,
            confidence=confidence,
            expected_value=(actual_value, actual_index),
            actual_value=(predicted_value, predicted_index),
            discrepancy=value_discrepancy,
            message=message
        )

    def verify_spikes(
        self,
        series: TimeSeriesData,
        predicted_indices: List[int],
        threshold: float = 3.0
    ) -> VerificationResult:
        """
        Verify spike detection

        Args:
            series: Time series data
            predicted_indices: Spike indices predicted by LLM
            threshold: Z-score threshold for ground truth

        Returns:
            VerificationResult
        """
        from scipy import stats

        values = series.normalized_values if series.normalized_values is not None else series.values

        # Compute ground truth spikes using z-score
        z_scores = np.abs(stats.zscore(values, nan_policy='omit'))
        actual_indices = set(np.where(z_scores > threshold)[0].tolist())
        predicted_set = set(predicted_indices)

        # Calculate metrics
        true_positives = len(predicted_set & actual_indices)
        false_positives = len(predicted_set - actual_indices)
        false_negatives = len(actual_indices - predicted_set)

        # Precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        is_valid = f1_score >= 0.5
        confidence = f1_score

        if f1_score >= 0.8:
            message = f"✓ Excellent spike detection (F1: {f1_score:.2f})"
        elif f1_score >= 0.5:
            message = f"✓ Good spike detection (F1: {f1_score:.2f})"
        else:
            message = f"✗ Poor spike detection (F1: {f1_score:.2f})"

        return VerificationResult(
            is_valid=is_valid,
            confidence=confidence,
            expected_value=list(actual_indices),
            actual_value=predicted_indices,
            message=message
        )

    def verify_trend(
        self,
        series: TimeSeriesData,
        predicted_trend: str
    ) -> VerificationResult:
        """
        Verify trend classification

        Args:
            series: Time series data
            predicted_trend: Trend predicted by LLM ("increase", "decrease", "stable", "volatile")

        Returns:
            VerificationResult
        """
        values = series.normalized_values if series.normalized_values is not None else series.values

        # Compute slope using linear regression
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)

        # Compute volatility
        volatility = np.std(values) / (np.abs(np.mean(values)) + 1e-8)

        # Determine ground truth trend
        if volatility > 0.3:
            actual_trend = "volatile"
        elif abs(slope) < 0.001:
            actual_trend = "stable"
        elif slope > 0:
            actual_trend = "increase"
        else:
            actual_trend = "decrease"

        # Normalize predicted trend
        predicted_normalized = predicted_trend.lower().strip()
        if "increas" in predicted_normalized:
            predicted_normalized = "increase"
        elif "decreas" in predicted_normalized:
            predicted_normalized = "decrease"
        elif "stable" in predicted_normalized or "flat" in predicted_normalized:
            predicted_normalized = "stable"
        elif "volatile" in predicted_normalized:
            predicted_normalized = "volatile"

        is_valid = predicted_normalized == actual_trend
        confidence = 1.0 if is_valid else 0.3  # Partial credit if close

        # Give partial credit for similar trends
        if not is_valid:
            if (actual_trend == "increase" and predicted_normalized == "volatile") or \
               (actual_trend == "decrease" and predicted_normalized == "volatile"):
                confidence = 0.6
                is_valid = True  # Accept as valid with lower confidence

        message = f"{'✓' if is_valid else '✗'} Predicted: {predicted_normalized}, Actual: {actual_trend} (slope: {slope:.4f}, volatility: {volatility:.2f})"

        return VerificationResult(
            is_valid=is_valid,
            confidence=confidence,
            expected_value=actual_trend,
            actual_value=predicted_normalized,
            message=message
        )

    def verify_change_points(
        self,
        series: TimeSeriesData,
        predicted_points: List[int],
        window_tolerance: int = 10
    ) -> VerificationResult:
        """
        Verify change point detection

        Args:
            series: Time series data
            predicted_points: Change points predicted by LLM
            window_tolerance: Tolerance window for matching (±N points)

        Returns:
            VerificationResult
        """
        from .preprocessing import TimeSeriesPreprocessor

        # Compute ground truth change points
        preprocessor = TimeSeriesPreprocessor()
        actual_points = preprocessor.detect_change_points(series, threshold=1.0)

        # Match predicted to actual within tolerance window
        predicted_set = set(predicted_points)
        actual_set = set(actual_points)

        matches = 0
        for pred in predicted_points:
            for actual in actual_points:
                if abs(pred - actual) <= window_tolerance:
                    matches += 1
                    break

        # Calculate metrics
        precision = matches / len(predicted_points) if len(predicted_points) > 0 else 0
        recall = matches / len(actual_points) if len(actual_points) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        is_valid = f1_score >= 0.4  # Lower threshold for change points (harder task)
        confidence = f1_score

        message = f"{'✓' if is_valid else '✗'} Change points - Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1_score:.2f}"

        return VerificationResult(
            is_valid=is_valid,
            confidence=confidence,
            expected_value=actual_points,
            actual_value=predicted_points,
            message=message
        )

    def verify_value_at_index(
        self,
        series: TimeSeriesData,
        index: int,
        predicted_value: float
    ) -> VerificationResult:
        """Verify value retrieval at specific index"""
        if index < 0 or index >= len(series.values):
            return VerificationResult(
                is_valid=False,
                confidence=0.0,
                message=f"✗ Index {index} out of bounds"
            )

        actual_value = series.values[index]
        discrepancy = abs(predicted_value - actual_value)
        relative_error = discrepancy / (abs(actual_value) + 1e-8)

        is_valid = relative_error <= self.tolerance
        confidence = max(0.0, 1.0 - relative_error * 2)  # Linear decay

        message = f"{'✓' if is_valid else '✗'} Value at index {index}: predicted {predicted_value:.4f}, actual {actual_value:.4f} (error: {relative_error:.1%})"

        return VerificationResult(
            is_valid=is_valid,
            confidence=confidence,
            expected_value=actual_value,
            actual_value=predicted_value,
            discrepancy=discrepancy,
            message=message
        )

    def verify_anomalies(
        self,
        series: TimeSeriesData,
        predicted_intervals: List[Tuple[int, int]],
        anomaly_direction: str,
        threshold: float = 2.0
    ) -> VerificationResult:
        """
        Verify anomaly detection in intervals

        Args:
            series: Time series data
            predicted_intervals: List of (start, end) tuples
            anomaly_direction: "upward", "downward", or "both"
            threshold: Z-score threshold

        Returns:
            VerificationResult
        """
        from scipy import stats

        values = series.normalized_values if series.normalized_values is not None else series.values
        z_scores = stats.zscore(values, nan_policy='omit')

        # Find actual anomalous points
        if anomaly_direction == "upward":
            anomalous = z_scores > threshold
        elif anomaly_direction == "downward":
            anomalous = z_scores < -threshold
        else:  # both
            anomalous = np.abs(z_scores) > threshold

        # Check if predicted intervals contain anomalies
        interval_scores = []
        for start, end in predicted_intervals:
            if start < 0 or end >= len(values):
                continue
            interval_anomaly_ratio = np.sum(anomalous[start:end+1]) / (end - start + 1)
            interval_scores.append(interval_anomaly_ratio)

        if len(interval_scores) == 0:
            avg_score = 0.0
        else:
            avg_score = np.mean(interval_scores)

        is_valid = avg_score >= 0.3  # At least 30% of predicted interval should be anomalous
        confidence = min(1.0, avg_score * 2)

        message = f"{'✓' if is_valid else '✗'} Anomaly detection - {len(predicted_intervals)} intervals, avg anomaly ratio: {avg_score:.2f}"

        return VerificationResult(
            is_valid=is_valid,
            confidence=confidence,
            expected_value=np.where(anomalous)[0].tolist(),
            actual_value=predicted_intervals,
            message=message
        )
