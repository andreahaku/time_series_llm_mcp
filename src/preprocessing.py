"""
Time Series Preprocessing Module
Inspired by TimeSense paper - handles normalization, segmentation, and feature extraction
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler


@dataclass
class TimeSeriesData:
    """Container for time series data with metadata"""
    name: str
    timestamps: np.ndarray
    values: np.ndarray
    normalized_values: Optional[np.ndarray] = None
    indices: Optional[np.ndarray] = None


@dataclass
class SegmentFeatures:
    """Features extracted from a time series segment"""
    start_idx: int
    end_idx: int
    mean: float
    std: float
    min_val: float
    max_val: float
    slope: float
    trend_label: str  # "increasing", "decreasing", "flat", "volatile"


class TimeSeriesPreprocessor:
    """
    Preprocesses time series data for LLM consumption
    Following TimeSense approach: normalize, add positions, extract features
    """

    def __init__(self, normalization: str = "zscore"):
        """
        Args:
            normalization: "zscore", "minmax", or "none"
        """
        self.normalization = normalization
        self.scalers = {}

    def normalize(self, series: TimeSeriesData) -> TimeSeriesData:
        """Normalize time series values"""
        if self.normalization == "none":
            series.normalized_values = series.values
            return series

        values = series.values.reshape(-1, 1)

        if self.normalization == "zscore":
            scaler = StandardScaler()
            normalized = scaler.fit_transform(values).flatten()
        elif self.normalization == "minmax":
            min_val, max_val = values.min(), values.max()
            if max_val - min_val > 0:
                normalized = (values - min_val) / (max_val - min_val)
                normalized = normalized.flatten()
            else:
                normalized = np.zeros_like(values).flatten()
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

        series.normalized_values = normalized
        series.indices = np.arange(len(series.values))
        self.scalers[series.name] = scaler if self.normalization == "zscore" else None

        return series

    def segment_series(
        self,
        series: TimeSeriesData,
        window_size: int = 50,
        overlap: int = 0
    ) -> List[SegmentFeatures]:
        """
        Segment time series into windows and extract features

        Args:
            series: TimeSeriesData object
            window_size: Size of each segment
            overlap: Overlap between segments

        Returns:
            List of SegmentFeatures
        """
        segments = []
        values = series.normalized_values if series.normalized_values is not None else series.values

        step = window_size - overlap
        for start_idx in range(0, len(values) - window_size + 1, step):
            end_idx = start_idx + window_size
            segment_values = values[start_idx:end_idx]

            # Compute features
            mean_val = np.mean(segment_values)
            std_val = np.std(segment_values)
            min_val = np.min(segment_values)
            max_val = np.max(segment_values)

            # Linear regression for slope
            x = np.arange(len(segment_values))
            slope, _ = np.polyfit(x, segment_values, 1)

            # Trend label
            if std_val > 0.3 * np.abs(mean_val):
                trend_label = "volatile"
            elif abs(slope) < 0.01:
                trend_label = "flat"
            elif slope > 0:
                trend_label = "increasing"
            else:
                trend_label = "decreasing"

            segments.append(SegmentFeatures(
                start_idx=start_idx,
                end_idx=end_idx,
                mean=float(mean_val),
                std=float(std_val),
                min_val=float(min_val),
                max_val=float(max_val),
                slope=float(slope),
                trend_label=trend_label
            ))

        return segments

    def detect_change_points(
        self,
        series: TimeSeriesData,
        threshold: float = 1.0
    ) -> List[int]:
        """
        Detect change points in time series using simple statistical method

        Args:
            series: TimeSeriesData object
            threshold: Threshold for change point detection (in standard deviations)

        Returns:
            List of change point indices
        """
        values = series.normalized_values if series.normalized_values is not None else series.values

        # Compute first derivative
        diff = np.diff(values)

        # Compute rolling statistics
        window = min(20, len(diff) // 4)
        rolling_mean = pd.Series(diff).rolling(window, center=True).mean().fillna(0)
        rolling_std = pd.Series(diff).rolling(window, center=True).std().fillna(1)

        # Detect points where derivative changes significantly
        z_scores = np.abs((diff - rolling_mean) / (rolling_std + 1e-8))
        change_points = np.where(z_scores > threshold)[0].tolist()

        # Filter out consecutive points
        if len(change_points) > 0:
            filtered = [change_points[0]]
            for cp in change_points[1:]:
                if cp - filtered[-1] > window // 2:
                    filtered.append(cp)
            return filtered

        return []

    def detect_spikes(
        self,
        series: TimeSeriesData,
        threshold: float = 3.0
    ) -> List[int]:
        """
        Detect spikes/anomalies using z-score method

        Args:
            series: TimeSeriesData object
            threshold: Z-score threshold for spike detection

        Returns:
            List of spike indices
        """
        values = series.normalized_values if series.normalized_values is not None else series.values

        # Use z-score method
        z_scores = np.abs(stats.zscore(values, nan_policy='omit'))
        spike_indices = np.where(z_scores > threshold)[0].tolist()

        return spike_indices

    def compute_fft_features(
        self,
        series: TimeSeriesData,
        top_k: int = 5
    ) -> Dict[str, Union[List[float], List[int]]]:
        """
        Compute FFT features for frequency domain analysis

        Args:
            series: TimeSeriesData object
            top_k: Number of top frequencies to return

        Returns:
            Dictionary with frequencies and power values
        """
        values = series.normalized_values if series.normalized_values is not None else series.values

        # Compute FFT
        fft_values = np.fft.fft(values)
        fft_freq = np.fft.fftfreq(len(values))

        # Get positive frequencies only
        positive_freq_idx = fft_freq > 0
        freqs = fft_freq[positive_freq_idx]
        power = np.abs(fft_values[positive_freq_idx]) ** 2

        # Get top k frequencies
        top_indices = np.argsort(power)[-top_k:][::-1]

        return {
            "top_frequencies": freqs[top_indices].tolist(),
            "top_powers": power[top_indices].tolist()
        }

    def extract_basic_stats(self, series: TimeSeriesData) -> Dict[str, float]:
        """Extract basic statistical features"""
        values = series.values

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "range": float(np.ptp(values)),
            "length": len(values)
        }
