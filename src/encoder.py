"""
Time Series Encoding Module
Converts time series to text representation with positional embeddings
Inspired by TimeSense's <ts> token approach
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from .preprocessing import TimeSeriesData, SegmentFeatures


@dataclass
class EncodedTimeSeries:
    """Container for encoded time series ready for LLM"""
    raw_encoding: str  # Full point-by-point encoding
    summary_encoding: str  # Compressed segment-based encoding
    metadata: Dict


class TimeSeriesEncoder:
    """
    Encodes time series into text format suitable for LLM consumption
    Following TimeSense approach with <ts> markers and positional information
    """

    def __init__(
        self,
        max_points: int = 500,
        downsample_method: str = "uniform"
    ):
        """
        Args:
            max_points: Maximum number of points to include in raw encoding
            downsample_method: "uniform", "peak_preserving", or "adaptive"
        """
        self.max_points = max_points
        self.downsample_method = downsample_method

    def encode_full_series(
        self,
        series: TimeSeriesData,
        include_timestamps: bool = True
    ) -> str:
        """
        Encode time series as point-by-point representation with positional info

        Format:
        <ts>
        metric_name index timestamp normalized_value
        ...
        </ts>
        """
        values = series.normalized_values if series.normalized_values is not None else series.values
        indices = series.indices if series.indices is not None else np.arange(len(values))

        # Downsample if needed
        if len(values) > self.max_points:
            indices, values, timestamps = self._downsample(
                indices, values, series.timestamps
            )
        else:
            timestamps = series.timestamps

        # Build encoding
        lines = ["<ts>"]
        for idx, val, ts in zip(indices, values, timestamps):
            if include_timestamps:
                ts_str = pd.Timestamp(ts).isoformat() if hasattr(ts, '__iter__') else str(ts)
                line = f"{series.name} {idx} {ts_str} {val:.4f}"
            else:
                line = f"{series.name} {idx} {val:.4f}"
            lines.append(line)
        lines.append("</ts>")

        return "\n".join(lines)

    def encode_summary(
        self,
        series: TimeSeriesData,
        segments: List[SegmentFeatures]
    ) -> str:
        """
        Encode time series as segment summaries

        Format:
        <ts_summary>
        metric_name:
          - segment 0: indices [start, end], mean X, slope Y, label "trend"
          ...
        </ts_summary>
        """
        lines = ["<ts_summary>", f"{series.name}:"]

        for i, seg in enumerate(segments):
            line = (
                f"  - segment {i}: indices [{seg.start_idx}, {seg.end_idx}], "
                f"mean {seg.mean:.2f}, slope {seg.slope:+.4f}, "
                f'label "{seg.trend_label}"'
            )
            lines.append(line)

        lines.append("</ts_summary>")
        return "\n".join(lines)

    def encode_multivariate(
        self,
        series_list: List[TimeSeriesData],
        use_summary: bool = False,
        segments_list: Optional[List[List[SegmentFeatures]]] = None
    ) -> str:
        """Encode multiple time series"""
        if use_summary and segments_list:
            encodings = [
                self.encode_summary(series, segments)
                for series, segments in zip(series_list, segments_list)
            ]
        else:
            encodings = [
                self.encode_full_series(series)
                for series in series_list
            ]

        return "\n\n".join(encodings)

    def _downsample(
        self,
        indices: np.ndarray,
        values: np.ndarray,
        timestamps: np.ndarray
    ) -> tuple:
        """Downsample time series to max_points"""
        if self.downsample_method == "uniform":
            # Uniform sampling
            step = len(values) // self.max_points
            sampled_idx = np.arange(0, len(values), step)[:self.max_points]

        elif self.downsample_method == "peak_preserving":
            # Keep peaks and troughs
            from scipy.signal import find_peaks

            # Find peaks and troughs
            peaks, _ = find_peaks(values)
            troughs, _ = find_peaks(-values)

            # Combine with uniform sampling
            important_points = np.concatenate([peaks, troughs])
            uniform_points = np.linspace(0, len(values)-1, self.max_points // 2, dtype=int)
            sampled_idx = np.unique(np.concatenate([important_points, uniform_points]))
            sampled_idx = np.sort(sampled_idx)[:self.max_points]

        else:  # adaptive
            # Keep points with high variance
            window = len(values) // self.max_points
            variances = []
            for i in range(0, len(values), window):
                end = min(i + window, len(values))
                variances.append((np.var(values[i:end]), i))

            # Sort by variance and keep top points
            variances.sort(reverse=True)
            sampled_idx = sorted([v[1] for v in variances[:self.max_points]])
            sampled_idx = np.array(sampled_idx)

        return (
            indices[sampled_idx],
            values[sampled_idx],
            timestamps[sampled_idx] if timestamps is not None else sampled_idx
        )


def create_task_specific_encoding(
    series_list: List[TimeSeriesData],
    task_type: str,
    segments_list: Optional[List[List[SegmentFeatures]]] = None,
    **kwargs
) -> str:
    """
    Create task-specific encodings based on EvalTS task types

    Args:
        series_list: List of time series
        task_type: One of "atomic", "molecular", "compositional"
        segments_list: Optional pre-computed segments
        **kwargs: Additional task-specific parameters

    Returns:
        Formatted encoding string
    """
    encoder = TimeSeriesEncoder()

    if task_type in ["atomic", "extreme", "spike", "trend", "change_point"]:
        # For atomic tasks, use full encoding for single series or first few hundred points
        if len(series_list) == 1 and len(series_list[0].values) <= 500:
            return encoder.encode_full_series(series_list[0])
        else:
            # Use summary for longer series
            if segments_list:
                return encoder.encode_summary(series_list[0], segments_list[0])
            return encoder.encode_full_series(series_list[0])

    elif task_type in ["molecular", "segment", "comparison", "relative"]:
        # For molecular tasks, use summary encoding
        if segments_list:
            return encoder.encode_multivariate(series_list, use_summary=True, segments_list=segments_list)
        return encoder.encode_multivariate(series_list, use_summary=False)

    else:  # compositional tasks
        # For complex tasks, combine both representations
        full = encoder.encode_multivariate(series_list[:3], use_summary=False)  # Limit to 3 series in full
        if segments_list:
            summary = encoder.encode_multivariate(series_list, use_summary=True, segments_list=segments_list)
            return f"{summary}\n\n# Detailed view (first 3 series):\n{full}"
        return full


# Helper to add pandas import
import pandas as pd
