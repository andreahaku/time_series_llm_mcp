"""
Basic usage examples for TimeSense MCP server
"""

import numpy as np
from datetime import datetime, timedelta


def generate_sample_data():
    """Generate sample time series data for testing"""

    # Example 1: Trending series with spike
    dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(100)]
    values = np.linspace(10, 30, 100) + np.random.normal(0, 2, 100)
    values[50] = 50  # Add a spike

    series_1 = {
        "name": "metric_a",
        "timestamps": [d.isoformat() for d in dates],
        "values": values.tolist()
    }

    # Example 2: Series with change point
    values_2 = np.concatenate([
        np.random.normal(10, 1, 40),  # Stable around 10
        np.random.normal(20, 1, 60),  # Jump to 20
    ])

    series_2 = {
        "name": "metric_b",
        "timestamps": [d.isoformat() for d in dates],
        "values": values_2.tolist()
    }

    # Example 3: Cyclical pattern
    t = np.arange(100)
    values_3 = 15 + 5 * np.sin(2 * np.pi * t / 20) + np.random.normal(0, 0.5, 100)

    series_3 = {
        "name": "metric_c",
        "timestamps": [d.isoformat() for d in dates],
        "values": values_3.tolist()
    }

    return series_1, series_2, series_3


# Example MCP tool calls (these would be called via MCP protocol)
EXAMPLE_REQUESTS = {
    "analyze_extreme": {
        "series": [generate_sample_data()[0]],
        "question": "What is the maximum value in this time series and where does it occur?",
        "task_type": "extreme",
        "verify": True
    },

    "analyze_trend": {
        "series": [generate_sample_data()[0]],
        "question": "What is the overall trend of this time series?",
        "task_type": "trend",
        "verify": True
    },

    "detect_spike": {
        "series": [generate_sample_data()[0]],
        "question": "Are there any spikes or anomalies in this time series?",
        "task_type": "spike",
        "verify": True
    },

    "detect_change_point": {
        "series": [generate_sample_data()[1]],
        "question": "Identify any change points where the series behavior changes significantly.",
        "task_type": "change_point",
        "verify": True
    },

    "segment_series": {
        "series": generate_sample_data()[2],
        "window_size": 20
    },

    "compare_series": {
        "series_a": generate_sample_data()[0],
        "series_b": generate_sample_data()[1],
        "aspect": "trends and volatility"
    },

    "detect_anomalies": {
        "series": [generate_sample_data()[0], generate_sample_data()[1]],
        "interval": [40, 60],
        "anomaly_rules": [
            "Series 1: values above 45 are anomalous",
            "Series 2: sudden jumps > 5 are anomalous"
        ]
    },

    "comprehensive_analysis": {
        "series": [generate_sample_data()[0]],
        "question": "Provide a comprehensive analysis of this time series including trends, anomalies, and key patterns.",
        "task_type": "describe",
        "verify": False
    }
}


def print_example_requests():
    """Print example requests for documentation"""
    import json

    print("# Example MCP Tool Requests\n")

    for name, request in EXAMPLE_REQUESTS.items():
        print(f"## {name}\n")
        print("```json")
        print(json.dumps(request, indent=2))
        print("```\n")


if __name__ == "__main__":
    print_example_requests()
