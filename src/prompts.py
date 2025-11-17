"""
Prompt Templates Module
Task-specific prompts aligned with EvalTS benchmark categories
"""

from typing import Dict, List, Optional
from enum import Enum


class TaskType(Enum):
    """EvalTS task categories"""
    # Atomic Understanding
    EXTREME = "extreme"
    SPIKE = "spike"
    TREND = "trend"
    CHANGE_POINT = "change_point"
    INDEX_VALUE = "index_value"

    # Molecular Reasoning
    SEGMENT = "segment"
    COMPARISON = "comparison"
    RELATIVE = "relative"

    # Compositional Tasks
    DESCRIBE = "describe"
    ANOMALY_DETECTION = "anomaly_detection"
    ROOT_CAUSE_ANALYSIS = "rca"


class PromptBuilder:
    """Builds prompts for different time series analysis tasks"""

    SYSTEM_PROMPT = """You are an expert in time-series analysis with deep knowledge of temporal patterns, statistical analysis, and anomaly detection.

When analyzing time series data:
1. Pay close attention to the absolute positions (indices) and values
2. Look for patterns like trends, seasonality, change points, and anomalies
3. Provide specific, quantitative answers with indices and values
4. Think step-by-step through the temporal dynamics

The time series data is presented between <ts> and </ts> markers or as <ts_summary>.
Each line contains: channel_name index [timestamp] value"""

    def __init__(self):
        self.task_templates = {
            TaskType.EXTREME: self._extreme_template,
            TaskType.SPIKE: self._spike_template,
            TaskType.TREND: self._trend_template,
            TaskType.CHANGE_POINT: self._change_point_template,
            TaskType.SEGMENT: self._segment_template,
            TaskType.COMPARISON: self._comparison_template,
            TaskType.DESCRIBE: self._describe_template,
            TaskType.ANOMALY_DETECTION: self._anomaly_detection_template,
        }

    def build_prompt(
        self,
        task_type: TaskType,
        encoded_series: str,
        question: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Build a complete prompt for a given task

        Args:
            task_type: Type of analysis task
            encoded_series: Encoded time series data
            question: Custom question (optional)
            **kwargs: Additional task-specific parameters

        Returns:
            Complete prompt string
        """
        template_fn = self.task_templates.get(task_type)
        if not template_fn:
            raise ValueError(f"Unknown task type: {task_type}")

        task_prompt = template_fn(question=question, **kwargs)

        return f"""{self.SYSTEM_PROMPT}

Here is the time series data:

{encoded_series}

{task_prompt}

Please provide your analysis:"""

    def _extreme_template(self, question: Optional[str] = None, **kwargs) -> str:
        """Template for finding extrema (min/max)"""
        if question:
            return question

        extremum_type = kwargs.get("extremum_type", "maximum")
        interval = kwargs.get("interval")

        if interval:
            return f"""Task: Find the {extremum_type} value in the interval from index {interval[0]} to {interval[1]}.

Provide your answer in the following format:
- The {extremum_type} value is: [value]
- Located at index: [index]"""
        else:
            return f"""Task: Find the {extremum_type} value in this time series.

Provide your answer in the following format:
- The {extremum_type} value is: [value]
- Located at index: [index]"""

    def _spike_template(self, question: Optional[str] = None, **kwargs) -> str:
        """Template for spike detection"""
        if question:
            return question

        return """Task: Detect any spikes or sudden anomalous changes in this time series.

A spike is defined as a point that deviates significantly from its local trend.

Provide your answer in the following format:
- Spikes detected: Yes/No
- If yes, spike positions (indices): [list of indices]
- Brief description of each spike"""

    def _trend_template(self, question: Optional[str] = None, **kwargs) -> str:
        """Template for trend identification"""
        if question:
            return question

        return """Task: Identify the overall trend of this time series.

Analyze the general direction and pattern of the series.

Provide your answer in the following format:
- Overall trend: [increase/decrease/stable/volatile]
- Confidence: [high/medium/low]
- Supporting observations: [brief explanation]"""

    def _change_point_template(self, question: Optional[str] = None, **kwargs) -> str:
        """Template for change point detection"""
        if question:
            return question

        return """Task: Identify change points in this time series.

A change point is where the statistical properties (mean, variance, trend) of the series change significantly.

Provide your answer in the following format:
- Change points detected: [list of indices]
- For each change point, describe:
  - Index: [index]
  - Type of change: [mean shift/trend change/variance change]
  - Before/after characteristics"""

    def _segment_template(self, question: Optional[str] = None, **kwargs) -> str:
        """Template for segmentation"""
        if question:
            return question

        return """Task: Segment this time series into distinct phases with different trends.

Identify intervals where the series exhibits consistent behavior, separated by change points.

Provide your answer in the following format:
- Number of segments: [count]
- For each segment:
  - Interval: [start_index, end_index]
  - Trend: [description]
  - Key characteristics: [mean, volatility, etc.]"""

    def _comparison_template(self, question: Optional[str] = None, **kwargs) -> str:
        """Template for comparing multiple series"""
        if question:
            return question

        comparison_aspect = kwargs.get("aspect", "overall behavior")

        return f"""Task: Compare the given time series in terms of {comparison_aspect}.

Analyze the similarities and differences between the series.

Provide your answer in the following format:
- Key similarities: [list]
- Key differences: [list]
- Notable intervals of divergence: [list of intervals with indices]
- Conclusion: [which series exhibits what characteristics]"""

    def _describe_template(self, question: Optional[str] = None, **kwargs) -> str:
        """Template for comprehensive description"""
        if question:
            return question

        return """Task: Provide a comprehensive description of this time series.

Analyze all aspects including trends, patterns, anomalies, and statistical properties.

Provide your answer covering:
1. Overall trend and direction
2. Variability and volatility
3. Notable patterns (seasonality, cycles, etc.)
4. Anomalies or unusual points
5. Change points or regime shifts
6. Key statistical summaries
7. Potential insights or interpretations"""

    def _anomaly_detection_template(self, question: Optional[str] = None, **kwargs) -> str:
        """Template for anomaly detection"""
        if question:
            return question

        rules = kwargs.get("anomaly_rules", [])
        interval = kwargs.get("interval")

        rules_text = ""
        if rules:
            rules_text = "\n\nAnomaly rules:\n" + "\n".join(f"- {rule}" for rule in rules)

        interval_text = ""
        if interval:
            interval_text = f"\n\nFocus on the interval from index {interval[0]} to {interval[1]}."

        return f"""Task: Detect anomalies in the given time series.{rules_text}{interval_text}

For each series, identify:
1. Whether anomalies are present
2. The type of anomaly (upward/downward deviation, point/interval)
3. The location (indices) of anomalies
4. The severity (how much deviation from normal)

Provide your answer in the following format:
- Series [name]:
  - Anomalies detected: Yes/No
  - Anomaly intervals: [list of [start, end] pairs]
  - Anomaly type: [upward/downward]
  - Severity: [description]"""


class ResponseParser:
    """Parse and validate LLM responses"""

    @staticmethod
    def parse_extreme_response(response: str) -> Dict:
        """Parse response from extreme value detection"""
        import re

        result = {
            "value": None,
            "index": None,
            "raw_response": response
        }

        # Try to extract value
        value_match = re.search(r'value.*?:?\s*([-+]?\d*\.?\d+)', response, re.IGNORECASE)
        if value_match:
            result["value"] = float(value_match.group(1))

        # Try to extract index
        index_match = re.search(r'(?:index|position|at).*?:?\s*(\d+)', response, re.IGNORECASE)
        if index_match:
            result["index"] = int(index_match.group(1))

        return result

    @staticmethod
    def parse_spike_response(response: str) -> Dict:
        """Parse response from spike detection"""
        import re

        result = {
            "has_spikes": False,
            "spike_indices": [],
            "raw_response": response
        }

        # Check if spikes detected
        if re.search(r'(yes|spikes?\s+detected)', response, re.IGNORECASE):
            result["has_spikes"] = True

            # Extract indices - look for lists or individual numbers
            indices = re.findall(r'\b(\d+)\b', response)
            result["spike_indices"] = [int(idx) for idx in indices]

        return result

    @staticmethod
    def parse_trend_response(response: str) -> Dict:
        """Parse response from trend identification"""
        import re

        result = {
            "trend": None,
            "confidence": None,
            "raw_response": response
        }

        # Extract trend
        trend_match = re.search(
            r'trend.*?:?\s*(increas(?:ing|e)|decreas(?:ing|e)|stable|volatile|flat)',
            response,
            re.IGNORECASE
        )
        if trend_match:
            trend_text = trend_match.group(1).lower()
            if 'increas' in trend_text:
                result["trend"] = "increase"
            elif 'decreas' in trend_text:
                result["trend"] = "decrease"
            elif 'stable' in trend_text or 'flat' in trend_text:
                result["trend"] = "stable"
            elif 'volatile' in trend_text:
                result["trend"] = "volatile"

        # Extract confidence
        conf_match = re.search(r'confidence.*?:?\s*(high|medium|low)', response, re.IGNORECASE)
        if conf_match:
            result["confidence"] = conf_match.group(1).lower()

        return result

    @staticmethod
    def parse_change_points_response(response: str) -> Dict:
        """Parse response from change point detection"""
        import re

        result = {
            "change_points": [],
            "raw_response": response
        }

        # Extract indices from lists or comma-separated values
        # Look for patterns like [1, 5, 10] or "indices: 1, 5, 10"
        list_match = re.search(r'\[([0-9,\s]+)\]', response)
        if list_match:
            indices_str = list_match.group(1)
            result["change_points"] = [int(x.strip()) for x in indices_str.split(',') if x.strip()]
        else:
            # Fallback: find all numbers in context of change points
            indices = re.findall(r'(?:index|point).*?:?\s*(\d+)', response, re.IGNORECASE)
            result["change_points"] = [int(idx) for idx in indices]

        return result
