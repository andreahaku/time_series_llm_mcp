#!/usr/bin/env python3
"""
TimeSense MCP Server
Provides time series analysis tools for LLMs via Model Context Protocol
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from pydantic import BaseModel, Field

from .preprocessing import TimeSeriesData, TimeSeriesPreprocessor
from .encoder import TimeSeriesEncoder, create_task_specific_encoding
from .prompts import PromptBuilder, TaskType, ResponseParser
from .verification import TimeSeriesVerifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for tool inputs
class TimeSeriesInput(BaseModel):
    """Input schema for time series data"""
    name: str = Field(description="Name/identifier for this time series")
    timestamps: List[Any] = Field(description="Timestamps (can be strings, numbers, or datetime)")
    values: List[float] = Field(description="Numeric values")


class AnalyzeTimeSeriesInput(BaseModel):
    """Input for analyze_time_series tool"""
    series: List[TimeSeriesInput] = Field(description="List of time series to analyze")
    question: str = Field(description="Analysis question or task description")
    task_type: Optional[str] = Field(
        default="auto",
        description="Task type: auto, extreme, spike, trend, change_point, segment, comparison, describe, anomaly_detection"
    )
    verify: bool = Field(default=True, description="Whether to verify LLM output against numerical computation")


class DescribeSegmentsInput(BaseModel):
    """Input for describe_segments tool"""
    series: TimeSeriesInput = Field(description="Time series to segment")
    window_size: int = Field(default=50, description="Segment window size")


class DetectAnomaliesInput(BaseModel):
    """Input for detect_anomalies tool"""
    series: List[TimeSeriesInput] = Field(description="Time series to analyze for anomalies")
    interval: Optional[List[int]] = Field(default=None, description="[start, end] interval to focus on")
    anomaly_rules: Optional[List[str]] = Field(default=None, description="Custom anomaly rules")


class CompareSeriesInput(BaseModel):
    """Input for compare_series tool"""
    series_a: TimeSeriesInput = Field(description="First time series")
    series_b: TimeSeriesInput = Field(description="Second time series")
    aspect: str = Field(default="overall behavior", description="What aspect to compare")


# MCP Server
class TimeSeriesMCPServer:
    """MCP Server for time series analysis"""

    def __init__(self):
        self.server = Server("timesense-mcp")
        self.preprocessor = TimeSeriesPreprocessor()
        self.encoder = TimeSeriesEncoder()
        self.prompt_builder = PromptBuilder()
        self.verifier = TimeSeriesVerifier()
        self.parser = ResponseParser()

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register MCP tools"""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available time series analysis tools"""
            return [
                Tool(
                    name="analyze_time_series",
                    description="""Analyze time series data using LLM reasoning.

Supports multiple task types:
- extreme: Find maximum/minimum values
- spike: Detect spikes and anomalies
- trend: Identify overall trends (increase/decrease/stable)
- change_point: Detect change points and regime shifts
- segment: Segment series into distinct phases
- comparison: Compare multiple series
- describe: Comprehensive description
- anomaly_detection: Detect and explain anomalies

Returns LLM analysis with optional numerical verification.""",
                    inputSchema=AnalyzeTimeSeriesInput.model_json_schema()
                ),
                Tool(
                    name="describe_segments",
                    description="""Segment a time series into distinct phases and describe each segment.

Returns detailed characteristics of each segment including trend, mean, volatility, etc.""",
                    inputSchema=DescribeSegmentsInput.model_json_schema()
                ),
                Tool(
                    name="detect_anomalies",
                    description="""Detect anomalies in time series with optional custom rules.

Can focus on specific intervals and apply domain-specific anomaly definitions.""",
                    inputSchema=DetectAnomaliesInput.model_json_schema()
                ),
                Tool(
                    name="compare_series",
                    description="""Compare two time series and identify differences.

Analyzes similarities, differences, and intervals of divergence.""",
                    inputSchema=CompareSeriesInput.model_json_schema()
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> list[TextContent]:
            """Handle tool calls"""
            try:
                if name == "analyze_time_series":
                    return await self._analyze_time_series(arguments)
                elif name == "describe_segments":
                    return await self._describe_segments(arguments)
                elif name == "detect_anomalies":
                    return await self._detect_anomalies(arguments)
                elif name == "compare_series":
                    return await self._compare_series(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error in {name}: {str(e)}", exc_info=True)
                return [TextContent(
                    type="text",
                    text=f"Error: {str(e)}"
                )]

    def _parse_series_input(self, series_input: TimeSeriesInput) -> TimeSeriesData:
        """Convert input to TimeSeriesData"""
        timestamps = np.array(series_input.timestamps)
        values = np.array(series_input.values)

        return TimeSeriesData(
            name=series_input.name,
            timestamps=timestamps,
            values=values
        )

    async def _analyze_time_series(self, arguments: Dict) -> list[TextContent]:
        """Main time series analysis tool"""
        args = AnalyzeTimeSeriesInput(**arguments)

        # Parse input series
        series_list = [self._parse_series_input(s) for s in args.series]

        # Preprocess
        for series in series_list:
            self.preprocessor.normalize(series)

        # Determine task type
        task_type_str = args.task_type
        if task_type_str == "auto":
            task_type_str = self._infer_task_type(args.question)

        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            task_type = TaskType.DESCRIBE

        # Extract features based on task type
        segments_list = None
        if task_type in [TaskType.SEGMENT, TaskType.DESCRIBE]:
            segments_list = [
                self.preprocessor.segment_series(series)
                for series in series_list
            ]

        # Encode time series
        encoded = create_task_specific_encoding(
            series_list,
            task_type_str,
            segments_list
        )

        # Build prompt
        prompt = self.prompt_builder.build_prompt(
            task_type=task_type,
            encoded_series=encoded,
            question=args.question
        )

        # Get LLM response (simulated for now - in production, call Anthropic API)
        llm_response = await self._call_llm(prompt)

        # Parse response
        parsed_response = self._parse_response(llm_response, task_type)

        # Verify if requested
        verification_result = None
        if args.verify and len(series_list) > 0:
            verification_result = self._verify_response(
                series_list[0],
                parsed_response,
                task_type
            )

        # Format output
        output = self._format_analysis_output(
            llm_response,
            parsed_response,
            verification_result,
            task_type
        )

        return [TextContent(type="text", text=output)]

    async def _describe_segments(self, arguments: Dict) -> list[TextContent]:
        """Segment and describe time series"""
        args = DescribeSegmentsInput(**arguments)

        # Parse and preprocess
        series = self._parse_series_input(args.series)
        self.preprocessor.normalize(series)

        # Segment
        segments = self.preprocessor.segment_series(
            series,
            window_size=args.window_size
        )

        # Extract statistics
        stats = self.preprocessor.extract_basic_stats(series)

        # Format output
        output = f"# Time Series Segmentation: {series.name}\n\n"
        output += f"## Overall Statistics\n"
        output += f"- Length: {stats['length']} points\n"
        output += f"- Mean: {stats['mean']:.4f}\n"
        output += f"- Std: {stats['std']:.4f}\n"
        output += f"- Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n\n"

        output += f"## Segments ({len(segments)} total)\n\n"
        for i, seg in enumerate(segments):
            output += f"### Segment {i+1}\n"
            output += f"- Interval: indices [{seg.start_idx}, {seg.end_idx}]\n"
            output += f"- Trend: **{seg.trend_label}**\n"
            output += f"- Mean: {seg.mean:.4f}\n"
            output += f"- Std: {seg.std:.4f}\n"
            output += f"- Slope: {seg.slope:+.6f}\n"
            output += f"- Range: [{seg.min_val:.4f}, {seg.max_val:.4f}]\n\n"

        return [TextContent(type="text", text=output)]

    async def _detect_anomalies(self, arguments: Dict) -> list[TextContent]:
        """Detect anomalies in time series"""
        args = DetectAnomaliesInput(**arguments)

        # Parse and preprocess
        series_list = [self._parse_series_input(s) for s in args.series]
        for series in series_list:
            self.preprocessor.normalize(series)

        # Detect spikes for each series
        anomaly_results = []
        for series in series_list:
            spikes = self.preprocessor.detect_spikes(series, threshold=3.0)
            change_points = self.preprocessor.detect_change_points(series)

            anomaly_results.append({
                "series_name": series.name,
                "spike_indices": spikes,
                "change_points": change_points,
                "num_spikes": len(spikes),
                "num_change_points": len(change_points)
            })

        # Build LLM prompt for interpretation
        encoded = self.encoder.encode_multivariate(series_list)
        prompt = self.prompt_builder.build_prompt(
            task_type=TaskType.ANOMALY_DETECTION,
            encoded_series=encoded,
            question=None,
            anomaly_rules=args.anomaly_rules or [],
            interval=args.interval
        )

        llm_response = await self._call_llm(prompt)

        # Format output
        output = "# Anomaly Detection Results\n\n"
        output += "## Statistical Detection\n\n"
        for result in anomaly_results:
            output += f"### {result['series_name']}\n"
            output += f"- Spikes detected: {result['num_spikes']}\n"
            if result['num_spikes'] > 0:
                output += f"  - Indices: {result['spike_indices'][:10]}"  # Show first 10
                if result['num_spikes'] > 10:
                    output += f" ... ({result['num_spikes']} total)"
                output += "\n"
            output += f"- Change points: {result['num_change_points']}\n"
            if result['num_change_points'] > 0:
                output += f"  - Indices: {result['change_points']}\n"
            output += "\n"

        output += "## LLM Analysis\n\n"
        output += llm_response

        return [TextContent(type="text", text=output)]

    async def _compare_series(self, arguments: Dict) -> list[TextContent]:
        """Compare two time series"""
        args = CompareSeriesInput(**arguments)

        # Parse and preprocess
        series_a = self._parse_series_input(args.series_a)
        series_b = self._parse_series_input(args.series_b)

        self.preprocessor.normalize(series_a)
        self.preprocessor.normalize(series_b)

        # Extract stats
        stats_a = self.preprocessor.extract_basic_stats(series_a)
        stats_b = self.preprocessor.extract_basic_stats(series_b)

        # Segment both
        segments_a = self.preprocessor.segment_series(series_a)
        segments_b = self.preprocessor.segment_series(series_b)

        # Encode
        encoded = self.encoder.encode_multivariate(
            [series_a, series_b],
            use_summary=True,
            segments_list=[segments_a, segments_b]
        )

        # Build prompt
        prompt = self.prompt_builder.build_prompt(
            task_type=TaskType.COMPARISON,
            encoded_series=encoded,
            question=None,
            aspect=args.aspect
        )

        llm_response = await self._call_llm(prompt)

        # Format output with statistics
        output = f"# Comparison: {series_a.name} vs {series_b.name}\n\n"
        output += "## Statistical Comparison\n\n"
        output += f"| Metric | {series_a.name} | {series_b.name} | Difference |\n"
        output += "|--------|---------|---------|------------|\n"

        for key in ['mean', 'std', 'min', 'max']:
            val_a = stats_a[key]
            val_b = stats_b[key]
            diff = val_b - val_a
            output += f"| {key.capitalize()} | {val_a:.4f} | {val_b:.4f} | {diff:+.4f} |\n"

        output += f"\n## LLM Analysis\n\n"
        output += llm_response

        return [TextContent(type="text", text=output)]

    def _infer_task_type(self, question: str) -> str:
        """Infer task type from question"""
        question_lower = question.lower()

        if any(word in question_lower for word in ['maximum', 'minimum', 'max', 'min', 'extreme']):
            return "extreme"
        elif any(word in question_lower for word in ['spike', 'anomaly', 'outlier']):
            return "spike"
        elif any(word in question_lower for word in ['trend', 'direction', 'increasing', 'decreasing']):
            return "trend"
        elif any(word in question_lower for word in ['change point', 'shift', 'regime']):
            return "change_point"
        elif any(word in question_lower for word in ['segment', 'phase', 'partition']):
            return "segment"
        elif any(word in question_lower for word in ['compare', 'difference', 'similar']):
            return "comparison"
        else:
            return "describe"

    async def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API (placeholder - implement with actual Anthropic API)

        In production, use the Anthropic API:
        ```python
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
        ```
        """
        # Placeholder response
        logger.info(f"LLM Prompt (first 200 chars): {prompt[:200]}...")

        return """Based on the time series analysis:

The series shows an overall increasing trend with moderate volatility.
Key observations:
- Maximum value: approximately 67.82 at index 145
- Notable spike detected at index 207
- Change point identified around index 112 where the trend accelerates
- The series can be segmented into 3 main phases: initial rise, plateau, and final acceleration

Overall trend: increase
Confidence: high"""

    def _parse_response(self, response: str, task_type: TaskType) -> Dict:
        """Parse LLM response based on task type"""
        if task_type == TaskType.EXTREME:
            return self.parser.parse_extreme_response(response)
        elif task_type == TaskType.SPIKE:
            return self.parser.parse_spike_response(response)
        elif task_type == TaskType.TREND:
            return self.parser.parse_trend_response(response)
        elif task_type == TaskType.CHANGE_POINT:
            return self.parser.parse_change_points_response(response)
        else:
            return {"raw_response": response}

    def _verify_response(
        self,
        series: TimeSeriesData,
        parsed_response: Dict,
        task_type: TaskType
    ) -> Optional[Any]:
        """Verify LLM response against numerical ground truth"""
        try:
            if task_type == TaskType.EXTREME:
                if parsed_response.get("value") and parsed_response.get("index") is not None:
                    return self.verifier.verify_extreme(
                        series,
                        parsed_response["value"],
                        parsed_response["index"],
                        "maximum"  # Could be inferred from question
                    )
            elif task_type == TaskType.SPIKE:
                if parsed_response.get("spike_indices"):
                    return self.verifier.verify_spikes(
                        series,
                        parsed_response["spike_indices"]
                    )
            elif task_type == TaskType.TREND:
                if parsed_response.get("trend"):
                    return self.verifier.verify_trend(
                        series,
                        parsed_response["trend"]
                    )
            elif task_type == TaskType.CHANGE_POINT:
                if parsed_response.get("change_points"):
                    return self.verifier.verify_change_points(
                        series,
                        parsed_response["change_points"]
                    )
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return None

        return None

    def _format_analysis_output(
        self,
        llm_response: str,
        parsed_response: Dict,
        verification_result: Optional[Any],
        task_type: TaskType
    ) -> str:
        """Format final output"""
        output = "# Time Series Analysis\n\n"

        output += "## LLM Analysis\n\n"
        output += llm_response + "\n\n"

        if parsed_response and parsed_response != {"raw_response": llm_response}:
            output += "## Parsed Results\n\n"
            output += "```json\n"
            output += json.dumps(parsed_response, indent=2)
            output += "\n```\n\n"

        if verification_result:
            output += "## Verification\n\n"
            output += f"**Status**: {verification_result.message}\n"
            output += f"**Confidence**: {verification_result.confidence:.2%}\n"

            if verification_result.expected_value is not None:
                output += f"\n- Expected: {verification_result.expected_value}\n"
                output += f"- Predicted: {verification_result.actual_value}\n"

                if verification_result.discrepancy is not None:
                    output += f"- Discrepancy: {verification_result.discrepancy:.4f}\n"

        return output

    async def run(self):
        """Run the MCP server"""
        logger.info("Starting TimeSense MCP server...")
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point"""
    server = TimeSeriesMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
