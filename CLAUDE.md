# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TimeSense MCP Server is a Model Context Protocol (MCP) server that provides time series analysis capabilities to LLMs. It implements key ideas from the TimeSense paper (arXiv:2511.06344) as a practical MVP, using **external validation** instead of model training to ground LLM reasoning in statistical reality.

## Commands

### Development Setup
```bash
# Install for development
pip install -e ".[dev]"

# Run the MCP server
python -m src.server

# Run examples
python examples/basic_usage.py
```

### Code Quality
```bash
# Format code
black src/ examples/

# Lint code
ruff check src/ examples/
```

### Testing
```bash
# Currently using placeholder LLM responses
# To enable real API calls: set ANTHROPIC_API_KEY and uncomment API code in src/server.py:_call_llm
```

## Architecture

### Data Flow Pipeline

The codebase follows a **unidirectional pipeline** from raw time series to verified LLM analysis:

```
TimeSeriesData → Preprocessing → Encoding → Prompt → LLM → Verification → Output
```

### Core Architecture Patterns

**1. External Temporal Sense Pattern**
Unlike the TimeSense paper which trains an internal reconstruction module, this implementation uses **external verification**:
- LLM generates analysis based on encoded time series
- `verification.py` computes ground truth using statistical methods
- Results are cross-checked and confidence scored
- No model training required - works with any LLM API

**2. Task-Specific Encoding Strategy**
The `encoder.py` module adapts encoding based on task complexity:
- **Atomic tasks** (extreme, spike): Full point-by-point encoding with `<ts>` markers
- **Molecular tasks** (segment, comparison): Summary-based encoding with segment features
- **Compositional tasks** (anomaly detection): Hybrid approach combining both

**3. Three-Layer Module Organization**

```
Layer 1: Data Preparation (preprocessing.py)
├─ Normalization (z-score/minmax)
├─ Feature extraction (stats, FFT)
├─ Segmentation (windowed with overlap)
└─ Anomaly detection (z-score, change points)

Layer 2: LLM Interface (encoder.py + prompts.py)
├─ TimeSense encoding: index + value pairs in <ts> tags
├─ EvalTS-aligned prompts (Atomic/Molecular/Compositional)
└─ Response parsing with regex extraction

Layer 3: Validation (verification.py)
├─ Ground truth computation
├─ Discrepancy measurement
└─ Confidence scoring (0.0-1.0)
```

### Key Design Decisions

**Positional Embeddings**: Each time point is encoded as `[channel_name index timestamp value]` to preserve absolute temporal position - critical for tasks like change point detection and interval queries.

**Downsampling Strategy**: Long series (>500 points) are downsampled using configurable methods:
- `uniform`: Regular sampling
- `peak_preserving`: Keeps extrema + uniform points
- `adaptive`: Variance-based selection

**Task Type Inference**: `_infer_task_type()` in `server.py` uses keyword matching to automatically select the appropriate analysis pipeline when `task_type="auto"`.

## MCP Tool Implementation

The server exposes 4 MCP tools via Pydantic schemas (`TimeSeriesInput`, `AnalyzeTimeSeriesInput`, etc.):

1. **analyze_time_series**: Main analysis tool with 8 task types
2. **describe_segments**: Pure preprocessing, no LLM call
3. **detect_anomalies**: Combines statistical detection + LLM interpretation
4. **compare_series**: Multivariate analysis

Each tool method follows this pattern:
```python
async def _tool_method(args: Dict) -> list[TextContent]:
    # 1. Parse input (TimeSeriesInput → TimeSeriesData)
    # 2. Preprocess (normalize, extract features)
    # 3. Encode (task-specific encoding)
    # 4. Build prompt (EvalTS-aligned template)
    # 5. Call LLM (placeholder or API)
    # 6. Parse response (regex extraction)
    # 7. Verify (optional, computes ground truth)
    # 8. Format output (markdown with stats)
```

## Working with Time Series Data

**Input Format**: All time series follow the `TimeSeriesInput` schema:
```python
{
  "name": str,           # Identifier
  "timestamps": List,    # ISO strings, unix timestamps, or indices
  "values": List[float]  # Numeric values
}
```

**Internal Representation**: Converted to `TimeSeriesData` dataclass:
```python
@dataclass
class TimeSeriesData:
    name: str
    timestamps: np.ndarray
    values: np.ndarray
    normalized_values: Optional[np.ndarray]  # After preprocessing
    indices: Optional[np.ndarray]            # Absolute positions
```

## EvalTS Task Categories

The codebase implements a subset of the EvalTS benchmark:

**Atomic Understanding** (single-series, fundamental patterns):
- `extreme`: Find min/max with indices
- `spike`: Detect outliers (z-score method)
- `trend`: Classify as increase/decrease/stable/volatile
- `change_point`: Detect regime shifts

**Molecular Reasoning** (relationships between units):
- `segment`: Partition into phases with trend labels
- `comparison`: Analyze differences between two series
- `relative`: Assess changes between consecutive trends

**Compositional Tasks** (complex, multivariate):
- `describe`: Comprehensive analysis combining multiple atomic tasks
- `anomaly_detection`: Detect + explain anomalies with custom rules

## Adding LLM API Integration

Currently uses placeholder responses. To integrate real Anthropic API:

1. Add dependency: `anthropic>=0.18.0` to `pyproject.toml`
2. Set environment variable: `ANTHROPIC_API_KEY`
3. In `src/server.py`, replace `_call_llm` method:

```python
async def _call_llm(self, prompt: str) -> str:
    from anthropic import Anthropic
    import os

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text
```

## Extension Points

**Adding New Task Types**:
1. Add to `TaskType` enum in `prompts.py`
2. Implement template method in `PromptBuilder`
3. Add parser method in `ResponseParser`
4. Add verification method in `TimeSeriesVerifier`
5. Update `_infer_task_type()` in `server.py`

**Custom Preprocessing**:
Extend `TimeSeriesPreprocessor` class - all methods accept `TimeSeriesData` and return processed versions or feature dictionaries.

**Alternative Encodings**:
Subclass `TimeSeriesEncoder` and override `encode_full_series()` or `encode_summary()` methods.

## Critical Implementation Notes

**Verification Tolerance**: `TimeSeriesVerifier.__init__(tolerance=0.1)` - 10% relative error is accepted. Adjust for stricter/looser validation.

**Segment Features**: The `SegmentFeatures` dataclass includes a `trend_label` computed via simple rules (slope + volatility). For more sophisticated trend detection, extend the logic in `TimeSeriesPreprocessor.segment_series()`.

**Context Limits**: The `max_points=500` default in `TimeSeriesEncoder` prevents context overflow. For longer series, rely on summary encoding or increase the limit cautiously.

**MCP Protocol**: The server uses `stdio_server()` for MCP communication. All tool responses must be `list[TextContent]` - never return raw strings or dicts.

## What NOT to Implement

Per the design philosophy (see `docs/project.md`), this is an **MVP pattern implementation**, not a full paper reproduction:

- ❌ Don't add model training/fine-tuning (defeats the external validation approach)
- ❌ Don't implement patch-based MLP encoders (uses text representation)
- ❌ Don't add ChronGen data generator unless building evaluation suite
- ❌ Don't implement internal reconstruction loss (verification is external)

Focus on:
- ✅ More sophisticated statistical methods (STL decomposition, Isolation Forest)
- ✅ Additional EvalTS task types (root cause analysis, multivariate anomaly detection)
- ✅ Streaming/online analysis modes
- ✅ Better prompt engineering for specific domains
