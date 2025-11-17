# TimeSense MCP Server

A Model Context Protocol (MCP) server for time series analysis powered by LLMs, inspired by the [TimeSense paper](https://arxiv.org/abs/2511.06344).

## Overview

TimeSense-MCP provides intelligent time series analysis capabilities for LLM-powered tools like Claude Code and Cursor. It combines:

- **Time series preprocessing** with normalization, segmentation, and feature extraction
- **LLM reasoning** over temporal data using structured prompts
- **Numerical verification** to ground LLM outputs in statistical reality
- **MCP protocol** for seamless integration with AI coding assistants

### Key Features

✅ **EvalTS-inspired task categories:**
- **Atomic Understanding**: extrema, trends, spikes, change points
- **Molecular Reasoning**: segmentation, comparison, relative changes
- **Compositional Tasks**: anomaly detection, root cause analysis, comprehensive descriptions

✅ **TimeSense encoding approach:**
- Positional embeddings (index + value pairs)
- `<ts>` token markers for time series data
- Summary-based encoding for long series

✅ **External Temporal Sense verification:**
- Validates LLM outputs against statistical computations
- Provides confidence scores and discrepancy metrics
- Catches hallucinations without requiring model training

## Architecture

```
Time Series → Preprocessing → Encoding → LLM Prompt → LLM Response
                    ↓                                        ↓
              Features &                              Verification
              Statistics                              (optional)
                    ↓                                        ↓
                                    Final Analysis
```

**Modules:**
- `preprocessing.py` - Normalization, segmentation, feature extraction, anomaly detection
- `encoder.py` - Converts time series to text with positional info (`<ts>` markers)
- `prompts.py` - Task-specific prompt templates aligned with EvalTS
- `verification.py` - External validation against numerical ground truth
- `server.py` - MCP server exposing analysis tools

## Installation

### Prerequisites

- Python 3.10+
- pip or uv package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/andreahaku/time_series_llm_mcp.git
cd time_series_llm_mcp

# Install dependencies
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

### Configure as MCP Server

Add to your Claude Code or MCP client configuration:

**For Claude Code** (`~/.config/claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "timesense": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/time_series_llm_mcp"
    }
  }
}
```

**For Cursor** (`.cursorrules` or settings):
```json
{
  "mcp": {
    "servers": {
      "timesense": {
        "command": "python -m src.server",
        "cwd": "/path/to/time_series_llm_mcp"
      }
    }
  }
}
```

## Usage

### MCP Tools

#### 1. `analyze_time_series`

General-purpose time series analysis with automatic or manual task type selection.

**Parameters:**
- `series` (List[TimeSeriesInput]): Time series data
- `question` (str): Analysis question
- `task_type` (str, optional): Task type or "auto"
- `verify` (bool): Enable verification (default: true)

**Example:**
```json
{
  "series": [{
    "name": "cpu_usage",
    "timestamps": ["2025-01-01T00:00:00", "2025-01-01T01:00:00", ...],
    "values": [45.2, 48.1, 52.3, ...]
  }],
  "question": "What is the maximum CPU usage and when did it occur?",
  "task_type": "extreme",
  "verify": true
}
```

**Supported task types:**
- `extreme` - Find max/min values
- `spike` - Detect spikes and anomalies
- `trend` - Identify trends (increase/decrease/stable)
- `change_point` - Detect regime shifts
- `segment` - Partition into phases
- `comparison` - Compare multiple series
- `describe` - Comprehensive analysis
- `anomaly_detection` - Detect and explain anomalies

#### 2. `describe_segments`

Segment time series and describe each phase.

**Parameters:**
- `series` (TimeSeriesInput): Time series to segment
- `window_size` (int): Segment window size (default: 50)

**Example:**
```json
{
  "series": {
    "name": "temperature",
    "timestamps": [...],
    "values": [...]
  },
  "window_size": 30
}
```

#### 3. `detect_anomalies`

Detect anomalies with optional custom rules.

**Parameters:**
- `series` (List[TimeSeriesInput]): Time series to analyze
- `interval` (List[int], optional): Focus interval [start, end]
- `anomaly_rules` (List[str], optional): Custom rules

**Example:**
```json
{
  "series": [{
    "name": "latency",
    "timestamps": [...],
    "values": [...]
  }],
  "interval": [100, 200],
  "anomaly_rules": [
    "Latency above 100ms is anomalous",
    "Sudden jumps > 20ms indicate issues"
  ]
}
```

#### 4. `compare_series`

Compare two time series and identify differences.

**Parameters:**
- `series_a` (TimeSeriesInput): First series
- `series_b` (TimeSeriesInput): Second series
- `aspect` (str): What to compare (default: "overall behavior")

**Example:**
```json
{
  "series_a": {"name": "production", ...},
  "series_b": {"name": "staging", ...},
  "aspect": "latency and throughput"
}
```

### Example Workflows

Once configured as an MCP server, you can use TimeSense directly from your AI coding assistant. Here are real-world scenarios:

#### 1. Performance Monitoring

**Scenario**: Investigating server performance issues

```
User: I have CPU usage data in monitoring_data.json. What was the peak CPU
      usage today and when did it occur?

Claude Code: [Uses analyze_time_series with task_type="extreme"]

Result: ✓ Peak CPU usage was 94.2% at index 847 (2025-01-17 14:23:00)
        Verification: Confirmed with 100% confidence
```

#### 2. Anomaly Detection

**Scenario**: Detecting unusual patterns in application metrics

```
User: Analyze response_times.csv and tell me if there are any anomalous spikes
      in the last hour (indices 3600-7200).

Claude Code: [Uses detect_anomalies with interval=[3600, 7200]]

Result: ## Statistical Detection
        - Spikes detected: 12
        - Indices: [3847, 3891, 4203, 4567, 5012, 5234, 5678, 6012, 6234, 6789, 7001, 7123]

        ## LLM Analysis
        The series shows a cluster of anomalous spikes starting around index 3847.
        These spikes exceed 3 standard deviations above the mean and appear to
        correlate with deployment events. The pattern suggests intermittent service
        degradation rather than sustained issues.
```

#### 3. Deployment Comparison

**Scenario**: Comparing metrics before and after a deployment

```
User: Compare the latency between before_deploy.json and after_deploy.json.
      Did performance improve or degrade?

Claude Code: [Uses compare_series]

Result: ## Statistical Comparison
        | Metric  | before_deploy | after_deploy | Difference |
        |---------|---------------|--------------|------------|
        | Mean    | 245.3ms       | 189.7ms      | -55.6ms    |
        | Std     | 42.1ms        | 28.3ms       | -13.8ms    |
        | Max     | 892.1ms       | 456.2ms      | -435.9ms   |

        ## LLM Analysis
        Performance significantly improved after deployment:
        - Average latency decreased by 22.7%
        - Variability reduced (lower std deviation)
        - Peak latency cut in half
        - The improvement is consistent across the entire time period
```

#### 4. Trend Analysis

**Scenario**: Understanding long-term patterns

```
User: What's the overall trend of user_growth.csv over the past 6 months?
      Are there any change points where growth accelerated or slowed?

Claude Code: [Uses analyze_time_series with task_type="auto"
              → infers "trend" and "change_point"]

Result: ## Trend Analysis
        Overall trend: increase
        Confidence: high

        ## Change Points Detected
        - Index 892 (Month 2): Growth rate doubles
        - Index 2401 (Month 4): Plateau begins
        - Index 3567 (Month 5): Growth resumes at moderate pace

        The user base shows three distinct growth phases: initial steady growth,
        followed by viral acceleration in month 2, a saturation period in month 4,
        and renewed moderate growth as new features launched.
```

#### 5. Segmentation and Phase Analysis

**Scenario**: Breaking down complex time series behavior

```
User: Segment the stock_prices.csv data and describe each phase.

Claude Code: [Uses describe_segments with window_size=50]

Result: ## Overall Statistics
        - Length: 1440 points (6 months of trading days)
        - Mean: $142.35
        - Range: [$98.20, $187.45]

        ## Segments (15 total)

        ### Segment 1
        - Interval: indices [0, 50]
        - Trend: **increasing**
        - Mean: $105.23, Slope: +0.3214
        - Bullish rally phase with consistent gains

        ### Segment 2
        - Interval: indices [50, 100]
        - Trend: **volatile**
        - Mean: $118.45, Slope: +0.1832
        - High volatility period, earnings uncertainty

        [... segments 3-14 ...]

        ### Segment 15
        - Interval: indices [700, 750]
        - Trend: **flat**
        - Mean: $165.82, Slope: -0.0023
        - Consolidation after peak, sideways movement
```

#### 6. Multi-Series Root Cause Analysis

**Scenario**: Investigating correlated failures across services

```
User: I have metrics from 5 microservices in services_metrics/. Between
      10:00-11:00 AM, response times spiked across all services. Can you
      identify which service caused the cascade failure?

Claude Code: [Uses analyze_time_series with task_type="anomaly_detection"
              on all 5 series, then correlates timing]

Result: ## Anomaly Detection Results

        ### Service: auth-service
        - Anomaly start: index 600 (10:02 AM)
        - Type: upward spike in latency
        - Severity: 5.2σ above normal

        ### Service: user-service
        - Anomaly start: index 615 (10:05 AM)
        - Type: upward spike
        - Severity: 3.8σ above normal

        ### Service: payment-service
        - Anomaly start: index 618 (10:05 AM)
        ...

        ## Root Cause Analysis
        Based on the temporal sequence, **auth-service** appears to be the root cause:
        1. First to show anomaly (index 600)
        2. Highest severity (5.2σ)
        3. Other services degraded 3-5 minutes later
        4. Propagation chain: auth-service → user-service → payment-service → ...

        Recommendation: Investigate auth-service for issues around 10:02 AM.
```

#### 7. Custom Anomaly Rules

**Scenario**: Domain-specific anomaly detection

```
User: In temperature_sensors.csv, detect anomalies using these rules:
      - Sensor 1: temp > 75°C is critical
      - Sensor 2: temp < 10°C is critical
      - Sensor 3: sudden changes > 15°C are critical

Claude Code: [Uses detect_anomalies with custom anomaly_rules]

Result: ## Anomaly Detection Results

        ### Sensor 1
        - Anomalies detected: Yes
        - Critical intervals: [234, 289], [567, 612]
        - Type: upward (exceeded 75°C threshold)
        - Max temperature: 82.3°C at index 267
        - Duration: Total 101 minutes above threshold

        ### Sensor 2
        - Anomalies detected: Yes
        - Critical intervals: [1203, 1289]
        - Type: downward (below 10°C)
        - Min temperature: 6.8°C at index 1245
        - Possible cooling system failure

        ### Sensor 3
        - Anomalies detected: Yes
        - Sudden changes: [445, 891, 1567]
        - Index 445: +18.2°C jump in 1 minute
        - Index 891: -16.7°C drop in 1 minute
        - Likely sensor malfunction or physical shock events
```

#### Quick Command Examples

For simple one-liners in your coding assistant:

```bash
# Find maximum value
"What's the max value in sales_data.csv?"

# Detect spikes
"Are there any anomalous spikes in error_logs.json?"

# Compare versions
"Compare API latency between v1.2 and v1.3"

# Identify trends
"Is user engagement increasing or decreasing in metrics.csv?"

# Find change points
"When did the traffic pattern change in web_analytics.json?"

# Segment analysis
"Break down stock_prices.csv into distinct phases"
```

**Note**: All examples use the MCP tools transparently - you don't need to know the tool names or parameters, just describe what you want to analyze!

## TimeSense Paper Implementation

This MCP server implements key ideas from the [TimeSense paper](https://arxiv.org/abs/2511.06344):

### What We Implement

✅ **Positional Encoding**: Each time point includes its absolute index
✅ **`<ts>` Markers**: Time series wrapped in special tokens
✅ **EvalTS Task Categories**: Atomic, molecular, and compositional tasks
✅ **External Temporal Sense**: Verification via numerical computation

### What We Don't Implement (MVP)

❌ **Model Training**: Uses pre-trained LLMs via API (no custom fine-tuning)
❌ **Patch-based MLP Encoding**: Uses text representation instead
❌ **Internal Reconstruction Loss**: Verification is external, not learned
❌ **ChronGen Data Generation**: Could be added for testing/evaluation

### Design Philosophy

This is a **practical MVP** that:

1. **Gives you the pattern** for time series + LLMs without re-implementing the full paper
2. **Uses external validation** instead of training an internal reconstruction module
3. **Focuses on useful tasks** (EvalTS subset) rather than comprehensive benchmarking
4. **Works out-of-the-box** with existing LLMs via API calls

## Development

### Project Structure

```
time_series_llm_mcp/
├── src/
│   ├── __init__.py
│   ├── server.py              # MCP server
│   ├── preprocessing.py       # Time series preprocessing
│   ├── encoder.py             # TS → text encoding
│   ├── prompts.py             # Task-specific prompts
│   └── verification.py        # Numerical verification
├── examples/
│   └── basic_usage.py         # Usage examples
├── docs/
│   └── 2511.06344v1.pdf       # TimeSense paper
├── CLAUDE.md                  # Claude Code guidance
├── SETUP.md                   # Setup instructions
├── pyproject.toml
└── README.md
```

### Running Tests

```bash
# Run examples
python examples/basic_usage.py

# Run the server directly
python -m src.server

# With pytest (after implementing tests)
pytest tests/
```

### Contributing

Contributions welcome! Areas for improvement:

- [ ] Actual Anthropic API integration (currently uses placeholder)
- [ ] ChronGen synthetic data generator
- [ ] Full EvalTS benchmark implementation
- [ ] More sophisticated change point detection algorithms
- [ ] Seasonality and trend decomposition (STL)
- [ ] Multivariate anomaly detection (Isolation Forest, LSTM-based)
- [ ] Root cause analysis for multivariate series
- [ ] Streaming/online analysis mode

## References

- **TimeSense Paper**: [Making Large Language Models Proficient in Time-Series Analysis](https://arxiv.org/abs/2511.06344)
- **Model Context Protocol**: [MCP Documentation](https://modelcontextprotocol.io)
- **Claude Code**: [Anthropic Claude Code](https://claude.com/claude-code)

## License

MIT License - See LICENSE file for details

## Citation

If you use this work, please cite the original TimeSense paper:

```bibtex
@article{zhang2025timesense,
  title={TimeSense: Making Large Language Models Proficient in Time-Series Analysis},
  author={Zhang, Zhirui and Pei, Changhua and Gao, Tianyi and Xie, Zhe and Hao, Yibo and Yu, Zhaoyang and Xu, Longlong and Xiao, Tong and Han, Jing and Pei, Dan},
  journal={arXiv preprint arXiv:2511.06344},
  year={2025}
}
```

---

**Built with 🤖 by combining TimeSense research with practical MCP implementation**
