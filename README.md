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

### Example Workflow

```python
# In your Claude Code or Cursor session:

# 1. Analyze a single metric
"Use timesense to analyze cpu_metrics.csv and identify any anomalies"

# 2. Compare two deployments
"Compare response times between deployment_v1.json and deployment_v2.json"

# 3. Detect change points
"Find change points in the user_growth.csv time series"

# 4. Comprehensive analysis
"Provide a full analysis of the stock_prices.csv data including trends, volatility, and notable events"
```

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

Following the project document's recommendation, this is a **practical MVP** that:

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
│   ├── project.md             # Design document
│   └── 2511.06344v1.pdf       # TimeSense paper
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
