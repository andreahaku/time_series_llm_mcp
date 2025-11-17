# Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
# Navigate to project directory
cd /Users/administrator/Development/Claude/time_series_llm_mcp

# Install in development mode
pip install -e .
```

### 2. Configure Your LLM API (Optional)

For production use with actual LLM calls, set up your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Then modify `src/server.py` to use the actual API instead of the placeholder:

```python
# In _call_llm method, replace placeholder with:
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=2048,
    messages=[{"role": "user", "content": prompt}]
)
return message.content[0].text
```

### 3. Configure as MCP Server

#### For Claude Code

Edit `~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "timesense": {
      "command": "python",
      "args": [
        "-m",
        "src.server"
      ],
      "cwd": "/Users/administrator/Development/Claude/time_series_llm_mcp",
      "env": {
        "ANTHROPIC_API_KEY": "your-key-here"
      }
    }
  }
}
```

#### For Cursor

Add to your workspace or global settings:

```json
{
  "mcp.servers": {
    "timesense": {
      "command": "python -m src.server",
      "cwd": "/Users/administrator/Development/Claude/time_series_llm_mcp"
    }
  }
}
```

### 4. Test the Server

```bash
# Run the server directly
python -m src.server

# Or run examples
python examples/basic_usage.py
```

## Verification

Once configured, you should see the TimeSense tools available in your MCP client:

- `analyze_time_series`
- `describe_segments`
- `detect_anomalies`
- `compare_series`

## Troubleshooting

### Import Errors

If you get import errors, ensure the package is installed:

```bash
pip install -e .
```

### MCP Server Not Showing Up

1. Check the path in your MCP configuration is correct
2. Verify Python is in your PATH
3. Check the logs in your MCP client (Claude Code, Cursor, etc.)

### LLM API Errors

If using the Anthropic API integration:

1. Verify your API key is set correctly
2. Check you have credits/quota available
3. Ensure `anthropic` package is installed: `pip install anthropic>=0.18.0`

## Development Mode

For development, install with dev dependencies:

```bash
pip install -e ".[dev]"
```

This includes:
- pytest for testing
- black for code formatting
- ruff for linting

Run formatters:

```bash
black src/ examples/
ruff check src/ examples/
```

## Next Steps

1. Try the examples in `examples/basic_usage.py`
2. Read the full documentation in `README.md`
3. Check out the TimeSense paper in `docs/2511.06344v1.pdf`
4. Experiment with your own time series data!

## Architecture Overview

```
User (via Claude Code/Cursor)
    ↓
MCP Protocol
    ↓
TimeSense MCP Server (src/server.py)
    ↓
┌─────────────────────────────────┐
│ Preprocessing (preprocessing.py)│
│ - Normalization                 │
│ - Segmentation                  │
│ - Feature extraction            │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Encoding (encoder.py)           │
│ - Position embeddings           │
│ - <ts> markers                  │
│ - Summary generation            │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Prompts (prompts.py)            │
│ - Task-specific templates       │
│ - EvalTS alignment              │
└─────────────────────────────────┘
    ↓
LLM (Anthropic Claude)
    ↓
┌─────────────────────────────────┐
│ Verification (verification.py)  │
│ - Numerical validation          │
│ - Confidence scoring            │
└─────────────────────────────────┘
    ↓
Formatted Results
```

## Sample Data Format

Time series data should be provided as JSON:

```json
{
  "name": "metric_name",
  "timestamps": [
    "2025-01-01T00:00:00",
    "2025-01-01T01:00:00",
    ...
  ],
  "values": [45.2, 48.1, 52.3, ...]
}
```

Timestamps can be:
- ISO format strings
- Unix timestamps (integers)
- Sequential indices (0, 1, 2, ...)

Values must be numeric (floats or integers).

## Performance Considerations

- **Long series**: Automatically downsampled to `max_points` (default 500)
- **Multivariate**: Can handle multiple series, but consider LLM context limits
- **Verification**: Adds ~100-500ms per analysis (worth it for accuracy)
- **LLM calls**: Typically 2-5 seconds per analysis (depends on model and length)

## Support

For issues or questions:
- GitHub Issues: https://github.com/andreahaku/time_series_llm_mcp/issues
- Refer to the TimeSense paper: https://arxiv.org/abs/2511.06344
- MCP Documentation: https://modelcontextprotocol.io
