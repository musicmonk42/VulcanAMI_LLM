# VULCAN CLI Usage Guide

Modern cross-platform command-line interface for VULCAN-AGI system.

## Features

- ✅ **Cross-Platform**: Works on Windows, Linux, and macOS with full feature parity
- ✅ **Command History**: Persistent history with up/down arrow navigation  
- ✅ **Tab Completion**: Auto-complete commands with Tab key
- ✅ **Auto-Suggest**: Suggests commands as you type based on history
- ✅ **Syntax Colors**: Beautiful colored output on all platforms
- ✅ **API Authentication**: Secure API key authentication
- ✅ **Real-Time Integration**: Direct HTTP communication with VULCAN server
- ✅ **Error Handling**: User-friendly error messages with troubleshooting guidance

## Installation

The CLI is included with VULCAN. Ensure you have the required dependencies:

```bash
pip install prompt_toolkit httpx
```

## Quick Start

### 1. Start the VULCAN Server

First, start the VULCAN API server:

```bash
python src/api_server.py
# Or using Docker
docker run -p 8000:8000 vulcanami-main
```

### 2. Launch Interactive Mode

```bash
# From Python
python -m vulcan.cli

# Or directly
python src/vulcan/cli/interactive.py
```

## Configuration

### Environment Variables

Configure the CLI using environment variables:

```bash
# Server URL (default: http://localhost:8000)
export VULCAN_SERVER_URL=https://your-server.com

# API Key for authentication
export VULCAN_API_KEY=your-api-key-here
```

### Config File (Optional)

Create `~/.vulcan/config.yaml`:

```yaml
server_url: https://your-server.com
api_key: your-api-key-here
```

**Note**: Environment variables take priority over config file.

## Available Commands

### query (alias: q)
Send a query to VULCAN's reasoning engine.

```
vulcan> query What is the capital of France?
vulcan> q Explain quantum entanglement
```

**Features**:
- Real-time processing with reasoning engine
- Displays metadata (reasoning type, processing time)
- Handles long responses gracefully

---

### status (alias: s)
Check system health and status.

```
vulcan> status
vulcan> s
```

**Output**:
- System operational status
- Server version
- Component health (if available)
- Uptime information

---

### memory (alias: m)
Search VULCAN's memory system.

```
vulcan> memory recent improvements
vulcan> m quantum mechanics
```

**Features**:
- Semantic search across memories
- Relevance scoring
- Configurable result count (default: 10)

---

### config
Display current CLI configuration.

```
vulcan> config
```

**Output**:
- Server URL
- Authentication status
- Environment variable names

---

### improve (alias: i)
Show self-improvement status *(placeholder - requires server implementation)*.

```
vulcan> improve
vulcan> i
```

---

### benchmark (alias: b)
Run performance benchmarks *(placeholder - requires server implementation)*.

```
vulcan> benchmark
vulcan> b
```

---

### help (alias: h, ?)
Display help message with available commands.

```
vulcan> help
vulcan> h
vulcan> ?
```

---

### exit / quit
Exit interactive mode.

```
vulcan> exit
vulcan> quit
```

**Shortcut**: Press `Ctrl+D` (Unix/Mac) or `Ctrl+Z` then `Enter` (Windows)

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `↑` / `↓` | Navigate command history |
| `Tab` | Auto-complete command |
| `Ctrl+C` | Cancel current input (doesn't exit) |
| `Ctrl+D` | Exit CLI (Unix/Mac) |
| `Ctrl+Z` then `Enter` | Exit CLI (Windows) |
| `Ctrl+L` | Clear screen |
| `Ctrl+A` | Move cursor to beginning of line |
| `Ctrl+E` | Move cursor to end of line |
| `Ctrl+K` | Delete from cursor to end of line |
| `Ctrl+U` | Delete entire line |

## Error Handling

The CLI provides helpful error messages for common issues:

### Authentication Error (401)
```
API Error: Authentication failed. Check your API key.
Set VULCAN_API_KEY environment variable or configure in settings.
```

**Solution**: Set `VULCAN_API_KEY` environment variable with valid API key.

---

### Connection Error
```
API Error: Could not connect to server at http://localhost:8000.
Details: [Errno 111] Connection refused
Is the server running?
```

**Solutions**:
1. Check if server is running
2. Verify server URL is correct
3. Check network connectivity
4. Verify firewall settings

---

### Not Found Error (404)
```
API Error: Endpoint not found: /v1/chat
Is the server running the correct version?
```

**Solution**: Ensure server and CLI versions are compatible.

---

### Timeout Error (408)
```
API Error: Request timed out after 30.0s.
The server may be overloaded or processing a complex query.
```

**Solution**: Complex queries may take longer. Consider:
- Simplifying the query
- Increasing timeout (requires code modification)
- Checking server load

---

### Rate Limit Error (429)
```
API Error: Rate limit exceeded. Retry after: 60s
```

**Solution**: Wait for the specified time before retrying.

## Advanced Usage

### Using the Client Programmatically

```python
from vulcan.cli import VulcanClient

# Create client
client = VulcanClient.from_settings()

# Or with explicit configuration
client = VulcanClient(
    base_url="https://your-server.com",
    api_key="your-key",
    timeout=60.0
)

# Send query
try:
    result = client.chat("What is AGI?")
    print(result["response"])
finally:
    client.close()

# Or use context manager
with VulcanClient.from_settings() as client:
    result = client.chat("What is AGI?")
    print(result["response"])
```

### Custom Configuration

```python
from vulcan.cli import CLIConfig

# Load configuration
config = CLIConfig()
print(f"Server: {config.get_server_url()}")
print(f"API Key: {'Set' if config.get_api_key() else 'Not set'}")

# Save configuration
config.save_config(
    server_url="https://new-server.com",
    api_key="new-key-123"
)
```

## Troubleshooting

### Command History Not Working

**Symptom**: Up/down arrows don't navigate history

**Solution**: History is automatically enabled. Ensure `~/.vulcan_history` is writable.

---

### Colors Not Showing

**Symptom**: Output appears without colors

**Solution**: 
- On Windows 10+: Colors should work automatically
- On older Windows: Install Windows Terminal
- On Unix/Mac: Check `TERM` environment variable

---

### API Key Not Recognized

**Symptom**: Getting 401 errors despite setting API key

**Solution**:
1. Verify environment variable is set: `echo $VULCAN_API_KEY`
2. Check for typos in variable name
3. Ensure variable is exported: `export VULCAN_API_KEY=...`
4. Restart CLI after setting variable

---

### Slow Response Times

**Symptom**: Queries take long time to complete

**Possible Causes**:
- Server processing complex query
- Network latency
- Server overloaded
- Cold start (first query after server restart)

**Solutions**:
- Wait for first query to complete (cold start)
- Check server logs for performance issues
- Verify network connectivity
- Consider simpler queries for testing

## Security Best Practices

1. **Never commit API keys to version control**
   ```bash
   # Good: Use environment variables
   export VULCAN_API_KEY=...
   
   # Bad: Hardcoded in scripts
   # VULCAN_API_KEY="secret123"  # DON'T DO THIS
   ```

2. **Use HTTPS in production**
   ```bash
   export VULCAN_SERVER_URL=https://vulcan-prod.com  # Good
   # export VULCAN_SERVER_URL=http://vulcan-prod.com   # Bad - unencrypted
   ```

3. **Protect config file**
   The CLI automatically sets secure permissions (0o600) on `~/.vulcan/config.yaml`.

4. **Rotate API keys regularly**
   Update API keys periodically for security.

5. **Use different keys for different environments**
   ```bash
   # Development
   export VULCAN_API_KEY=dev-key-123
   
   # Production
   export VULCAN_API_KEY=prod-key-456
   ```

## Integration Examples

### Shell Scripts

```bash
#!/bin/bash
# Query VULCAN from shell script

export VULCAN_SERVER_URL=http://localhost:8000
export VULCAN_API_KEY=your-key

python3 << 'EOF'
from vulcan.cli import VulcanClient

with VulcanClient.from_settings() as client:
    result = client.chat("Analyze system logs")
    print(result["response"])
EOF
```

### Python Scripts

```python
#!/usr/bin/env python3
"""Automated VULCAN queries."""

import os
from vulcan.cli import VulcanClient

def main():
    # Configure from environment
    client = VulcanClient.from_settings()
    
    queries = [
        "What are the latest system metrics?",
        "Any security alerts?",
        "System health check"
    ]
    
    for query in queries:
        try:
            result = client.chat(query)
            print(f"Q: {query}")
            print(f"A: {result['response']}\n")
        except Exception as e:
            print(f"Error: {e}\n")
    
    client.close()

if __name__ == "__main__":
    main()
```

### CI/CD Integration

```yaml
# GitHub Actions example
name: VULCAN Query Test

on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install prompt_toolkit httpx
      
      - name: Query VULCAN
        env:
          VULCAN_SERVER_URL: ${{ secrets.VULCAN_SERVER_URL }}
          VULCAN_API_KEY: ${{ secrets.VULCAN_API_KEY }}
        run: |
          python3 -c "
          from vulcan.cli import VulcanClient
          client = VulcanClient.from_settings()
          result = client.health()
          print(f'Status: {result[\"status\"]}')
          client.close()
          "
```

## FAQs

**Q: Does the CLI work offline?**

A: No, the CLI requires connection to a running VULCAN server. However, the server can run locally on `localhost`.

---

**Q: Can I use multiple API keys?**

A: Yes, switch between keys by changing the `VULCAN_API_KEY` environment variable or config file.

---

**Q: Is command history shared between sessions?**

A: Yes, history is stored in `~/.vulcan_history` and persists across sessions.

---

**Q: Can I customize the prompt?**

A: Currently not supported. The prompt is fixed as `vulcan> `.

---

**Q: How do I clear command history?**

A: Delete the history file:
```bash
rm ~/.vulcan_history
```

---

**Q: Can I run the CLI in non-interactive mode?**

A: Yes, use the `VulcanClient` API directly in Python scripts (see Advanced Usage section).

## Support

For issues or questions:
- Check server logs for errors
- Verify configuration with `vulcan> config`
- Consult API server documentation
- Report bugs to repository issue tracker

## Version History

### v2.0.0 (2026-01-16)
- ✨ Complete modernization with `prompt_toolkit`
- ✨ Cross-platform support (Windows, Linux, macOS)
- ✨ Real API integration (no more placeholders)
- ✨ Comprehensive error handling
- ✨ API key authentication
- ✨ Configuration management
- ✨ Industry-standard code quality

### v1.0.0 (Legacy)
- Basic readline-based interface (Unix only)
- Placeholder command handlers
- Limited error handling

---

**Happy querying! 🚀**
