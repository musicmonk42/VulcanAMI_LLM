# VULCAN Chat Interface

Simple, beautiful chat interface for VulcanAMI.

## Prerequisites

Before running the platform, ensure you have the required setup:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create environment file (recommended):**
   ```bash
   cp .env.example .env
   # Edit .env and configure as needed
   ```

   The `.env` file helps configure the platform. See `.env.example` for available options.

3. **AWS Configuration (optional):**
   If you have `boto3` installed and want to use AWS Secrets Manager, ensure AWS is configured:
   ```bash
   export AWS_DEFAULT_REGION=us-east-1  # or your preferred region
   ```
   If you don't need AWS Secrets Manager, the platform will fall back to environment variables.

## Quick Start

### Option 1: All-in-One Script

```bash
chmod +x start_chat_interface.sh
./start_chat_interface.sh
```

### Option 2: Manual Start

1. Start the backend:
```bash
python src/full_platform.py
```

2. Open `vulcan_chat.html` in your browser

## Configuration

Click the ⚙️ icon in the chat interface to configure:
- **API URL**: Default is `http://localhost:8080`
- **API Key**: Optional (if you've enabled authentication)

Settings are saved in browser localStorage.

## Features

### Frontend (vulcan_chat.html)
- ✅ Clean, modern UI
- ✅ Message history
- ✅ Typing indicators
- ✅ Shows which systems were used
- ✅ Example prompts
- ✅ Auto-reconnect
- ✅ Single HTML file (no dependencies)
- ✅ **Internal Metrics Tabs** - View system internals directly from the UI:
  - 📊 **Metrics Tab** - Health status, system status, cognitive systems, LLM status
  - ⚠️ **Warnings Tab** - Safety status, audit logs, adversarial monitoring
  - 🔧 **Internals Tab** - World model, memory, hardware, routing, and API endpoints list

### Backend (VULCAN Platform)
- ✅ Full VULCAN platform integration
- ✅ World Model (causal reasoning)
- ✅ Unified Reasoner (5 modes)
- ✅ Memory (Graph RAG)
- ✅ Safety Validator (CSIU protocol)
- ✅ Planning Engine (when needed)
- ✅ LLM Core (GraphixTransformer)

## API Endpoints

The chat functionality is available through the VULCAN platform:

### VULCAN Chat (`/vulcan/v1/chat`)

The main VULCAN service includes a comprehensive chat endpoint with full platform integration.

### Request Format

**POST** `/vulcan/v1/chat`

```json
{
  "message": "What would happen if we increased marketing spend by 50%?",
  "max_tokens": 1024,
  "enable_reasoning": true,
  "enable_memory": true,
  "enable_safety": true,
  "enable_planning": true,
  "enable_causal": true,
  "history": [
    {"role": "user", "content": "Previous question"},
    {"role": "assistant", "content": "Previous answer"}
  ]
}
```

### Response Format

```json
{
  "response": "Based on causal analysis of your business model...",
  "systems_used": ["world_model", "reasoner", "memory", "safety"]
}
```

### Health Check

**GET** `/vulcan/health`

```json
{
  "status": "healthy",
  "service": "vulcan-chat",
  "active_systems": "7/7",
  "systems": {
    "world_model": "active",
    "reasoner": "active",
    "memory": "active",
    "safety": "active",
    "planner": "active",
    "llm": "active",
    "graph_rag": "active"
  }
}
```

### Internal Metrics Endpoints (used by Metrics Tabs)

The chat interface's metrics tabs consume the following API endpoints:

| Tab | Endpoints |
|-----|-----------|
| Metrics | `/health`, `/v1/status`, `/v1/cognitive/status`, `/v1/llm/status` |
| Warnings | `/safety/status`, `/safety/audit/recent`, `/api/adversarial/status` |
| Internals | `/world-model/status`, `/memory/status`, `/hardware/status`, `/v1/routing/status` |

## Customization

### Change Port

Edit environment variable or pass command line argument:

```bash
# Environment variable
export PORT=9000
python src/full_platform.py

# Or command line
python src/full_platform.py --port 9000
```

### Enable Authentication

Set API key via environment variable:

```bash
export UNIFIED_API_KEY="your-secret-key"
python src/full_platform.py --auth-method api_key
```

Then include the key in your requests:

```javascript
fetch('http://localhost:8080/vulcan/v1/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'your-secret-key'
  },
  body: JSON.stringify({
    message: 'Hello!',
    enable_reasoning: true
  })
})
```

## Troubleshooting

### AWS Region Error (NoRegionError)

If you see an error like:
```
botocore.exceptions.NoRegionError: You must specify a region.
```

This happens when `boto3` is installed but AWS is not configured. Solutions:

1. **Set AWS region** (if you want to use AWS Secrets Manager):
   ```bash
   export AWS_DEFAULT_REGION=us-east-1
   ```

2. **Or configure AWS credentials properly**:
   ```bash
   aws configure
   ```

3. **Or ignore AWS** - the platform will fall back to environment variables for secrets.

### "Offline" status in chat interface

1. Make sure `src/full_platform.py` is running
2. Check the console for errors
3. Verify URL in settings (⚙️ icon)
4. Check CORS is enabled (it is by default)

### Slow responses

1. First message initializes all systems (takes ~5-10s)
2. Subsequent messages are much faster
3. Check system resources (CPU/RAM)

### CORS errors

The backend has CORS enabled for all origins by default. If you still see issues:

1. Check that you're accessing from `http://` not `file://`
2. Run a local server for your HTML:
   ```bash
   python -m http.server 3000
   ```
3. Then open `http://localhost:3000/vulcan_chat.html`

### Components not loading

Some components may not be available depending on dependencies. Check the `/vulcan/health` endpoint to see which systems are active.

## Architecture

```
User → vulcan_chat.html → HTTP POST → /vulcan/v1/chat
                                           ↓
                               ┌─────────────────────────┐
                               │     VULCAN Chat Endpoint       │
                               │  (unified_chat.py)     │
                               └─────────────────────────┘
                                           ↓
                           ┌────────────────┴────────────────┐
                           ↓                                 ↓
                    EXAMINE Phase                     SELECT Phase
                    - Graph RAG Retrieval             - Unified Reasoning
                    - World Model State               - Safety Validation
                                                      - Planning (if needed)
                           ↓                                 ↓
                    APPLY Phase                      REMEMBER Phase
                    - LLM Generation                 - Store in Memory
                    - Safety Filtering               - Update World Model
                           ↓
                    Response → JSON → vulcan_chat.html → Display
```

## Performance

Typical response times:

| Phase | Time |
|-------|------|
| First message (cold start) | 5-10s |
| Subsequent messages | 1-3s |
| With planning enabled | +2-3s |
| Memory retrieval | +0.5-1s |

Systems activated per message: 5-7 depending on query type and enabled features.

## Files

| File | Description |
|------|-------------|
| `src/vulcan/endpoints/unified_chat.py` | VULCAN chat endpoint with full platform integration |
| `src/vulcan/main.py` | VULCAN main app with integrated `/v1/chat` |
| `src/full_platform.py` | Unified platform that mounts all services |
| `start_chat_interface.sh` | One-click startup script |
| `vulcan_chat.html` | Frontend chat interface |
| `README_CHAT.md` | This documentation |

## Related Documentation

- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [COMPREHENSIVE_REPO_OVERVIEW.md](COMPREHENSIVE_REPO_OVERVIEW.md) - Full system overview
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment instructions
