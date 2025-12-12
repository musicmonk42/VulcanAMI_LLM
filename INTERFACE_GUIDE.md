# VulcanAMI Platform HTML Interface Guide

## Overview

The VulcanAMI Platform now includes a comprehensive web-based HTML interface (`vulcan_unified.html`) that provides full access to all platform functions through an intuitive, single-page application.

**Note:** This unified interface consolidates all features from previous separate interfaces.

## Quick Start

### 1. Start the Platform

```bash
# Start the unified platform
python src/full_platform.py

# Or with custom settings
python src/full_platform.py --host 0.0.0.0 --port 8080
```

### 2. Open the Interface

Simply open `vulcan_unified.html` in your web browser:

```bash
# On Linux/Mac
xdg-open vulcan_unified.html

# On Windows
start vulcan_unified.html

# Or manually: File > Open > vulcan_unified.html
```

### 3. Connect

1. Enter platform URL (default: `http://127.0.0.1:8080`)
2. Optional: Enter API key if authentication is enabled
3. Click "🔌 Connect"

## Features by Tab

### 📊 Dashboard
- **Platform Status**: View version, worker info, and configuration
- **Service Grid**: Monitor all mounted services (VULCAN, Arena, Registry)
- **Real-time Updates**: Auto-refresh on connection

### 🧠 VULCAN-AGI
**Complete access to cognitive AI functions:**

- **Health & Status**: Check VULCAN system health
- **Stream Events**: Real-time SSE event monitoring
- **Invoke Operations**:
  - Reason: Logical inference
  - Plan: Strategic planning
  - Execute: Action execution
  - Self-Improve: Autonomous enhancement
- **World Model Query**: Ask about causal relationships

### ⚔️ Arena
**Agent competition and training:**

- **Run Agent Task**: Execute tasks on specific agents
- **Submit Feedback**: Provide RLHF feedback
- **Run Tournament**: Multi-agent competitions
- **Feedback Dispatch**: Protocol-level feedback routing

### 📝 Registry
**Agent management and authentication:**

- **Get Nonce**: Step 1 of authentication (nonce generation)
- **Login**: Step 2 with cryptographic signature
- **Bootstrap**: Create the first admin agent
- **Onboard Agent**: Register new agents (admin required)
- **Logout**: Revoke JWT tokens

### 📈 Graph IR
**Graph execution and compilation:**

- **Submit Proposals**: GraphixIR graph definitions
- **Sample Templates**:
  - Simple Pipeline
  - ML Workflow
  - LLM Chain
- **Interactive Editor**: JSON-based graph editing

### ⚖️ Governance
**Consensus and proposal management:**

- **View Proposals**: See all active proposals
- **Vote**: Approve/reject/abstain on proposals
- **Trust-Weighted**: Governance based on agent trust scores

### 🔐 Authentication
**Token and credential management:**

- **Get JWT Token**: Generate bearer tokens
- **Test Protected**: Verify authentication
- **Token Display**: View current token
- **Clear Token**: Remove saved credentials

### 📡 Monitoring
**System observability:**

- **Health Status**: Real-time service health
- **Platform Info**: Detailed system information
- **Audit Logs**: View security audit trail (configurable limits)
- **Prometheus Metrics**: Access metrics endpoint
- **System Logs**: Real-time activity log

### 🛠️ Developer Tools
**Advanced features:**

- **Raw API Call**: Direct HTTP requests with custom method/endpoint/body
- **Quick Links**: Direct access to all API documentation
- **Demo Integration**: Links to existing demos (SSE viewer, artifact card, preferences)

## Authentication Methods

### 1. No Authentication
If `auth_method=none`, simply connect - no credentials needed.

### 2. API Key
1. Set API key in platform environment: `export UNIFIED_API_KEY=your-key`
2. Enter key in interface connection bar
3. Connect

### 3. JWT (JSON Web Tokens)
**Option A - Via Registry (Cryptographic):**
1. Go to Registry tab
2. Get Nonce with your agent ID
3. Sign `<agentId>:<nonce>` with your private key
4. Login with signature
5. Token is automatically saved

**Option B - Via Auth Tab (API Key Required):**
1. Go to Auth tab
2. Click "Get Token"
3. Token is automatically saved and used for subsequent requests

## API Endpoints Covered

### Platform Endpoints
- `GET /` - Platform home
- `GET /health` - Health check
- `GET /api/status` - Status information
- `GET /metrics` - Prometheus metrics
- `POST /auth/token` - Generate JWT token
- `GET /api/protected` - Test authentication

### VULCAN Endpoints
- `GET /vulcan/health` - VULCAN health
- `GET /vulcan/v1/status` - Detailed status
- `GET /vulcan/v1/stream` - SSE event stream
- `POST /vulcan/v1/{operation}` - Invoke operations

### Arena Endpoints
- `POST /api/arena/run/{agent_id}` - Run agent task
- `POST /api/arena/feedback` - Submit feedback
- `POST /api/arena/tournament` - Start tournament
- `POST /api/arena/feedback_dispatch` - Dispatch feedback protocol

### Registry Endpoints
- `POST /registry/auth/nonce` - Get authentication nonce
- `POST /registry/auth/login` - Login with signature
- `POST /registry/auth/logout` - Logout
- `POST /registry/bootstrap` - Bootstrap first agent
- `POST /registry/onboard` - Onboard new agent
- `POST /registry/ir/propose` - Submit IR proposal
- `GET /registry/audit/logs` - View audit logs

## Usage Examples

### Example 1: Query VULCAN World Model
1. Go to **VULCAN** tab
2. Scroll to "World Model Query"
3. Enter: `What are the causal effects of X on Y?`
4. Click "🔍 Query"
5. View JSON response

### Example 2: Run Agent in Arena
1. Go to **Arena** tab
2. Enter Agent ID: `agent-001`
3. Enter task JSON:
```json
{
  "task": "generate",
  "prompt": "Write a poem",
  "max_tokens": 100
}
```
4. Click "▶️ Run Task"

### Example 3: Submit Graph IR
1. Go to **Graph IR** tab
2. Click "📄 Load Sample" or choose a template
3. Modify graph as needed
4. Click "📝 Submit Proposal"

### Example 4: View System Health
1. Go to **Monitoring** tab
2. Click "🔄 Refresh" in Health Status card
3. View table with all services, status, and latency

## Configuration Storage

The interface uses browser localStorage to persist:
- Platform URL
- API Key
- JWT Access Token

These are saved automatically and restored on page reload.

### Clear Saved Data
```javascript
// In browser console
localStorage.clear();
```

## Troubleshooting

### Connection Issues
- **Check platform is running**: `curl http://127.0.0.1:8080/health`
- **CORS errors**: Set `UNIFIED_CORS_ORIGINS` in platform environment
- **Port mismatch**: Verify URL matches platform port

### Authentication Failures
- **API Key**: Verify key matches `UNIFIED_API_KEY`
- **JWT**: Ensure token hasn't expired (default: 30 minutes)
- **Registry Login**: Signature must be base64-encoded

### SSE Stream Issues
- **Connection drops**: Network timeouts, check logs
- **No events**: VULCAN may not be generating events
- **Stop/restart**: Use "⛔ Stop Stream" before starting new one

## Security Considerations

### Production Deployment
1. **Use HTTPS**: Never use HTTP in production
2. **Strong API Keys**: Generate cryptographically secure keys
3. **JWT Secrets**: Use long, random secrets
4. **CORS**: Restrict origins to known domains
5. **Rate Limiting**: Platform has built-in rate limits
6. **Token Expiry**: Rotate tokens regularly

### Local Development
- Interface is safe for localhost development
- Do NOT expose interface to public internet without authentication
- Use `.env` file for secrets (not in HTML)

## Browser Compatibility

Tested and working on:
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

Requires:
- JavaScript enabled
- Fetch API support
- EventSource API (for SSE)
- localStorage

## Comparison with Existing Demos

| Feature | vulcan_unified.html | sse_mind.html | artifact_card.html |
|---------|---------------------|---------------|-------------------|
| Full Platform Access | ✅ | ❌ | ❌ |
| VULCAN Functions | ✅ | ❌ | ❌ |
| Arena Functions | ✅ | ❌ | ❌ |
| Registry/Auth | ✅ | ❌ | ❌ |
| SSE Streaming | ✅ | ✅ | ❌ |
| Artifact Viewing | ✅ | ❌ | ✅ |
| Graph IR | ✅ | ❌ | ❌ |
| Governance | ✅ | ❌ | ❌ |
| Monitoring | ✅ | ❌ | ❌ |

**Use Cases:**
- **vulcan_unified.html**: Complete platform control (use this!)
- **sse_mind.html**: Dedicated SSE stream viewer
- **artifact_card.html**: Specific artifact inspection

## Advanced Usage

### Custom API Calls
Use the **Tools** tab > Raw API Call to:
1. Test custom endpoints
2. Debug API responses
3. Prototype new features

Example:
```
Method: POST
Endpoint: /custom/endpoint
Body: {"key": "value"}
```

### Extending the Interface
The interface is pure HTML/JS with no build step. To add features:

1. Add new tab in `<div class="tabs">`
2. Add new panel `<div id="newtab" class="panel">`
3. Add JavaScript functions
4. Use `callAPI()` helper for requests

Example:
```javascript
async function myNewFeature() {
    await callAPI('POST', '/my/endpoint', {data: 'value'}, 'resultDiv');
}
```

## Support

For issues or questions:
1. Check platform logs: `tail -f unified_platform.log`
2. Review browser console for errors
3. Verify API endpoints in platform docs: `http://127.0.0.1:8080/docs`
4. Check VULCAN docs: `http://127.0.0.1:8080/vulcan/docs`

## Next Steps

1. **Explore**: Try each tab and feature
2. **Authenticate**: Set up proper authentication
3. **Monitor**: Use monitoring tab for system health
4. **Integrate**: Connect your agents to Arena
5. **Deploy**: Follow security guidelines for production

---

**Note**: This interface provides complete access to all platform functions. Handle credentials securely and follow security best practices.
