# VulcanAMI Unified Interface

## Overview

`vulcan_unified.html` is the **single, comprehensive web interface** for the VulcanAMI Platform. It provides complete access to all platform functions through an intuitive, browser-based interface.

**This unified interface replaces and consolidates previous separate HTML interfaces (`index.html` and `vulcan_interface.html`).**

## Quick Start

### Prerequisites
- VulcanAMI Platform backend running (see below)
- Modern web browser (Chrome 90+, Firefox 88+, Safari 14+, Edge 90+)
- JavaScript enabled

### Step 1: Start the Platform

```bash
# Navigate to the repository
cd /path/to/VulcanAMI_LLM

# Start the unified platform server
python src/full_platform.py

# The server will start on http://127.0.0.1:8000 by default
```

### Step 2: Open the Interface

```bash
# Simply open the HTML file in your browser
open vulcan_unified.html

# Or on Linux
xdg-open vulcan_unified.html

# Or on Windows
start vulcan_unified.html
```

### Step 3: Connect

1. The interface will open in your browser
2. Enter the Platform URL (default: `http://127.0.0.1:8000`)
3. (Optional) Enter your API key if authentication is enabled
4. Click **🔌 Connect**
5. Wait for the connection to be established
6. Explore the platform!

## Features

### All-in-One Interface

The unified interface provides access to:

#### 📊 Dashboard
- Platform status and version information
- Service health monitoring
- Real-time connection status
- Mounted services overview

#### 🤖 Agent Pool
- View agent pool statistics
- Spawn new agents
- Monitor agent capabilities
- Track job completion status

#### 🧠 VULCAN-AGI
- System health checks
- Cognitive operations (reason, plan, execute, improve)
- Real-time status monitoring

#### 🌍 World Model
- Causal DAG visualization
- Query causal relationships
- Run interventions
- Prediction engine access

#### ⚔️ Arena
- Run agent tasks
- Submit feedback for RLHF
- Manage tournaments
- Track competition results

#### 📝 Registry
- Agent authentication (nonce + signature)
- Bootstrap first admin agent
- Onboard new agents
- Propose Graph IR

#### 🛡️ Safety
- Safety system status
- Validate actions
- View safety constraints
- Monitor compliance

#### 💬 LLM
- Chat interface
- Reasoning queries
- Token control
- Temperature settings

#### 🔐 Authentication
- JWT token management
- API key configuration
- Protected endpoint testing
- Token display and copy

#### 🛠️ Tools
- Raw API call interface
- Quick links to documentation
- Endpoint reference
- Custom HTTP requests

#### 📋 Logs
- Real-time activity logs
- Request/response tracking
- Error monitoring
- Log management

## Configuration

### Default Settings

The interface comes pre-configured with sensible defaults:
- **Platform URL**: `http://127.0.0.1:8000`
- **Port**: 8000
- **Authentication**: Optional (configurable)

### Persistent Storage

The interface uses browser localStorage to save:
- Platform URL
- API Key
- JWT Access Token

These settings persist across browser sessions for convenience.

### Clear Saved Data

To reset saved configuration:
1. Go to the **Auth** tab
2. Click **Clear Token**
3. Or use browser developer tools: `localStorage.clear()`

## Authentication

### Method 1: No Authentication (Development)

If the platform is running without authentication:
1. Just enter the Platform URL
2. Click Connect
3. No credentials needed

### Method 2: API Key

If the platform requires an API key:
1. Enter Platform URL
2. Enter your API Key in the password field
3. Click Connect
4. The interface will automatically obtain a JWT token

### Method 3: Registry (Cryptographic)

For cryptographic authentication via Registry:
1. Go to the **Registry** tab
2. Enter your Agent ID
3. Click **Get Nonce**
4. Sign `<agentId>:<nonce>` with your private key
5. Enter the signature (base64-encoded)
6. Click **Login**
7. JWT token is saved automatically

## API Endpoint Coverage

The unified interface provides access to **30+ API endpoints** across all platform services:

### Platform (6 endpoints)
- Health check, Status, Metrics, Authentication, Protected endpoints

### VULCAN (10+ endpoints)
- Health, Status, Agent orchestration, World model, Safety, LLM

### Arena (4 endpoints)
- Run tasks, Submit feedback, Tournaments, Feedback dispatch

### Registry (7 endpoints)
- Authentication, Bootstrap, Onboarding, Graph IR, Audit logs

### Governance (2 endpoints)
- Proposals, Voting

## Troubleshooting

### Connection Issues

**Problem**: "Connection failed" error

**Solutions**:
1. Verify platform is running: `curl http://127.0.0.1:8000/health`
2. Check URL matches platform address
3. Ensure no firewall blocking
4. Try `127.0.0.1` instead of `localhost`

### Authentication Failures

**Problem**: 401 Unauthorized

**Solutions**:
1. Verify API key is correct
2. Check JWT token hasn't expired (default: 30 min)
3. Clear saved tokens and reconnect
4. Check platform authentication configuration

### Browser Compatibility

**Problem**: Interface not working properly

**Solutions**:
1. Use a modern browser (Chrome 90+, Firefox 88+, etc.)
2. Enable JavaScript
3. Check browser console for errors
4. Clear browser cache
5. Disable browser extensions that might interfere

### CORS Errors

**Problem**: "CORS policy" errors in console

**Solutions**:
1. Serve HTML via HTTP: `python -m http.server 8001`
2. Then open: `http://localhost:8001/vulcan_unified.html`
3. Or configure platform CORS settings to allow `file://` protocol

## Verification

To verify the interface is complete and functional:

```bash
# Run the verification script
python verify_vulcan_unified.py

# Should show: "✅ SUCCESS: vulcan_unified.html is complete and ready to use!"
```

## Comparison with Previous Interfaces

| Feature | vulcan_unified.html | index.html (removed) | vulcan_interface.html (removed) |
|---------|---------------------|----------------------|----------------------------------|
| Tabs | 11 tabs | 9 tabs | 9 tabs |
| Agent Pool | ✅ Full support | ✅ Basic | ✅ Full |
| World Model | ✅ Dedicated tab | ❌ | ⚠️ Limited |
| Safety | ✅ Full controls | ⚠️ Basic | ✅ Full |
| LLM | ✅ Dedicated tab | ⚠️ Basic | ⚠️ Basic |
| Auth | ✅ Full controls | ✅ Basic | ✅ Full |
| Tools | ✅ Full dev tools | ⚠️ Limited | ✅ Full |
| Logs | ✅ Dedicated tab | ✅ Embedded | ✅ Embedded |
| File Size | 65 KB | 62 KB | 43 KB |

**Benefits of Unified Interface**:
- ✅ Single file to maintain
- ✅ Consistent UX across all features
- ✅ All features in one place
- ✅ Better organization with 11 tabs
- ✅ No confusion about which interface to use

## Best Practices

### Security

1. **Never expose publicly without authentication**
2. Use HTTPS in production
3. Set strong API keys
4. Rotate JWT tokens regularly
5. Don't commit API keys to version control

### Performance

1. Keep browser tab active for real-time updates
2. Clear logs periodically to reduce memory usage
3. Close interface when not in use
4. Use dedicated demos for specific tasks

### Usage

1. Start with Dashboard to verify connectivity
2. Check Agent Pool status before operations
3. Monitor Logs tab for errors
4. Use Tools tab for API testing
5. Save important responses before navigating away

## Support

### Documentation
- **General Guide**: `INTERFACE_GUIDE.md`
- **Technical Details**: `HTML_INTERFACE_README.md`
- **Visual Overview**: `INTERFACE_SUMMARY.md`
- **API Mapping**: `API_ENDPOINT_MAPPING.md`

### Platform Documentation
- `/docs` - Platform API documentation
- `/vulcan/docs` - VULCAN-specific docs
- `/arena/docs` - Arena API docs
- `/registry/spec` - Registry specification

### Reporting Issues

When reporting issues, include:
1. Browser version
2. Platform version
3. Steps to reproduce
4. Error messages (browser console + server logs)
5. Screenshot if applicable

## Summary

**vulcan_unified.html** is your **one-stop interface** for the VulcanAMI Platform:

✅ **Complete** - All platform features accessible  
✅ **Simple** - Just open in browser, no build needed  
✅ **Secure** - JWT authentication support  
✅ **Fast** - Lightweight, pure HTML/JS  
✅ **Tested** - Verified with 42 automated checks  
✅ **Documented** - Comprehensive guides available  

**Get started in 3 steps:**
```bash
1. python src/full_platform.py
2. open vulcan_unified.html
3. Connect and explore!
```

Enjoy the unified VulcanAMI experience! 🚀
