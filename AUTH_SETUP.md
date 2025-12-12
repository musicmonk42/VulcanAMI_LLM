# JWT Authentication Setup for Frontend Interfaces

## Overview

Both HTML interfaces (`index.html` and `vulcan_interface.html`) now support JWT authentication when connecting to the VulcanAMI platform.

## Authentication Flow

1. **API Key → JWT Token → Authenticated Requests**

The frontend automatically handles the following flow:
- User provides API key in the connection bar
- Frontend calls `POST /auth/token` with `X-API-Key` header
- Backend returns JWT token
- Frontend includes `Authorization: Bearer <token>` in all subsequent requests

## Usage Instructions

### For Users

1. **Open the HTML interface** in your browser:
   - `vulcan_unified.html` - Unified interface with all platform features

2. **Enter connection details**:
   - Platform URL: `http://127.0.0.1:8000` (or your server URL)
   - API Key: Enter your API key (optional, required if server uses JWT auth)

3. **Click Connect**:
   - If API key is provided, the interface will:
     - Authenticate with the server
     - Acquire a JWT token
     - Use the token for all API calls
   - If no API key is provided:
     - Will attempt to connect without authentication
     - May work if server allows unauthenticated access

4. **Monitor logs**:
   - Check the monitoring panel for authentication status
   - Success: "✅ JWT token acquired" and "✅ Connected successfully!"
   - Failure: Error messages will indicate what went wrong

### For Administrators

#### Configuring the Backend

The backend must be configured to use JWT authentication:

```bash
# Set environment variables
export AUTH_METHOD=jwt
export API_KEY=your-secure-api-key-here
export JWT_SECRET=your-jwt-secret-here
export JWT_ALGORITHM=HS256
export JWT_EXPIRATION=3600
```

Or in `.env` file:
```env
AUTH_METHOD=jwt
API_KEY=your-secure-api-key-here
JWT_SECRET=your-jwt-secret-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600
```

#### Generating Secure Keys

```bash
# Generate a secure API key
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate a secure JWT secret
python3 -c "import secrets; print(secrets.token_urlsafe(64))"
```

## Technical Details

### Frontend Changes

Both HTML interfaces now include:

1. **API Key Input Field**: Password field for entering API key
2. **JWT Token Storage**: Token stored in memory and localStorage
3. **Auth Headers Helper**: `getAuthHeaders()` function that adds Authorization header
4. **Token Acquisition**: Automatic token fetch on connection
5. **Error Handling**: Clear error messages for auth failures

### API Endpoints

#### Get Token
```
POST /auth/token
Headers:
  X-API-Key: <your-api-key>
  Content-Type: application/json
Body (optional):
  {"sub": "api_user"}

Response:
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### Using Token
```
GET /api/status
Headers:
  Authorization: Bearer <jwt-token>
  Content-Type: application/json
```

## Security Best Practices

1. **Keep API Keys Secret**: Never commit API keys to version control
2. **Use HTTPS**: Always use HTTPS in production
3. **Rotate Keys Regularly**: Change API keys and JWT secrets periodically
4. **Set Short Expiration**: Keep JWT expiration time reasonable (1-24 hours)
5. **Clear Tokens on Logout**: Frontend clears tokens on disconnect
6. **Use Strong Secrets**: Generate cryptographically secure random keys

## Troubleshooting

### Connection Failed: Token acquisition failed
- **Cause**: Invalid API key or server not configured for JWT
- **Solution**: Check API key matches server configuration

### 401 Unauthorized on API calls
- **Cause**: Token expired or invalid
- **Solution**: Disconnect and reconnect to get a new token

### JWT not available
- **Cause**: Backend missing `python-jose` library
- **Solution**: Install with `pip install python-jose[cryptography]`

### No token in requests
- **Cause**: API key not provided or token acquisition failed
- **Solution**: Enter API key in connection bar before connecting

## Example Workflow

```
1. User opens vulcan_unified.html in browser
2. User enters:
   - URL: http://localhost:8000
   - API Key: Abc123XyzSecureKey456
3. User clicks "Connect"
4. Frontend: POST /auth/token with X-API-Key header
5. Backend: Validates API key, returns JWT token
6. Frontend: Stores token, uses in all requests
7. Frontend: GET /api/status with Authorization: Bearer <token>
8. Backend: Validates token, returns status
9. User interacts with platform through authenticated interface
```

## Support

For issues or questions:
- Check server logs for authentication errors
- Review environment variables for correct configuration
- Ensure all required Python packages are installed
- Verify API key matches between frontend and backend
