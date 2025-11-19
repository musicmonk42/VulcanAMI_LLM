# Security Policy

## Reporting a Vulnerability

Novatrax Labs takes the security of VulcanAMI_LLM seriously. If you have discovered a security vulnerability, we appreciate your help in disclosing it to us in a responsible manner.

### Where to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please report security vulnerabilities by:

1. **Email:** security@novatraxlabs.com
2. **Subject Line:** "[SECURITY] VulcanAMI_LLM - Brief Description"

### What to Include

Please include the following information in your report:

- **Type of vulnerability** (e.g., SQL injection, XSS, remote code execution)
- **Full paths of source file(s)** related to the vulnerability
- **Location of the affected source code** (tag/branch/commit or direct URL)
- **Step-by-step instructions to reproduce** the issue
- **Proof-of-concept or exploit code** (if possible)
- **Impact of the issue**, including how an attacker might exploit it
- **Your name/handle** for acknowledgment (optional)

### Response Timeline

- **Initial Response:** Within 48 hours
- **Vulnerability Assessment:** Within 5 business days
- **Fix Development:** Timeline depends on severity
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Next scheduled release
- **Coordinated Disclosure:** Typically 90 days after fix is available

## Security Update Policy

### Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.3.x   | :white_check_mark: |
| 1.2.x   | :white_check_mark: |
| < 1.2   | :x:                |

### Security Updates

Security updates are released through:
- Security advisories (for enterprise customers)
- GitHub Security Advisories (when applicable)
- Release notes clearly marked with [SECURITY] tags

## Known Security Considerations

### Current Vulnerabilities

See `SECURITY_AUDIT_REPORT.md` for a comprehensive list of identified vulnerabilities and their status.

As of 2025-11-19, the following CRITICAL issues have been addressed:
- ✅ CWE-502: Unsafe deserialization (pickle/torch.load)
- ✅ Race condition in bootstrap authentication
- ✅ Hardcoded file paths
- ✅ Weak JWT claims
- ✅ Timing attack vulnerabilities

### High-Risk Areas

The following components require careful security review:
1. **Pickle/Torch deserialization** - Use `safe_pickle_load()` only
2. **Bootstrap endpoint** - Critical for initial system setup
3. **Authentication flow** - Challenge-response with signatures
4. **JWT token handling** - Revocation list management
5. **File operations** - Potential path traversal
6. **GraphQL endpoints** - Query complexity attacks
7. **Redis connections** - Require authentication in production

### Security Best Practices

When deploying VulcanAMI_LLM:

#### Required
- ✅ Set strong `JWT_SECRET_KEY` (32+ characters, high entropy)
- ✅ Enable HTTPS/TLS for all endpoints
- ✅ Use Redis with password authentication
- ✅ Configure CORS with explicit origin allowlist
- ✅ Set `ENFORCE_HTTPS_BOOTSTRAP=true` in production
- ✅ Rotate bootstrap key after first use
- ✅ Enable audit logging
- ✅ Set appropriate rate limits

#### Recommended
- 📋 Use certificate pinning for external API calls
- 📋 Enable database encryption at rest
- 📋 Implement network segmentation
- 📋 Use secrets manager (not environment variables)
- 📋 Enable SIEM integration
- 📋 Implement log retention policies (90 days recommended)
- 📋 Regular dependency scanning
- 📋 Periodic penetration testing

#### Environment Variables

**Critical Security Variables:**
```bash
# REQUIRED - Must be strong and unique
JWT_SECRET_KEY=<32+ character random string>

# REQUIRED - One-time bootstrap protection  
BOOTSTRAP_KEY=<32+ character random string>

# REQUIRED for production
ENFORCE_HTTPS_BOOTSTRAP=true
ENFORCE_HTTPS_ALL=true

# REQUIRED - Redis with auth
REDIS_URL=redis://:password@host:port/db
REDIS_PASSWORD=<strong password>

# RECOMMENDED
CORS_ORIGINS=https://your-frontend.com
MAX_CONTENT_LENGTH_BYTES=16777216  # 16MB
IR_MAX_BYTES=2097152  # 2MB
NONCE_TTL_SECONDS=300  # 5 minutes
JWT_EXP_MINUTES=30  # Token lifetime

# OPTIONAL
SLACK_WEBHOOK_URL=<webhook for security alerts>
AUDIT_DB_PATH=/secure/path/to/audit.db
```

**Never commit secrets to version control!**

### Security Testing

Before deploying:

1. **Static Analysis**
   ```bash
   # Run security scanners
   bandit -r src/
   semgrep --config=auto src/
   
   # Check dependencies
   safety check
   pip-audit
   ```

2. **Dynamic Testing**
   ```bash
   # API security testing
   zap-cli quick-scan http://localhost:5000
   
   # Authentication testing
   pytest tests/security/
   ```

3. **Manual Testing**
   - Verify bootstrap key cannot be reused
   - Test rate limiting under load
   - Verify HTTPS enforcement
   - Test JWT revocation
   - Verify audit logging

## Compliance

### GDPR Considerations

- Audit logs may contain personal data (IP addresses, agent IDs)
- Implement data retention policies (see SECURITY_AUDIT_REPORT.md)
- Provide data export functionality for users
- Implement "right to be forgotten" where applicable

### SOC 2 Considerations

- Enable comprehensive audit logging
- Implement change management tracking
- Define disaster recovery procedures
- Document incident response plan

## Security Development Lifecycle

### For Contributors

When contributing code:

1. **Never commit:**
   - Secrets, API keys, passwords
   - Personal data or PII
   - Debug/test credentials

2. **Always:**
   - Validate all inputs
   - Use parameterized queries
   - Follow principle of least privilege
   - Add security tests for new features
   - Document security considerations

3. **Security Checklist:**
   - [ ] Input validation implemented
   - [ ] Output encoding where necessary
   - [ ] Authentication/authorization checked
   - [ ] Rate limiting considered
   - [ ] Error handling doesn't leak info
   - [ ] Secrets not in code
   - [ ] Dependencies are up to date
   - [ ] Security tests added

### Code Review Requirements

All PRs must:
- Pass automated security scans (Bandit, Semgrep)
- Have security review for auth/crypto changes
- Include security test cases
- Update security documentation

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers (with permission) in:
- Security advisories
- Release notes
- Hall of Fame (planned)

## References

- [OWASP Top 10](https://owasp.org/Top10/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)

## Changes

| Date       | Version | Changes |
|------------|---------|---------|
| 2025-11-19 | 1.3.0   | Initial security policy, addressed critical vulnerabilities |

---

**For questions about this security policy, contact: security@novatraxlabs.com**
