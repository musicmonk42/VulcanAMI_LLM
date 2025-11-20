# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.3.x   | :white_check_mark: |
| < 1.3   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

### For Novatrax Labs Customers and Partners

If you discover a security vulnerability in Graphix Vulcan, please report it to your designated Novatrax Labs account team or security contact as specified in your service agreement.

### For Security Researchers

If you believe you have found a security vulnerability, please follow our responsible disclosure process:

1. **Do Not** disclose the vulnerability publicly until it has been addressed
2. **Do Not** exploit the vulnerability beyond what is necessary to demonstrate it
3. Contact: security@novatrax.com (or the designated security contact in your agreement)
4. Provide detailed information about the vulnerability:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested remediation (if any)

### What to Expect

- **Initial Response:** Within 48 hours
- **Status Update:** Within 5 business days
- **Resolution Timeline:** Varies based on severity
  - Critical: 7-14 days
  - High: 30 days
  - Medium: 60 days
  - Low: 90 days

## Security Measures

### Authentication & Authorization
- JWT-based authentication with configurable expiration
- Multi-algorithm cryptographic key support (RSA, Ed25519, ECDSA)
- Role-based access control (RBAC)
- Certificate support for enhanced security

### Data Protection
- Comprehensive audit logging with tamper-evident storage
- Encryption support for sensitive data
- Secure configuration management
- Input validation and sanitization

### Infrastructure Security
- Rate limiting with Redis backend
- HTTPS enforcement options
- SQL injection prevention via ORM
- Command injection prevention
- Pickle deserialization protection (see SECURITY_FIXES_PICKLE.md)

### Secure Development
- Regular dependency vulnerability scanning
- Security-focused code reviews
- Automated security testing
- Comprehensive security audit documentation

## Security Updates

Security patches and updates are distributed through:
- Your Novatrax Labs account team
- Private security advisories (for customers)
- Release notes and changelogs

## Security Best Practices

### For Deployment

1. **Secrets Management**
   - Never commit secrets to version control
   - Use environment variables or secret management services
   - Rotate secrets regularly
   - Use strong, randomly generated values

2. **Network Security**
   - Deploy behind authenticated gateways
   - Use TLS/HTTPS for all production traffic
   - Implement network segmentation
   - Configure firewalls appropriately

3. **Access Control**
   - Follow principle of least privilege
   - Implement MFA where possible
   - Regular access audits
   - Secure SSH/remote access

4. **Monitoring**
   - Enable comprehensive logging
   - Monitor for security events
   - Set up alerting for suspicious activity
   - Regular log review

5. **Updates**
   - Keep dependencies up to date
   - Apply security patches promptly
   - Test updates in staging first
   - Have rollback procedures ready

### For Development

1. **Code Security**
   - Follow secure coding guidelines
   - Review OWASP Top 10
   - Use provided security utilities (e.g., `src.utils.secure_pickle`)
   - Implement input validation
   - Handle errors securely

2. **Dependencies**
   - Audit new dependencies before adding
   - Use lock files (requirements.txt with versions)
   - Scan for vulnerabilities regularly
   - Keep dependencies minimal

3. **Testing**
   - Write security-focused tests
   - Include negative test cases
   - Test authentication and authorization
   - Validate input sanitization

4. **Documentation**
   - Document security considerations
   - Update security documentation with changes
   - Include security warnings where appropriate

## Known Security Considerations

### Pickle Deserialization
**Risk Level:** HIGH  
**Status:** Mitigated with utilities

Python's pickle module is used in some legacy code. We provide secure alternatives:
- Use `src.utils.secure_pickle.RestrictedUnpickler` for untrusted data
- Use `src.utils.secure_pickle.SecurePickle` for trusted data with integrity protection

See `SECURITY_FIXES_PICKLE.md` for detailed guidance.

### Rate Limiting
**Risk Level:** MEDIUM  
**Status:** Implemented with fallback

Production deployments should use Redis for rate limiting. In-memory fallback is only suitable for development.

### CORS Configuration
**Risk Level:** MEDIUM  
**Status:** Configurable

Ensure CORS origins are explicitly whitelisted in production. Do not use wildcard (`*`) origins.

## Compliance

This software is designed to support various compliance requirements:
- Audit logging for compliance tracking
- Encryption capabilities for data protection
- Access control mechanisms
- Tamper-evident audit trails

Specific compliance certifications and attestations are available to enterprise customers through Novatrax Labs.

## Security Audit

A comprehensive security audit was conducted on 2025-11-20. Key findings:
- **Critical Vulnerabilities:** 0
- **High Severity:** 1 (mitigated with secure pickle utilities)
- **Medium Severity:** 8 (documented with recommendations)
- **Low Severity:** 12 (best practice improvements)

Full audit report: `SECURITY_AUDIT_REPORT.md`

## Security Contact

For security concerns, questions, or reports:
- Email: security@novatrax.com
- Enterprise customers: Contact your account team
- Response time: 48 hours for initial response

## Acknowledgments

We appreciate responsible disclosure from security researchers. Contributors may be acknowledged in release notes and security advisories (with permission).

## Changes to This Policy

This security policy may be updated periodically. Check the last updated date below.

---

**Last Updated:** 2025-11-20  
**Version:** 1.0  
**Contact:** security@novatrax.com
