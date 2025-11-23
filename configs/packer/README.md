# Packer Configurations

This directory contains HashiCorp Packer configurations for building machine images for VulcanAMI.

## Contents

- `packer.toml` - Main Packer configuration file
- `semver.txt` - Semantic versioning for image builds

## Purpose

Packer configurations enable:
- **Automated image builds**: Create reproducible machine images
- **Multi-platform support**: Build for AWS, Azure, GCP, etc.
- **Consistent environments**: Ensure dev, staging, and production parity
- **Immutable infrastructure**: Deploy pre-configured, tested images

## Configuration Structure

The `packer.toml` file defines:
- **Source images**: Base AMIs or images to build from
- **Provisioners**: Scripts and tools to install
- **Post-processors**: Image artifact handling
- **Build variables**: Parameterized configuration

## Building Images

To build an image:

```bash
# Validate configuration
packer validate packer.toml

# Build image
packer build packer.toml

# Build with variables
packer build -var 'version=1.2.3' packer.toml
```

## Versioning

The `semver.txt` file tracks image versions using semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking changes, major updates
- **MINOR**: New features, backwards-compatible
- **PATCH**: Bug fixes, security patches

## Image Contents

Typical image includes:
- Operating system (Ubuntu, Amazon Linux, etc.)
- Python runtime and dependencies
- VulcanAMI application code
- System configurations
- Monitoring agents
- Security hardening

## Provisioning Steps

1. **System updates**: Patch OS and install base packages
2. **Dependencies**: Install Python, libraries, tools
3. **Application**: Copy and configure VulcanAMI code
4. **Configuration**: Set up configs, environment variables
5. **Security**: Harden system, configure firewalls
6. **Cleanup**: Remove temporary files, clear caches

## Cloud-Specific Configurations

### AWS
- AMI naming and tagging
- Instance metadata integration
- EBS volume configuration
- IAM role attachment

### Azure
- Managed image creation
- Resource group placement
- Virtual network configuration

### GCP
- Image family management
- Project and zone settings
- Custom metadata

## Testing

Images should be tested:
- **Smoke tests**: Basic functionality checks
- **Integration tests**: Full system validation
- **Security scans**: Vulnerability assessment
- **Performance tests**: Resource utilization

## CI/CD Integration

Packer builds are triggered:
- On tagged releases
- On scheduled builds (monthly security updates)
- On manual request for testing

## Security Considerations

- Use minimal base images to reduce attack surface
- Apply security patches during build
- Scan images for vulnerabilities
- Rotate credentials and secrets
- Use encrypted volumes
- Enable security groups/firewalls

## Best Practices

1. **Idempotent provisioning**: Scripts should be re-runnable
2. **Minimal changes**: Only install what's needed
3. **Version everything**: Tag images with versions and metadata
4. **Test thoroughly**: Validate images before deployment
5. **Document changes**: Update changelog and README

## Troubleshooting

Common issues:
- **Build timeouts**: Increase timeout settings
- **Network errors**: Check firewall and proxy settings
- **Permission errors**: Verify cloud credentials and IAM roles
- **Provisioning failures**: Check logs in `/var/log/`

## Version

Current Packer configuration version: 1.0.0

## References

- [Packer Documentation](https://www.packer.io/docs)
- [Packer Best Practices](https://www.packer.io/guides/hcl)
- [AWS AMI Best Practices](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html)
