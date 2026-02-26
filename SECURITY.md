# Security Policy

## Supported Versions

We actively support the following versions of NeMo with security updates:

| Version | Supported          |
| ------- | ------------------ |
| latest (main) | ✅ Active support |
| stable  | ✅ Active support |
| < stable | ❌ End of life |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you believe you have found a security vulnerability in NeMo, please report it responsibly by emailing **security@nvidia.com** or by using [NVIDIA's Product Security Incident Response Team (PSIRT)](https://www.nvidia.com/en-us/security/).

### What to Include

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

- **Acknowledgment**: We will acknowledge receipt of your vulnerability report within **48 hours**.
- **Status Update**: We will send a more detailed response within **7 days** indicating next steps.
- **Resolution**: We aim to resolve critical vulnerabilities within **30 days** of confirmation.

## Security Best Practices for NeMo Users

### Model Loading

As of PyTorch 2.6, `torch.load` defaults to `weights_only=True`. When loading model checkpoints:

```python
# Preferred (safe): load weights only
model = torch.load("checkpoint.pt", weights_only=True)

# Use with caution: only for trusted checkpoints
# TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 environment variable
model = torch.load("trusted_checkpoint.pt", weights_only=False)
```

**Never load checkpoints from untrusted sources with `weights_only=False`**, as this can lead to arbitrary code execution.

### Container Security

When running NeMo Docker containers:

- Pull containers only from official NVIDIA NGC registry (`nvcr.io/nvidia/nemo`)
- Verify container image digests before use
- Run containers with least-privilege principles
- Do not expose GPU containers to public networks without authentication

### API Keys and Credentials

- Never hard-code API keys or credentials in scripts or notebooks
- Use environment variables or secrets management tools
- Rotate credentials regularly
- Use `.env` files locally and never commit them (`.gitignore` is configured for this)

### Data Privacy

- Be cautious about training on or fine-tuning with personally identifiable information (PII)
- Follow your organization's data governance policies
- Consider using differential privacy techniques for sensitive datasets

## Disclosure Policy

We follow coordinated vulnerability disclosure. Once a fix is available, we will:

1. Release a patched version
2. Publish a security advisory on this repository
3. Credit the reporter (unless they prefer to remain anonymous)
