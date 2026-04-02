# SRE Incident Response - OpenEnv Environment

See [sre_env/README.md](sre_env/README.md) for full documentation.

## Quick Start

```bash
pip install -e sre_env/
python inference.py
```

## Docker

```bash
docker build -t sre-env .
docker run -p 8000:8000 sre-env
```
