# Sandbox for the test-coding agent's pytest runs (agents/coder/sandbox.py runs it with --network none, read-only).
# Build:  docker build -t multi-agents/coder-sandbox:1 sandbox/coder
# Contents follow gpu-validation's requirements.txt; when that file changes, bump the tag here and in
# config/coder.yaml (sandbox.image) and rebuild.
FROM python:3.12-slim
RUN pip install --no-cache-dir "pytest>=8" "pyyaml>=6" \
 && python -c "import pytest, yaml"
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /w
