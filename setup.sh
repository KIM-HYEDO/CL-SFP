#!/usr/bin/env bash
# One-shot setup for streaming-flow-policy on a new machine.
# Usage: bash setup.sh
set -e

echo "==> Installing dependencies with uv..."
uv sync

echo "==> Applying post-install patches..."
uv run python scripts/apply_patches.py

echo ""
echo "Setup complete. Activate the env with: source .venv/bin/activate"
