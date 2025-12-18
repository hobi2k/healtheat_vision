#!/bin/bash

# Ensure we are in the project root
cd "$(dirname "$0")/.."

echo "Creating virtual environment..."
uv venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
uv pip install -r requirements.txt

echo "Setup complete! Activate the environment with: source .venv/bin/activate"
