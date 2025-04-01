#!/bin/bash

# Exit on error
set -e

echo "🔧 Creating virtual environment in .venv/"
python3 -m venv .venv

echo "🐍 Activating virtual environment"
source .venv/bin/activate

echo "⬆️  Upgrading pip"
pip install --upgrade pip

echo "📦 Installing PIXAL in editable mode"
pip install -e .

echo "✅ PIXAL setup complete!"
echo "👉 To activate your environment later, run:"
echo "   source .venv/bin/activate"
