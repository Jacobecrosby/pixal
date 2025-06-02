#!/bin/bash

set -e

PYTHON_VERSION=3.10.9

# Detect OS
IS_MAC=false
IS_WINDOWS=false

case "$(uname -s)" in
    Darwin)
        IS_MAC=true
        ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
        IS_WINDOWS=true
        ;;
    Linux)
        # Nothing special
        ;;
    *)
        echo "❌ Unsupported OS: $(uname -s)"
        exit 1
        ;;
esac

# Check if pyenv is installed
if ! command -v pyenv &>/dev/null; then
    echo "❌ pyenv is not installed. Please install it first:"
    echo "   macOS: brew install pyenv"
    exit 1
fi

# Install Python 3.10.9 if missing
if ! pyenv versions --bare | grep -qx "$PYTHON_VERSION"; then
    echo "⬇️  Installing Python $PYTHON_VERSION via pyenv..."
    pyenv install "$PYTHON_VERSION"
fi

# Set local version
pyenv local "$PYTHON_VERSION"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Rehash to ensure correct Python path
pyenv rehash

# Confirm Python version
PYTHON_CMD="$(pyenv which python)"
echo "🐍 Using Python: $($PYTHON_CMD --version)"

# Create virtual environment
echo "🔧 Creating virtual environment in .venv/"
$PYTHON_CMD -m venv .venv

# Activate virtual environment
echo "🐍 Activating virtual environment"
if $IS_WINDOWS; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

echo "⬆️  Upgrading pip"
python -m pip install --upgrade pip

echo "📦 Installing base PIXAL dependencies"
pip install -e .

# Platform-specific extras
if $IS_MAC; then
    echo "🍎 Adding macOS-specific dependencies"
    pip install -e '.[mac]'
elif $IS_WINDOWS; then
    echo "🪟 Adding Windows-specific dependencies"
    pip install -e '.[gpu]'
else
    echo "🐧 Adding Linux GPU dependencies"
    pip install -e '.[gpu]'
fi

echo "✅ PIXAL setup complete!"
echo "👉 To activate your environment later, run:"
if $IS_WINDOWS; then
    echo "   source .venv/Scripts/activate"
else
    echo "   source .venv/bin/activate"
fi