#!/bin/bash

# Exit on error
set -e

# Detect OS
IS_MAC=false
IS_WINDOWS=false

case "$(uname -s)" in
    Darwin)
        IS_MAC=true
        ;;
    Linux)
        # Nothing special needed for Linux
        ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
        IS_WINDOWS=true
        ;;
    *)
        echo "Unsupported OS: $(uname -s)"
        exit 1
        ;;
esac

# Determine python executable
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
else
    echo "❌ Python not found. Please install Python 3.x first."
    exit 1
fi

# Detect OS
IS_MAC=false
IS_WINDOWS=false

case "$(uname -s)" in
    Darwin)
        IS_MAC=true
        ;;
    Linux)
        # Nothing special needed for Linux
        ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
        IS_WINDOWS=true
        ;;
    *)
        echo "Unsupported OS: $(uname -s)"
        exit 1
        ;;
esac

# Determine python executable
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
else
    echo "❌ Python not found. Please install Python 3.x first."
    exit 1
fi

echo "🔧 Creating virtual environment in .venv/"
$PYTHON_CMD -m venv .venv
$PYTHON_CMD -m venv .venv

echo "🐍 Activating virtual environment"
if $IS_WINDOWS; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi
if $IS_WINDOWS; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

echo "⬆️  Upgrading pip"
python -m pip install --upgrade pip
python -m pip install --upgrade pip

echo "📦 Installing PIXAL in editable mode"
pip install -e .

echo "✅ PIXAL setup complete!"
echo "👉 To activate your environment later, run:"

if $IS_WINDOWS; then
    echo "   source .venv/Scripts/activate"
else
    echo "   source .venv/bin/activate"
fi