#!/bin/bash

# Game Theory Iran Model - Quick Start Script

echo "🎮 Game Theory Model: Iran-Israel-US Strategic Analysis"
echo "========================================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
uv sync

# Create graphics directory
mkdir -p graphics

echo ""
echo "🚀 Choose how to run the application:"
echo ""
echo "1. 🌐 Interactive Web Interface (Streamlit)"
echo "2. 💻 Command Line Analysis"
echo "3. 🎨 Export Graphics for X Posts"
echo "4. 🐳 Run with Docker"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "🌐 Starting Streamlit web interface..."
        echo "   Access at: http://localhost:8501"
        uv run streamlit run frontend/app.py
        ;;
    2)
        echo "💻 Running command line analysis..."
        uv run python run_model.py --analyze-strategies
        ;;
    3)
        echo "🎨 Exporting graphics for X posts..."
        uv run python run_model.py --export-graphics
        echo "✅ Graphics exported to ./graphics/"
        ;;
    4)
        echo "🐳 Building and running with Docker..."
        docker-compose up --build
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac