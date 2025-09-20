#!/bin/bash

# Render Build Script
echo "🚀 Starting Render build..."

# Install system dependencies
echo "📦 Installing system dependencies..."
apt-get update
apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/uploads data/vector_db

echo "✅ Build completed successfully!"