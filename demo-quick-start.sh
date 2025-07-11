#!/bin/bash

# Rental ML System - Demo Quick Start Script
# This script provides a one-command setup and launch for the Streamlit demo

set -e

echo "🏠 Rental ML System - Demo Quick Start"
echo "======================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python $PYTHON_VERSION is installed, but Python $REQUIRED_VERSION or higher is required."
    exit 1
fi

print_status "Python $PYTHON_VERSION is installed"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install pip."
    exit 1
fi

print_status "pip is available"

# Navigate to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

print_info "Working directory: $PROJECT_DIR"

# Check if we're in the right directory
if [ ! -f "src/presentation/demo/app.py" ]; then
    print_error "Demo application not found. Please run this script from the project root directory."
    exit 1
fi

print_status "Demo application found"

# Create virtual environment if it doesn't exist
VENV_DIR="demo-venv"
if [ ! -d "$VENV_DIR" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install base requirements
print_info "Installing base requirements..."
if [ -f "requirements/base.txt" ]; then
    pip install -r requirements/base.txt > /dev/null 2>&1
    print_status "Base requirements installed"
else
    print_warning "Base requirements file not found, proceeding with demo requirements only"
fi

# Install demo-specific requirements
print_info "Installing demo requirements..."
if [ -f "src/presentation/demo/requirements-demo.txt" ]; then
    pip install -r src/presentation/demo/requirements-demo.txt > /dev/null 2>&1
    print_status "Demo requirements installed"
else
    print_error "Demo requirements file not found"
    exit 1
fi

# Set environment variables
export DEMO_PROPERTY_COUNT=100
export DEMO_USER_COUNT=50
export RANDOM_SEED=42

print_status "Environment variables configured"

# Function to handle cleanup on exit
cleanup() {
    print_info "Cleaning up..."
    deactivate 2>/dev/null || true
    exit 0
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Parse command line arguments
PORT=8501
HOST="localhost"
DEBUG=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -d|--debug)
            DEBUG=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -p, --port PORT    Port to run on (default: 8501)"
            echo "  -h, --host HOST    Host to bind to (default: localhost)"
            echo "  -d, --debug        Enable debug mode"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            print_warning "Unknown option: $1"
            shift
            ;;
    esac
done

# Check if port is available
if command -v netstat &> /dev/null; then
    if netstat -tuln | grep -q ":$PORT "; then
        print_warning "Port $PORT is already in use. Trying to find an available port..."
        for try_port in $(seq $((PORT + 1)) $((PORT + 10))); do
            if ! netstat -tuln | grep -q ":$try_port "; then
                PORT=$try_port
                print_info "Using port $PORT instead"
                break
            fi
        done
    fi
fi

# Launch the demo
print_status "All dependencies installed successfully!"
echo ""
print_info "Launching Rental ML System Demo..."
print_info "URL: http://$HOST:$PORT"
print_info "Press Ctrl+C to stop the demo"
echo ""
print_info "Features available:"
echo "  • Property Search with Advanced Filters"
echo "  • ML-Powered Recommendations"  
echo "  • User Preference Management"
echo "  • Analytics Dashboard"
echo "  • Performance Monitoring"
echo "  • Property Comparison Tools"
echo "  • Market Insights"
echo ""
echo "===========================================" 

# Change to demo directory and launch
cd src/presentation/demo

# Build streamlit command
STREAMLIT_CMD="streamlit run app.py --server.port $PORT --server.address $HOST"

if [ "$DEBUG" = true ]; then
    STREAMLIT_CMD="$STREAMLIT_CMD --server.runOnSave true"
    print_info "Debug mode enabled - auto-reload on file changes"
else
    STREAMLIT_CMD="$STREAMLIT_CMD --server.headless true"
fi

# Launch the application
eval $STREAMLIT_CMD