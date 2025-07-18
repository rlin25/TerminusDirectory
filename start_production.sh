#!/bin/bash

# Production Startup Script for Rental ML System
# This script starts all production services in the correct order

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PORT="${API_PORT:-8001}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
TIMEOUT=60

echo -e "${BLUE}🏠 Rental ML System - Production Startup${NC}"
echo -e "${BLUE}================================================${NC}"

# Function to check if a service is running
check_service() {
    local url=$1
    local name=$2
    local timeout=$3
    
    echo -e "${YELLOW}⏳ Waiting for $name to start...${NC}"
    
    for i in $(seq 1 $timeout); do
        if curl -s "$url" >/dev/null 2>&1; then
            echo -e "${GREEN}✅ $name is running${NC}"
            return 0
        fi
        sleep 1
        if [ $((i % 10)) -eq 0 ]; then
            echo -e "${YELLOW}⏳ Still waiting for $name... ($i/$timeout)${NC}"
        fi
    done
    
    echo -e "${RED}❌ $name failed to start within $timeout seconds${NC}"
    return 1
}

# Function to display service status
show_status() {
    echo -e "\n${BLUE}📊 Service Status${NC}"
    echo -e "${BLUE}=================${NC}"
    
    # Check Docker containers
    echo -e "\n${YELLOW}Docker Containers:${NC}"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(rental-ml|NAMES)"
    
    # Check processes
    echo -e "\n${YELLOW}Application Processes:${NC}"
    if pgrep -f "main_production.py" >/dev/null; then
        echo -e "${GREEN}✅ API Server (main_production.py)${NC}"
    else
        echo -e "${RED}❌ API Server (main_production.py)${NC}"
    fi
    
    if pgrep -f "streamlit.*app.py" >/dev/null; then
        echo -e "${GREEN}✅ Streamlit Demo${NC}"
    else
        echo -e "${RED}❌ Streamlit Demo${NC}"
    fi
    
    # Check service endpoints
    echo -e "\n${YELLOW}Service Health:${NC}"
    if curl -s "http://localhost:$API_PORT/" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ API Server (http://localhost:$API_PORT)${NC}"
    else
        echo -e "${RED}❌ API Server (http://localhost:$API_PORT)${NC}"
    fi
    
    if curl -s "http://localhost:$STREAMLIT_PORT/" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Streamlit Demo (http://localhost:$STREAMLIT_PORT)${NC}"
    else
        echo -e "${RED}❌ Streamlit Demo (http://localhost:$STREAMLIT_PORT)${NC}"
    fi
}

# Function to perform health checks
health_check() {
    echo -e "\n${BLUE}🏥 Health Check${NC}"
    echo -e "${BLUE}===============${NC}"
    
    # API Health Check
    if health_response=$(curl -s "http://localhost:$API_PORT/health/" 2>/dev/null); then
        echo -e "${GREEN}✅ API Health Check Successful${NC}"
        echo "$health_response" | python3 -m json.tool 2>/dev/null || echo "$health_response"
    else
        echo -e "${RED}❌ API Health Check Failed${NC}"
    fi
}

# Step 1: Check environment
echo -e "\n${YELLOW}1. Checking Environment...${NC}"
cd "$SCRIPT_DIR"

if [ ! -f ".env.production" ]; then
    echo -e "${RED}❌ .env.production file not found${NC}"
    exit 1
fi

if [ ! -f "main_production.py" ]; then
    echo -e "${RED}❌ main_production.py not found${NC}"
    exit 1
fi

# Source environment variables
set -a
source .env.production
set +a

echo -e "${GREEN}✅ Environment configured${NC}"

# Step 2: Start Docker services
echo -e "\n${YELLOW}2. Starting Docker Services...${NC}"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running${NC}"
    exit 1
fi

# Start databases
docker-compose up -d postgres redis

echo -e "${GREEN}✅ Database services starting...${NC}"

# Wait for databases to be ready
check_service "http://localhost:5432" "PostgreSQL" 30 || {
    echo -e "${YELLOW}⚠️ PostgreSQL health check failed, but continuing...${NC}"
}

check_service "http://localhost:6379" "Redis" 30 || {
    echo -e "${YELLOW}⚠️ Redis health check failed, but continuing...${NC}"
}

# Step 3: Start API Server
echo -e "\n${YELLOW}3. Starting API Server...${NC}"

# Kill any existing processes
pkill -f "main_production.py" 2>/dev/null || true

# Set Python path
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Start API server in background
nohup python3 main_production.py > /tmp/production_app.log 2>&1 &
API_PID=$!

# Wait for API to start
if check_service "http://localhost:$API_PORT/" "API Server" $TIMEOUT; then
    echo -e "${GREEN}✅ API Server started successfully (PID: $API_PID)${NC}"
else
    echo -e "${RED}❌ API Server failed to start${NC}"
    exit 1
fi

# Step 4: Start Streamlit Demo
echo -e "\n${YELLOW}4. Starting Streamlit Demo...${NC}"

# Kill any existing Streamlit processes
pkill -f "streamlit.*app.py" 2>/dev/null || true

# Start Streamlit in background
cd src/presentation/demo
nohup streamlit run app.py --server.port $STREAMLIT_PORT --server.address 0.0.0.0 > /tmp/streamlit_demo.log 2>&1 &
STREAMLIT_PID=$!
cd "$SCRIPT_DIR"

# Wait for Streamlit to start
if check_service "http://localhost:$STREAMLIT_PORT/" "Streamlit Demo" $TIMEOUT; then
    echo -e "${GREEN}✅ Streamlit Demo started successfully (PID: $STREAMLIT_PID)${NC}"
else
    echo -e "${YELLOW}⚠️ Streamlit Demo failed to start (non-critical)${NC}"
fi

# Step 5: Show status and perform health checks
show_status
health_check

# Step 6: Show access information
echo -e "\n${BLUE}🌐 Access Information${NC}"
echo -e "${BLUE}=====================${NC}"
echo -e "${GREEN}🔗 API Server:${NC} http://localhost:$API_PORT"
echo -e "${GREEN}📖 API Docs:${NC} http://localhost:$API_PORT/docs"
echo -e "${GREEN}🏥 Health Check:${NC} http://localhost:$API_PORT/health/"
echo -e "${GREEN}🎮 Streamlit Demo:${NC} http://localhost:$STREAMLIT_PORT"

echo -e "\n${BLUE}📋 Logs${NC}"
echo -e "${BLUE}========${NC}"
echo -e "${GREEN}API Logs:${NC} tail -f /tmp/production_app.log"
echo -e "${GREEN}Demo Logs:${NC} tail -f /tmp/streamlit_demo.log"

echo -e "\n${GREEN}🎉 Production system startup complete!${NC}"
echo -e "${YELLOW}💡 Note: Some features may be limited due to missing ML dependencies${NC}"
echo -e "${YELLOW}💡 Database schema may need updates for full functionality${NC}"

# Optional: Keep script running to monitor
if [ "${MONITOR:-false}" = "true" ]; then
    echo -e "\n${BLUE}🔍 Monitoring mode enabled. Press Ctrl+C to exit.${NC}"
    while true; do
        sleep 30
        echo -e "\n${YELLOW}$(date): Checking services...${NC}"
        show_status
    done
fi