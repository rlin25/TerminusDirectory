#!/bin/bash
set -e

# Scraping Service Entrypoint Script
# Handles browser setup, proxy configuration, and scraping environment initialization

echo "=== Scraping Service Initialization ==="

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to setup display for headless browsers
setup_display() {
    log "Setting up virtual display for browsers..."
    
    # Start Xvfb if not running
    if ! pgrep Xvfb > /dev/null; then
        Xvfb :99 -screen 0 ${SCREEN_WIDTH:-1920}x${SCREEN_HEIGHT:-1080}x${SCREEN_DEPTH:-24} -ac +extension GLX +render -noreset &
        export DISPLAY=:99
        sleep 2
        log "Virtual display started on :99"
    fi
}

# Function to setup browser configurations
setup_browsers() {
    log "Configuring browsers for scraping..."
    
    # Create browser profile directories
    mkdir -p /app/profiles/chrome
    mkdir -p /app/profiles/firefox
    mkdir -p /app/downloads
    mkdir -p /app/screenshots
    
    # Set browser preferences
    export CHROME_USER_DATA_DIR="/app/profiles/chrome"
    export FIREFOX_PROFILE_DIR="/app/profiles/firefox"
    export DOWNLOAD_DIR="/app/downloads"
    export SCREENSHOT_DIR="/app/screenshots"
    
    # Configure Chrome options
    export CHROME_OPTIONS="--no-sandbox --disable-dev-shm-usage --disable-gpu --headless --disable-extensions --disable-plugins --disable-images --disable-javascript --user-data-dir=/app/profiles/chrome"
    
    # Configure Firefox options
    export FIREFOX_OPTIONS="-headless -profile /app/profiles/firefox"
    
    log "Browser configurations completed"
}

# Function to validate browser installations
validate_browsers() {
    log "Validating browser installations..."
    
    # Check Chrome
    if command -v google-chrome &> /dev/null; then
        log "✓ Google Chrome is available"
        google-chrome --version
    else
        log "⚠ Google Chrome not found"
    fi
    
    # Check ChromeDriver
    if command -v chromedriver &> /dev/null; then
        log "✓ ChromeDriver is available"
        chromedriver --version
    else
        log "⚠ ChromeDriver not found"
    fi
    
    # Check Firefox
    if command -v firefox &> /dev/null; then
        log "✓ Firefox is available"
        firefox --version
    else
        log "⚠ Firefox not found"
    fi
    
    # Check Playwright browsers
    python -c "
try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browsers = ['chromium', 'firefox', 'webkit']
        for browser_name in browsers:
            try:
                browser = getattr(p, browser_name)
                browser.launch(headless=True).close()
                print(f'✓ Playwright {browser_name} is available')
            except Exception as e:
                print(f'⚠ Playwright {browser_name} failed: {e}')
except ImportError:
    print('⚠ Playwright not available')
"
}

# Function to setup proxy configuration
setup_proxy() {
    if [ -n "$PROXY_URL" ]; then
        log "Configuring proxy settings: $PROXY_URL"
        
        export HTTP_PROXY="$PROXY_URL"
        export HTTPS_PROXY="$PROXY_URL"
        export http_proxy="$PROXY_URL"
        export https_proxy="$PROXY_URL"
        
        # Add proxy to Chrome options
        export CHROME_OPTIONS="$CHROME_OPTIONS --proxy-server=$PROXY_URL"
        
        log "Proxy configuration completed"
    else
        log "No proxy configuration specified"
    fi
}

# Function to setup rate limiting
setup_rate_limiting() {
    log "Configuring rate limiting..."
    
    # Set default rate limiting values
    export SCRAPER_RATE_LIMIT="${SCRAPER_RATE_LIMIT:-10}"
    export SCRAPER_DELAY="${SCRAPER_DELAY:-1}"
    export SCRAPER_CONCURRENT_REQUESTS="${SCRAPER_CONCURRENT_REQUESTS:-5}"
    
    # Configure Scrapy settings
    export SCRAPY_SETTINGS="
CONCURRENT_REQUESTS=${SCRAPER_CONCURRENT_REQUESTS}
DOWNLOAD_DELAY=${SCRAPER_DELAY}
RANDOMIZE_DOWNLOAD_DELAY=0.5
AUTOTHROTTLE_ENABLED=True
AUTOTHROTTLE_START_DELAY=1
AUTOTHROTTLE_MAX_DELAY=10
AUTOTHROTTLE_TARGET_CONCURRENCY=2.0
"
    
    log "Rate limiting configured - Limit: ${SCRAPER_RATE_LIMIT}/sec, Delay: ${SCRAPER_DELAY}s"
}

# Function to validate scraping dependencies
validate_scraping_dependencies() {
    log "Validating scraping dependencies..."
    
    python -c "
import sys
import importlib

required_packages = [
    'selenium', 'beautifulsoup4', 'scrapy', 'playwright',
    'requests', 'aiohttp', 'fake_useragent', 'cloudscraper'
]

missing_packages = []
for package in required_packages:
    try:
        if package == 'beautifulsoup4':
            importlib.import_module('bs4')
        elif package == 'fake_useragent':
            importlib.import_module('fake_useragent')
        else:
            importlib.import_module(package)
        print(f'✓ {package} imported successfully')
    except ImportError:
        missing_packages.append(package)
        print(f'✗ {package} is missing')

if missing_packages:
    print(f'Missing packages: {missing_packages}')
    sys.exit(1)
else:
    print('All required scraping packages are available')
"
    
    if [ $? -eq 0 ]; then
        log "All scraping dependencies validated successfully"
    else
        log "ERROR: Missing required scraping dependencies"
        exit 1
    fi
}

# Function to test browser functionality
test_browser_functionality() {
    log "Testing browser functionality..."
    
    # Test Selenium with Chrome
    python -c "
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

try:
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    
    driver = webdriver.Chrome(options=options)
    driver.get('https://httpbin.org/status/200')
    status = driver.title
    driver.quit()
    print('✓ Selenium Chrome test passed')
except Exception as e:
    print(f'✗ Selenium Chrome test failed: {e}')
"
    
    # Test Playwright
    python -c "
try:
    from playwright.sync_api import sync_playwright
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto('https://httpbin.org/status/200')
        browser.close()
        print('✓ Playwright test passed')
except Exception as e:
    print(f'✗ Playwright test failed: {e}')
"
}

# Function to setup GDPR compliance
setup_gdpr_compliance() {
    log "Setting up GDPR compliance features..."
    
    # Set compliance environment variables
    export GDPR_ENABLED="${GDPR_ENABLED:-true}"
    export RESPECT_ROBOTS_TXT="${RESPECT_ROBOTS_TXT:-true}"
    export USER_AGENT_ROTATION="${USER_AGENT_ROTATION:-true}"
    export DATA_RETENTION_DAYS="${DATA_RETENTION_DAYS:-30}"
    
    # Create compliance directories
    mkdir -p /app/data/compliance/logs
    mkdir -p /app/data/compliance/requests
    
    log "GDPR compliance setup completed"
}

# Function to setup monitoring
setup_monitoring() {
    log "Setting up scraping monitoring..."
    
    # Create monitoring directories
    mkdir -p /app/logs/scraping
    mkdir -p /app/logs/errors
    mkdir -p /app/logs/metrics
    
    # Set monitoring environment variables
    export SCRAPING_METRICS_ENABLED="${SCRAPING_METRICS_ENABLED:-true}"
    export ERROR_REPORTING_ENABLED="${ERROR_REPORTING_ENABLED:-true}"
    export PERFORMANCE_MONITORING="${PERFORMANCE_MONITORING:-true}"
    
    log "Scraping monitoring setup completed"
}

# Function to run database connectivity check
check_database_connectivity() {
    log "Checking database connectivity for scraped data storage..."
    
    python -c "
import sys
import os
sys.path.append('/app/src')

try:
    from infrastructure.data.config import get_database_url
    from sqlalchemy import create_engine, text
    
    engine = create_engine(get_database_url())
    with engine.connect() as conn:
        result = conn.execute(text('SELECT 1'))
        print('Database connection successful')
        
except Exception as e:
    print(f'Database connection failed: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        log "Database connectivity verified"
    else
        log "ERROR: Database connectivity check failed"
        exit 1
    fi
}

# Function to perform health check
health_check() {
    log "Performing scraping service health check..."
    
    # Check if scraping modules exist
    if [ ! -d "/app/src/infrastructure/scrapers" ]; then
        log "ERROR: Scraping modules not found"
        exit 1
    fi
    
    # Test scraping framework imports
    python -c "
import sys
sys.path.append('/app/src')
from infrastructure.scrapers.production_scraping_orchestrator import *
print('Scraping framework imports successful')
"
    
    if [ $? -eq 0 ]; then
        log "Health check passed"
    else
        log "ERROR: Health check failed"
        exit 1
    fi
}

# Function to cleanup on shutdown
cleanup() {
    log "Received shutdown signal, cleaning up scraping service..."
    
    # Kill browser processes
    pkill -f chrome || true
    pkill -f firefox || true
    pkill -f chromium || true
    pkill -f Xvfb || true
    
    # Clean up temporary files
    rm -rf /tmp/chrome-* || true
    rm -rf /tmp/firefox-* || true
    rm -rf /tmp/.org.chromium.* || true
    
    log "Scraping service cleanup completed"
    exit 0
}

# Main initialization sequence
main() {
    log "Starting Scraping Service..."
    
    # Run initialization steps
    setup_display
    setup_browsers
    validate_browsers
    setup_proxy
    setup_rate_limiting
    validate_scraping_dependencies
    test_browser_functionality
    setup_gdpr_compliance
    setup_monitoring
    check_database_connectivity
    health_check
    
    log "Scraping Service initialization completed"
    log "Executing command: $@"
    
    # Execute the main command
    exec "$@"
}

# Set signal handlers
trap cleanup SIGTERM SIGINT

# Run main function
main "$@"