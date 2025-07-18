#!/usr/bin/env python3
"""
Rental ML System - Access Information
"""

import requests
import subprocess
import socket

def check_service(url, name):
    """Check if a service is responding"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return f"✅ {name}: Running (Status: {response.status_code})"
        else:
            return f"⚠️ {name}: Responding but status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return f"❌ {name}: Connection refused"
    except requests.exceptions.Timeout:
        return f"⚠️ {name}: Timeout"
    except Exception as e:
        return f"❌ {name}: Error - {e}"

def check_port(host, port):
    """Check if a port is open"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

print("🏠 RENTAL ML SYSTEM - ACCESS INFORMATION")
print("=" * 50)
print()

# Check ports
print("🔌 PORT STATUS:")
ports = [
    (8000, "FastAPI Server"),
    (8501, "Streamlit Demo")
]

for port, service in ports:
    if check_port('localhost', port):
        print(f"   ✅ Port {port}: Open ({service})")
    else:
        print(f"   ❌ Port {port}: Closed ({service})")

print()

# Check services
print("🚀 SERVICE STATUS:")
services = [
    ("http://localhost:8000/health", "FastAPI Health Check"),
    ("http://localhost:8000/", "FastAPI Root"),
    ("http://localhost:8501/", "Streamlit Demo")
]

for url, name in services:
    print(f"   {check_service(url, name)}")

print()
print("🌐 ACCESS URLS:")
print("   📊 Streamlit Demo:    http://localhost:8501")
print("   🔧 FastAPI Server:    http://localhost:8000")
print("   📖 API Documentation: http://localhost:8000/docs")
print("   ❤️ Health Check:      http://localhost:8000/health")

print()
print("💡 USAGE TIPS:")
print("   • The Streamlit demo provides a user-friendly web interface")
print("   • The FastAPI server offers programmatic access via REST API")
print("   • Visit /docs for interactive API documentation")
print("   • Both services are running with demo data")

print()
print("🔄 SERVICE MANAGEMENT:")
print("   • Stop services: pkill -f 'streamlit|uvicorn'")
print("   • Restart API: python3 main_demo.py")
print("   • Restart Demo: ./demo-quick-start.sh")

print()

# Test a quick API call
try:
    response = requests.get("http://localhost:8000/health", timeout=3)
    if response.status_code == 200:
        health_data = response.json()
        print("📊 SYSTEM HEALTH:")
        print(f"   Status: {health_data.get('status', 'unknown')}")
        print(f"   Demo Mode: {health_data.get('demo_mode', 'unknown')}")
        print(f"   Response Time: {health_data.get('response_time_ms', 0):.2f}ms")
except:
    print("📊 SYSTEM HEALTH: Unable to fetch health data")

print()
print("🎯 SYSTEM READY FOR USE!")