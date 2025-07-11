#!/usr/bin/env python3
"""
Streamlit Demo Launcher Script

This script provides a convenient way to launch the Rental ML System demo
with proper configuration and error handling.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import plotly
        import pandas
        import numpy
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements-demo.txt")
        return False

def check_project_structure():
    """Verify the project structure is correct"""
    current_dir = Path(__file__).parent
    required_files = [
        "app.py",
        "components.py", 
        "sample_data.py",
        "utils.py",
        "config.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not (current_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ Project structure is correct")
    return True

def launch_demo(port=8501, host="localhost", debug=False):
    """Launch the Streamlit demo application"""
    
    # Change to the demo directory
    demo_dir = Path(__file__).parent
    os.chdir(demo_dir)
    
    # Build the streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(port),
        "--server.address", host,
        "--server.runOnSave", str(debug).lower(),
        "--server.fileWatcherType", "poll" if debug else "auto"
    ]
    
    if not debug:
        cmd.extend(["--server.headless", "true"])
    
    print(f"🚀 Launching Rental ML System Demo...")
    print(f"📍 URL: http://{host}:{port}")
    print(f"🔧 Debug mode: {'enabled' if debug else 'disabled'}")
    print("=" * 50)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching demo: {e}")
        sys.exit(1)

def setup_environment():
    """Setup environment variables for the demo"""
    
    # Set default environment variables if not already set
    env_defaults = {
        "DEMO_PROPERTY_COUNT": "100",
        "DEMO_USER_COUNT": "50", 
        "RANDOM_SEED": "42"
    }
    
    for key, value in env_defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"🔧 Set {key}={value}")

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Launch Rental ML System Demo")
    parser.add_argument("--port", "-p", type=int, default=8501, 
                       help="Port to run the demo on (default: 8501)")
    parser.add_argument("--host", "-h", default="localhost",
                       help="Host to bind to (default: localhost)")
    parser.add_argument("--debug", "-d", action="store_true",
                       help="Enable debug mode with auto-reload")
    parser.add_argument("--check-only", "-c", action="store_true",
                       help="Only check requirements and structure, don't launch")
    parser.add_argument("--setup", "-s", action="store_true",
                       help="Setup environment and install requirements")
    
    args = parser.parse_args()
    
    print("🏠 Rental ML System - Demo Launcher")
    print("=" * 40)
    
    # Setup mode
    if args.setup:
        print("🔧 Setting up demo environment...")
        
        # Install requirements
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements-demo.txt"
            ], check=True)
            print("✅ Requirements installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install requirements")
            sys.exit(1)
        
        # Setup environment
        setup_environment()
        print("✅ Setup completed successfully")
        return
    
    # Pre-flight checks
    if not check_project_structure():
        sys.exit(1)
    
    if not check_requirements():
        print("💡 Try running: python run_demo.py --setup")
        sys.exit(1)
    
    # Check-only mode
    if args.check_only:
        print("✅ All checks passed - demo is ready to launch")
        return
    
    # Setup environment
    setup_environment()
    
    # Launch the demo
    launch_demo(port=args.port, host=args.host, debug=args.debug)

if __name__ == "__main__":
    main()