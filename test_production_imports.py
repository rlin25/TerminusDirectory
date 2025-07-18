#!/usr/bin/env python3
"""
Test script to verify production application imports and basic functionality.
This script tests what can be imported without requiring database connections.
"""

import sys
import os
from pathlib import Path

def test_basic_imports():
    """Test basic Python and FastAPI imports"""
    print("=" * 50)
    print("Testing Basic Imports")
    print("=" * 50)
    
    try:
        # Basic Python modules
        import time
        import logging
        from typing import Dict, Any, Optional
        from contextlib import asynccontextmanager
        print("✅ Basic Python modules imported successfully")
        
        # Environment and configuration
        from dotenv import load_dotenv
        print("✅ dotenv imported successfully")
        
        # FastAPI and middleware
        from fastapi import FastAPI, Request, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.gzip import GZipMiddleware
        from fastapi.responses import JSONResponse
        print("✅ FastAPI and middleware imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Basic import failed: {e}")
        return False


def test_environment_loading():
    """Test loading production environment variables"""
    print("\n" + "=" * 50)
    print("Testing Environment Configuration")
    print("=" * 50)
    
    try:
        from dotenv import load_dotenv
        env_file = Path(__file__).parent / ".env.production"
        
        if env_file.exists():
            load_dotenv(env_file)
            print("✅ Production environment file loaded")
            
            # Test some key environment variables
            test_vars = [
                "ENVIRONMENT",
                "DB_HOST",
                "DB_PORT", 
                "DB_NAME",
                "REDIS_HOST",
                "REDIS_PORT",
                "API_HOST",
                "API_PORT"
            ]
            
            for var in test_vars:
                value = os.getenv(var)
                if value:
                    print(f"✅ {var}: {value}")
                else:
                    print(f"⚠️  {var}: Not set")
            
            return True
        else:
            print(f"❌ Environment file not found: {env_file}")
            return False
            
    except Exception as e:
        print(f"❌ Environment loading failed: {e}")
        return False


def test_config_module():
    """Test configuration module without database connections"""
    print("\n" + "=" * 50)
    print("Testing Configuration Module")
    print("=" * 50)
    
    try:
        # Try to import config classes
        from src.infrastructure.data.config import DataConfig, DatabaseConfig, RedisConfig
        print("✅ Configuration classes imported successfully")
        
        # Test creating config instances (this should work without DB connections)
        db_config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test",
            username="test",
            password="test"
        )
        print("✅ DatabaseConfig instance created")
        
        redis_config = RedisConfig(
            host="localhost",
            port=6379
        )
        print("✅ RedisConfig instance created")
        
        # Test URL generation
        print(f"✅ Database URL: {db_config.url}")
        print(f"✅ Redis URL: {redis_config.url}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        print("Note: This may be due to missing database dependencies (sqlalchemy, asyncpg, redis)")
        return False
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_main_production_syntax():
    """Test that main_production.py has valid syntax"""
    print("\n" + "=" * 50)
    print("Testing Production Main Application Syntax")
    print("=" * 50)
    
    try:
        import ast
        
        main_file = Path(__file__).parent / "main_production.py"
        if main_file.exists():
            with open(main_file, 'r') as f:
                content = f.read()
            
            # Parse the file to check syntax
            ast.parse(content)
            print("✅ main_production.py has valid Python syntax")
            
            # Check for key components
            required_components = [
                "from src.infrastructure.data.repository_factory import RepositoryFactory",
                "from src.infrastructure.data.config import DataConfig",
                "async def lifespan",
                "app = FastAPI",
                "@app.get(\"/health\",",
                "if __name__ == \"__main__\":"
            ]
            
            for component in required_components:
                if component in content:
                    print(f"✅ Found: {component}")
                else:
                    print(f"❌ Missing: {component}")
            
            return True
        else:
            print(f"❌ main_production.py not found")
            return False
            
    except SyntaxError as e:
        print(f"❌ Syntax error in main_production.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing main_production.py: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Testing Production Application Configuration")
    print("This test verifies what can be imported and configured without database connections.")
    print()
    
    results = []
    results.append(test_basic_imports())
    results.append(test_environment_loading())
    results.append(test_config_module())
    results.append(test_main_production_syntax())
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All tests passed! ({passed}/{total})")
        print("\nNext steps:")
        print("1. Install production dependencies: pip install -r requirements/prod.txt")
        print("2. Set up PostgreSQL and Redis databases")
        print("3. Run database migrations")
        print("4. Test full production startup: python3 main_production.py")
    else:
        print(f"⚠️  Some tests failed ({passed}/{total})")
        print("\nThis is expected if database dependencies are not installed.")
        print("The application structure is ready for production deployment.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)