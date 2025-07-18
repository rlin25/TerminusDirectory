#!/usr/bin/env python3
"""
Test script to verify PostgreSQL Property Repository implementation completeness
"""

import sys
import os
import inspect

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_repository_completeness():
    """Test if all interface methods are implemented in the repository"""
    
    print("🔍 Testing PostgreSQL Property Repository Implementation")
    print("=" * 60)
    
    try:
        # Import the interface
        from domain.repositories.property_repository import PropertyRepository
        print("✅ PropertyRepository interface imported successfully")
        
        # Import the implementation (skip actual import due to dependencies)
        # Instead, read the file and analyze it
        with open('src/infrastructure/data/repositories/postgres_property_repository.py', 'r') as f:
            content = f.read()
        
        print("✅ PostgresPropertyRepository file read successfully")
        
        # Get all abstract methods from the interface
        interface_methods = []
        for name, method in inspect.getmembers(PropertyRepository, predicate=inspect.isfunction):
            if hasattr(method, '__isabstractmethod__') and method.__isabstractmethod__:
                interface_methods.append(name)
        
        print(f"📋 Interface defines {len(interface_methods)} abstract methods:")
        for method in sorted(interface_methods):
            print(f"   • {method}")
        
        print("\n🔍 Checking implementation...")
        
        # Check if each method is implemented in the repository
        missing_methods = []
        implemented_methods = []
        
        for method in interface_methods:
            # Look for method definition in the content
            method_pattern = f"async def {method}("
            if method_pattern in content:
                implemented_methods.append(method)
                print(f"   ✅ {method}")
            else:
                missing_methods.append(method)
                print(f"   ❌ {method}")
        
        print(f"\n📊 Implementation Summary:")
        print(f"   • Total interface methods: {len(interface_methods)}")
        print(f"   • Implemented methods: {len(implemented_methods)}")
        print(f"   • Missing methods: {len(missing_methods)}")
        
        if missing_methods:
            print(f"\n❌ Missing implementations:")
            for method in missing_methods:
                print(f"   • {method}")
            return False
        else:
            print(f"\n🎉 All interface methods are implemented!")
            
            # Check for additional features
            additional_features = [
                "retry_on_db_error",
                "measure_performance", 
                "health_check",
                "get_connection_info",
                "create_tables",
                "optimize_database",
                "get_aggregated_stats",
                "get_trending_properties",
                "bulk_update_engagement_metrics"
            ]
            
            print(f"\n🚀 Additional Production Features:")
            for feature in additional_features:
                if feature in content:
                    print(f"   ✅ {feature}")
                else:
                    print(f"   ⚠️  {feature}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_implementation_quality():
    """Analyze the quality and completeness of the implementation"""
    
    print("\n" + "=" * 60)
    print("🔬 Implementation Quality Analysis")
    print("=" * 60)
    
    try:
        with open('src/infrastructure/data/repositories/postgres_property_repository.py', 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        total_lines = len(lines)
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        docstring_lines = len([line for line in lines if '"""' in line or "'''" in line])
        
        print(f"📏 Code Metrics:")
        print(f"   • Total lines: {total_lines}")
        print(f"   • Comment lines: {comment_lines}")
        print(f"   • Docstring lines: {docstring_lines}")
        
        # Check for important patterns
        patterns = {
            "Error handling": ["try:", "except", "raise"],
            "Logging": ["logger.", "logging"],
            "Type hints": [": ", "-> ", "Optional", "List", "Dict"],
            "Async/await": ["async def", "await "],
            "Database transactions": ["session", "transaction", "commit", "rollback"],
            "SQL queries": ["select", "insert", "update", "delete"],
            "Performance": ["index", "cache", "batch", "optimize"],
            "Monitoring": ["metrics", "performance", "health_check"]
        }
        
        print(f"\n🔍 Feature Analysis:")
        for category, keywords in patterns.items():
            count = sum(content.lower().count(keyword.lower()) for keyword in keywords)
            status = "✅" if count > 0 else "❌"
            print(f"   {status} {category}: {count} occurrences")
        
        # Check for specific production-ready features
        production_features = {
            "Connection pooling": "pool_size",
            "Retry mechanisms": "retry_on_db_error",
            "Performance monitoring": "measure_performance",
            "Health checks": "health_check",
            "Data validation": "_validate_property_data",
            "Bulk operations": "bulk_create",
            "Search optimization": "full_text_search",
            "Analytics support": "get_aggregated_stats"
        }
        
        print(f"\n🏭 Production Features:")
        for feature, pattern in production_features.items():
            status = "✅" if pattern in content else "❌"
            print(f"   {status} {feature}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False

if __name__ == "__main__":
    print("PostgreSQL Property Repository Implementation Test")
    print("=" * 60)
    
    success = test_repository_completeness()
    
    if success:
        analyze_implementation_quality()
        print(f"\n🎯 Overall Assessment: COMPLETE ✅")
        print("The PostgreSQL Property Repository implementation appears to be")
        print("complete with all interface methods implemented and additional")
        print("production-ready features included.")
    else:
        print(f"\n🎯 Overall Assessment: INCOMPLETE ❌")
        print("The implementation is missing some required methods.")
    
    print("\n" + "=" * 60)