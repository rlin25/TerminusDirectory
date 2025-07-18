#!/usr/bin/env python3
"""
Production Data Pipeline Setup Script
Comprehensive setup and testing of the production data ingestion pipeline
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load production environment
load_dotenv(project_root / ".env.production")

# Import our custom scripts
from scripts.production_data_seeder import ProductionDataSeeder
from scripts.data_ingestion_pipeline import DataIngestionPipeline
from scripts.test_data_operations import DataOperationsTester
from scripts.data_management_utilities import DataManagementUtilities

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionDataPipelineSetup:
    """Complete setup and testing of the production data pipeline"""
    
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rental_ml")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        
    async def run_complete_setup(self) -> Dict[str, Any]:
        """Run complete data pipeline setup and testing"""
        
        setup_results = {
            'started_at': datetime.now().isoformat(),
            'database_url': self.database_url,
            'steps': {},
            'overall_success': False,
            'summary': {}
        }
        
        print("="*80)
        print("🚀 RENTAL ML SYSTEM - PRODUCTION DATA PIPELINE SETUP")
        print("="*80)
        print(f"Database: {self.database_url}")
        print(f"Redis: {self.redis_url}")
        print(f"Started: {setup_results['started_at']}")
        
        try:
            # Step 1: Test database connectivity
            print("\n📋 STEP 1: Testing Database Connectivity...")
            connectivity_result = await self._test_connectivity()
            setup_results['steps']['connectivity'] = connectivity_result
            
            if not connectivity_result['success']:
                print("❌ Database connectivity failed. Cannot proceed.")
                return setup_results
            
            print("✅ Database connectivity verified")
            
            # Step 2: Seed initial data
            print("\n📋 STEP 2: Seeding Production Data...")
            seeding_result = await self._seed_production_data()
            setup_results['steps']['seeding'] = seeding_result
            
            if seeding_result['success']:
                print(f"✅ Successfully seeded {seeding_result['results'].get('properties', 0)} properties and {seeding_result['results'].get('users', 0)} users")
            else:
                print("⚠️ Data seeding had issues, but continuing...")
            
            # Step 3: Test data operations
            print("\n📋 STEP 3: Testing Data Operations...")
            operations_result = await self._test_data_operations()
            setup_results['steps']['operations'] = operations_result
            
            if operations_result['overall_success']:
                print("✅ All data operations working correctly")
            else:
                print("⚠️ Some data operations had issues")
            
            # Step 4: Data quality check
            print("\n📋 STEP 4: Running Data Quality Check...")
            quality_result = await self._check_data_quality()
            setup_results['steps']['quality'] = quality_result
            
            print("✅ Data quality analysis completed")
            
            # Step 5: Test data ingestion pipeline
            print("\n📋 STEP 5: Testing Data Ingestion Pipeline...")
            ingestion_result = await self._test_ingestion_pipeline()
            setup_results['steps']['ingestion'] = ingestion_result
            
            if ingestion_result['success']:
                print("✅ Data ingestion pipeline working")
            else:
                print("⚠️ Data ingestion pipeline had issues")
            
            # Calculate overall success
            critical_steps = ['connectivity', 'seeding', 'operations']
            setup_results['overall_success'] = all(
                setup_results['steps'][step].get('success', False) or 
                setup_results['steps'][step].get('overall_success', False)
                for step in critical_steps
            )
            
            # Generate summary
            setup_results['summary'] = self._generate_summary(setup_results)
            
        except Exception as e:
            logger.error(f"Setup failed with error: {e}")
            setup_results['error'] = str(e)
            
        finally:
            setup_results['completed_at'] = datetime.now().isoformat()
            
        return setup_results
    
    async def _test_connectivity(self) -> Dict[str, Any]:
        """Test database connectivity"""
        
        try:
            pipeline = DataIngestionPipeline(self.database_url)
            await pipeline.initialize()
            
            status = await pipeline.check_database_status()
            await pipeline.close()
            
            return {
                'success': status['connected'] and status['tables_exist'],
                'details': status
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _seed_production_data(self) -> Dict[str, Any]:
        """Seed the database with production data"""
        
        try:
            seeder = ProductionDataSeeder(self.database_url)
            
            # Seed with moderate amount of data
            results = await seeder.seed_all(
                users_count=50,          # 50 users
                properties_count=200,    # 200 properties
                interactions_count=1000, # 1000 interactions
                queries_count=500,       # 500 search queries
                clear_existing=True      # Clear existing data
            )
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _test_data_operations(self) -> Dict[str, Any]:
        """Test all data operations"""
        
        try:
            tester = DataOperationsTester()
            results = await tester.run_all_tests()
            return results
            
        except Exception as e:
            return {
                'overall_success': False,
                'error': str(e)
            }
    
    async def _check_data_quality(self) -> Dict[str, Any]:
        """Check data quality"""
        
        try:
            utilities = DataManagementUtilities(self.database_url)
            await utilities.initialize()
            
            quality_results = await utilities.run_quality_check()
            await utilities.close()
            
            return {
                'success': True,
                'quality_analysis': quality_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _test_ingestion_pipeline(self) -> Dict[str, Any]:
        """Test data ingestion pipeline"""
        
        try:
            pipeline = DataIngestionPipeline(self.database_url)
            await pipeline.initialize()
            
            # Test with sample data
            results = await pipeline.ingest_sample_properties(count=10)
            await pipeline.close()
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_summary(self, setup_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate setup summary"""
        
        summary = {
            'overall_status': 'SUCCESS' if setup_results['overall_success'] else 'PARTIAL',
            'steps_completed': len(setup_results['steps']),
            'critical_issues': [],
            'recommendations': []
        }
        
        # Check for critical issues
        if not setup_results['steps'].get('connectivity', {}).get('success', False):
            summary['critical_issues'].append("Database connectivity failed")
        
        if not setup_results['steps'].get('seeding', {}).get('success', False):
            summary['critical_issues'].append("Data seeding failed")
        
        if not setup_results['steps'].get('operations', {}).get('overall_success', False):
            summary['critical_issues'].append("Data operations tests failed")
        
        # Generate recommendations
        if setup_results['overall_success']:
            summary['recommendations'].extend([
                "✅ Production data pipeline is ready",
                "✅ You can now run property searches and recommendations",
                "✅ Data quality monitoring is available",
                "💡 Consider setting up automated data quality checks",
                "💡 Monitor database performance as data grows"
            ])
        else:
            summary['recommendations'].extend([
                "⚠️ Review and fix critical issues before production use",
                "🔧 Check database configuration and connectivity",
                "🔧 Verify all required tables exist and have proper schema",
                "📞 Contact support if issues persist"
            ])
        
        return summary
    
    def print_detailed_results(self, results: Dict[str, Any]):
        """Print detailed setup results"""
        
        print("\n" + "="*80)
        print("📊 DETAILED SETUP RESULTS")
        print("="*80)
        
        # Overall status
        status = "SUCCESS" if results['overall_success'] else "PARTIAL/FAILED"
        print(f"\n🎯 Overall Status: {status}")
        print(f"⏱️  Duration: {results.get('completed_at', 'N/A')}")
        
        # Step-by-step results
        print(f"\n📋 Step Results:")
        for step_name, step_result in results.get('steps', {}).items():
            success = step_result.get('success', step_result.get('overall_success', False))
            status_icon = "✅" if success else "❌"
            print(f"  {status_icon} {step_name.title()}: {'PASS' if success else 'FAIL'}")
            
            # Show key metrics
            if step_name == 'seeding' and 'results' in step_result:
                for table, count in step_result['results'].items():
                    print(f"    📊 {table}: {count} records")
            
            elif step_name == 'operations' and 'summary' in step_result:
                summary = step_result['summary']
                total_tests = summary.get('total_individual_tests', 0)
                passed_tests = summary.get('successful_individual_tests', 0)
                print(f"    📊 Tests: {passed_tests}/{total_tests} passed")
            
            elif step_name == 'quality' and 'quality_analysis' in step_result:
                prop_analysis = step_result['quality_analysis'].get('property_analysis', {})
                total_props = prop_analysis.get('total_properties', 0)
                print(f"    📊 Properties analyzed: {total_props}")
        
        # Summary and recommendations
        summary = results.get('summary', {})
        if summary.get('critical_issues'):
            print(f"\n🚨 Critical Issues:")
            for issue in summary['critical_issues']:
                print(f"  ❗ {issue}")
        
        if summary.get('recommendations'):
            print(f"\n💡 Recommendations:")
            for rec in summary['recommendations']:
                print(f"  {rec}")
        
        print("\n" + "="*80)


async def main():
    """Main function"""
    
    setup = ProductionDataPipelineSetup()
    
    # Run complete setup
    results = await setup.run_complete_setup()
    
    # Print detailed results
    setup.print_detailed_results(results)
    
    # Final status
    if results['overall_success']:
        print("🎉 PRODUCTION DATA PIPELINE SETUP COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("  1. Start the production API server: python main_production.py")
        print("  2. Test the API endpoints for property search and recommendations")
        print("  3. Set up monitoring and alerts for data quality")
    else:
        print("⚠️  SETUP COMPLETED WITH ISSUES")
        print("\nPlease review the issues above and fix them before proceeding to production.")
    
    return 0 if results['overall_success'] else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)