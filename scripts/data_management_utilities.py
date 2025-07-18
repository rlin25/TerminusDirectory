#!/usr/bin/env python3
"""
Data Management Utilities for Rental ML System
Provides utilities for data quality checks, cleaning/updating, and export/import functionality
"""

import asyncio
import logging
import os
import sys
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load production environment
load_dotenv(project_root / ".env.production")

from src.infrastructure.data.repository_factory import DatabaseConnectionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataQualityAnalyzer:
    """Analyze data quality across the rental ML system"""
    
    def __init__(self, connection_manager: DatabaseConnectionManager):
        self.connection_manager = connection_manager
        
    async def analyze_property_quality(self) -> Dict[str, Any]:
        """Analyze property data quality"""
        logger.info("Analyzing property data quality...")
        
        analysis = {
            'total_properties': 0,
            'quality_issues': {},
            'data_completeness': {},
            'price_analysis': {},
            'location_analysis': {},
            'date_analysis': {},
            'recommendations': []
        }
        
        async with self.connection_manager.get_connection() as conn:
            # Basic counts
            analysis['total_properties'] = await conn.fetchval("SELECT COUNT(*) FROM properties")
            
            if analysis['total_properties'] == 0:
                analysis['recommendations'].append("No properties found in database")
                return analysis
            
            # Check for missing required fields
            missing_title = await conn.fetchval("SELECT COUNT(*) FROM properties WHERE title IS NULL OR title = ''")
            missing_price = await conn.fetchval("SELECT COUNT(*) FROM properties WHERE price IS NULL OR price <= 0")
            missing_location = await conn.fetchval("SELECT COUNT(*) FROM properties WHERE location IS NULL OR location = ''")
            missing_bedrooms = await conn.fetchval("SELECT COUNT(*) FROM properties WHERE bedrooms IS NULL")
            missing_bathrooms = await conn.fetchval("SELECT COUNT(*) FROM properties WHERE bathrooms IS NULL")
            
            analysis['quality_issues'] = {
                'missing_title': missing_title,
                'missing_price': missing_price,
                'missing_location': missing_location,
                'missing_bedrooms': missing_bedrooms,
                'missing_bathrooms': missing_bathrooms
            }
            
            # Data completeness
            with_description = await conn.fetchval("SELECT COUNT(*) FROM properties WHERE description IS NOT NULL AND description != ''")
            with_square_feet = await conn.fetchval("SELECT COUNT(*) FROM properties WHERE square_feet IS NOT NULL AND square_feet > 0")
            with_amenities = await conn.fetchval("SELECT COUNT(*) FROM properties WHERE amenities IS NOT NULL AND array_length(amenities, 1) > 0")
            with_images = await conn.fetchval("SELECT COUNT(*) FROM properties WHERE images IS NOT NULL AND array_length(images, 1) > 0")
            with_contact = await conn.fetchval("SELECT COUNT(*) FROM properties WHERE contact_info IS NOT NULL")
            
            total = analysis['total_properties']
            analysis['data_completeness'] = {
                'description_completion': round((with_description / total) * 100, 2) if total > 0 else 0,
                'square_feet_completion': round((with_square_feet / total) * 100, 2) if total > 0 else 0,
                'amenities_completion': round((with_amenities / total) * 100, 2) if total > 0 else 0,
                'images_completion': round((with_images / total) * 100, 2) if total > 0 else 0,
                'contact_completion': round((with_contact / total) * 100, 2) if total > 0 else 0
            }
            
            # Price analysis
            price_stats = await conn.fetchrow("""
                SELECT 
                    MIN(price) as min_price,
                    MAX(price) as max_price,
                    AVG(price) as avg_price,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) as median_price
                FROM properties WHERE price > 0
            """)
            
            if price_stats:
                analysis['price_analysis'] = {
                    'min_price': float(price_stats['min_price']),
                    'max_price': float(price_stats['max_price']),
                    'avg_price': round(float(price_stats['avg_price']), 2),
                    'median_price': round(float(price_stats['median_price']), 2)
                }
                
                # Check for outliers
                very_cheap = await conn.fetchval("SELECT COUNT(*) FROM properties WHERE price < 500")
                very_expensive = await conn.fetchval("SELECT COUNT(*) FROM properties WHERE price > 10000")
                
                analysis['price_analysis']['outliers'] = {
                    'very_cheap_properties': very_cheap,
                    'very_expensive_properties': very_expensive
                }
            
            # Location analysis
            location_stats = await conn.fetch("""
                SELECT location, COUNT(*) as count 
                FROM properties 
                WHERE location IS NOT NULL AND location != ''
                GROUP BY location 
                ORDER BY count DESC 
                LIMIT 10
            """)
            
            analysis['location_analysis'] = {
                'unique_locations': len(location_stats),
                'top_locations': [(row['location'], row['count']) for row in location_stats]
            }
            
            # Date analysis
            date_stats = await conn.fetchrow("""
                SELECT 
                    MIN(created_at) as oldest_property,
                    MAX(created_at) as newest_property,
                    AVG(created_at) as avg_created_at
                FROM properties
            """)
            
            if date_stats:
                analysis['date_analysis'] = {
                    'oldest_property': date_stats['oldest_property'].isoformat() if date_stats['oldest_property'] else None,
                    'newest_property': date_stats['newest_property'].isoformat() if date_stats['newest_property'] else None,
                    'data_span_days': (date_stats['newest_property'] - date_stats['oldest_property']).days if date_stats['newest_property'] and date_stats['oldest_property'] else 0
                }
            
            # Generate recommendations
            if missing_title > 0:
                analysis['recommendations'].append(f"Fix {missing_title} properties with missing titles")
            if missing_price > 0:
                analysis['recommendations'].append(f"Fix {missing_price} properties with missing or invalid prices")
            if analysis['data_completeness']['description_completion'] < 80:
                analysis['recommendations'].append("Improve description completion rate (currently below 80%)")
            if analysis['price_analysis'].get('outliers', {}).get('very_expensive_properties', 0) > 0:
                analysis['recommendations'].append("Review very expensive properties for data entry errors")
        
        return analysis
    
    async def analyze_user_quality(self) -> Dict[str, Any]:
        """Analyze user data quality"""
        logger.info("Analyzing user data quality...")
        
        analysis = {
            'total_users': 0,
            'email_analysis': {},
            'preference_analysis': {},
            'activity_analysis': {},
            'recommendations': []
        }
        
        async with self.connection_manager.get_connection() as conn:
            # Basic counts
            analysis['total_users'] = await conn.fetchval("SELECT COUNT(*) FROM users")
            
            if analysis['total_users'] == 0:
                analysis['recommendations'].append("No users found in database")
                return analysis
            
            # Email analysis
            invalid_emails = await conn.fetchval("""
                SELECT COUNT(*) FROM users 
                WHERE email IS NULL OR email = '' OR email NOT LIKE '%@%'
            """)
            
            duplicate_emails = await conn.fetchval("""
                SELECT COUNT(*) FROM (
                    SELECT email FROM users 
                    GROUP BY email 
                    HAVING COUNT(*) > 1
                ) as duplicates
            """)
            
            analysis['email_analysis'] = {
                'invalid_emails': invalid_emails,
                'duplicate_emails': duplicate_emails
            }
            
            # Preference analysis
            users_with_price_prefs = await conn.fetchval("""
                SELECT COUNT(*) FROM users 
                WHERE min_price IS NOT NULL AND max_price IS NOT NULL
            """)
            
            users_with_location_prefs = await conn.fetchval("""
                SELECT COUNT(*) FROM users 
                WHERE preferred_locations IS NOT NULL AND array_length(preferred_locations, 1) > 0
            """)
            
            total = analysis['total_users']
            analysis['preference_analysis'] = {
                'price_preference_completion': round((users_with_price_prefs / total) * 100, 2) if total > 0 else 0,
                'location_preference_completion': round((users_with_location_prefs / total) * 100, 2) if total > 0 else 0
            }
            
            # Activity analysis
            active_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE status = 'active'")
            
            analysis['activity_analysis'] = {
                'active_users': active_users,
                'inactive_users': total - active_users,
                'active_percentage': round((active_users / total) * 100, 2) if total > 0 else 0
            }
            
            # Generate recommendations
            if invalid_emails > 0:
                analysis['recommendations'].append(f"Fix {invalid_emails} users with invalid email addresses")
            if duplicate_emails > 0:
                analysis['recommendations'].append(f"Resolve {duplicate_emails} duplicate email addresses")
            if analysis['preference_analysis']['price_preference_completion'] < 70:
                analysis['recommendations'].append("Encourage users to complete price preferences")
        
        return analysis
    
    async def analyze_interaction_quality(self) -> Dict[str, Any]:
        """Analyze user interaction data quality"""
        logger.info("Analyzing interaction data quality...")
        
        analysis = {
            'total_interactions': 0,
            'interaction_types': {},
            'data_integrity': {},
            'temporal_analysis': {},
            'recommendations': []
        }
        
        async with self.connection_manager.get_connection() as conn:
            # Basic counts
            analysis['total_interactions'] = await conn.fetchval("SELECT COUNT(*) FROM user_interactions")
            
            if analysis['total_interactions'] == 0:
                analysis['recommendations'].append("No user interactions found in database")
                return analysis
            
            # Interaction types
            type_stats = await conn.fetch("""
                SELECT interaction_type, COUNT(*) as count 
                FROM user_interactions 
                GROUP BY interaction_type 
                ORDER BY count DESC
            """)
            
            analysis['interaction_types'] = {row['interaction_type']: row['count'] for row in type_stats}
            
            # Data integrity checks
            orphaned_user_interactions = await conn.fetchval("""
                SELECT COUNT(*) FROM user_interactions ui
                LEFT JOIN users u ON ui.user_id = u.id
                WHERE u.id IS NULL
            """)
            
            orphaned_property_interactions = await conn.fetchval("""
                SELECT COUNT(*) FROM user_interactions ui
                LEFT JOIN properties p ON ui.property_id = p.id
                WHERE p.id IS NULL
            """)
            
            analysis['data_integrity'] = {
                'orphaned_user_interactions': orphaned_user_interactions,
                'orphaned_property_interactions': orphaned_property_interactions
            }
            
            # Temporal analysis
            recent_interactions = await conn.fetchval("""
                SELECT COUNT(*) FROM user_interactions 
                WHERE timestamp > NOW() - INTERVAL '30 days'
            """)
            
            old_interactions = await conn.fetchval("""
                SELECT COUNT(*) FROM user_interactions 
                WHERE timestamp < NOW() - INTERVAL '90 days'
            """)
            
            analysis['temporal_analysis'] = {
                'recent_interactions_30d': recent_interactions,
                'old_interactions_90d': old_interactions
            }
            
            # Generate recommendations
            if orphaned_user_interactions > 0:
                analysis['recommendations'].append(f"Clean up {orphaned_user_interactions} orphaned user interactions")
            if orphaned_property_interactions > 0:
                analysis['recommendations'].append(f"Clean up {orphaned_property_interactions} orphaned property interactions")
            if recent_interactions == 0:
                analysis['recommendations'].append("No recent user interactions - check data ingestion")
        
        return analysis


class DataCleaner:
    """Clean and update data in the rental ML system"""
    
    def __init__(self, connection_manager: DatabaseConnectionManager):
        self.connection_manager = connection_manager
        
    async def clean_orphaned_records(self) -> Dict[str, Any]:
        """Clean orphaned records from the database"""
        logger.info("Cleaning orphaned records...")
        
        results = {
            'cleaned_tables': {},
            'total_deleted': 0,
            'errors': []
        }
        
        async with self.connection_manager.get_connection() as conn:
            try:
                # Clean orphaned user interactions
                deleted_user_interactions = await conn.execute("""
                    DELETE FROM user_interactions 
                    WHERE user_id NOT IN (SELECT id FROM users)
                """)
                results['cleaned_tables']['user_interactions_users'] = deleted_user_interactions
                
                deleted_property_interactions = await conn.execute("""
                    DELETE FROM user_interactions 
                    WHERE property_id NOT IN (SELECT id FROM properties)
                """)
                results['cleaned_tables']['user_interactions_properties'] = deleted_property_interactions
                
                # Clean orphaned search queries
                deleted_search_queries = await conn.execute("""
                    DELETE FROM search_queries 
                    WHERE user_id IS NOT NULL AND user_id NOT IN (SELECT id FROM users)
                """)
                results['cleaned_tables']['search_queries'] = deleted_search_queries
                
                results['total_deleted'] = sum(results['cleaned_tables'].values())
                
            except Exception as e:
                logger.error(f"Error cleaning orphaned records: {e}")
                results['errors'].append(str(e))
        
        return results
    
    async def update_property_prices(self, price_adjustments: Dict[str, float]) -> Dict[str, Any]:
        """Update property prices based on adjustments"""
        logger.info("Updating property prices...")
        
        results = {
            'updated_properties': 0,
            'adjustments_applied': {},
            'errors': []
        }
        
        async with self.connection_manager.get_connection() as conn:
            for location, adjustment_factor in price_adjustments.items():
                try:
                    updated_count = await conn.execute("""
                        UPDATE properties 
                        SET price = price * $1, updated_at = NOW()
                        WHERE location ILIKE $2 AND price > 0
                    """, adjustment_factor, f"%{location}%")
                    
                    results['adjustments_applied'][location] = updated_count
                    results['updated_properties'] += updated_count
                    
                except Exception as e:
                    logger.error(f"Error updating prices for {location}: {e}")
                    results['errors'].append(f"{location}: {e}")
        
        return results
    
    async def standardize_locations(self) -> Dict[str, Any]:
        """Standardize location names"""
        logger.info("Standardizing location names...")
        
        # Define location mappings
        location_mappings = {
            'downtown': 'Downtown',
            'dt': 'Downtown',
            'city center': 'Downtown',
            'midtown': 'Midtown',
            'uptown': 'Uptown',
            'suburban': 'Suburban Heights',
            'suburbs': 'Suburban Heights'
        }
        
        results = {
            'standardized_locations': {},
            'total_updated': 0,
            'errors': []
        }
        
        async with self.connection_manager.get_connection() as conn:
            for old_name, new_name in location_mappings.items():
                try:
                    updated_count = await conn.execute("""
                        UPDATE properties 
                        SET location = $1, updated_at = NOW()
                        WHERE LOWER(location) LIKE $2
                    """, new_name, f"%{old_name.lower()}%")
                    
                    if updated_count > 0:
                        results['standardized_locations'][old_name] = {
                            'new_name': new_name,
                            'count': updated_count
                        }
                        results['total_updated'] += updated_count
                        
                except Exception as e:
                    logger.error(f"Error standardizing location {old_name}: {e}")
                    results['errors'].append(f"{old_name}: {e}")
        
        return results
    
    async def fix_invalid_data(self) -> Dict[str, Any]:
        """Fix common data validation issues"""
        logger.info("Fixing invalid data...")
        
        results = {
            'fixes_applied': {},
            'total_fixed': 0,
            'errors': []
        }
        
        async with self.connection_manager.get_connection() as conn:
            try:
                # Fix negative prices
                fixed_prices = await conn.execute("""
                    UPDATE properties 
                    SET price = ABS(price), updated_at = NOW()
                    WHERE price < 0
                """)
                results['fixes_applied']['negative_prices'] = fixed_prices
                
                # Fix excessive bedroom counts (likely data entry errors)
                fixed_bedrooms = await conn.execute("""
                    UPDATE properties 
                    SET bedrooms = 5, updated_at = NOW()
                    WHERE bedrooms > 10
                """)
                results['fixes_applied']['excessive_bedrooms'] = fixed_bedrooms
                
                # Fix excessive bathroom counts
                fixed_bathrooms = await conn.execute("""
                    UPDATE properties 
                    SET bathrooms = 5.0, updated_at = NOW()
                    WHERE bathrooms > 10
                """)
                results['fixes_applied']['excessive_bathrooms'] = fixed_bathrooms
                
                # Fix empty titles
                fixed_titles = await conn.execute("""
                    UPDATE properties 
                    SET title = CONCAT(bedrooms, ' bed, ', bathrooms, ' bath ', property_type), 
                        updated_at = NOW()
                    WHERE title IS NULL OR title = ''
                """)
                results['fixes_applied']['empty_titles'] = fixed_titles
                
                results['total_fixed'] = sum(results['fixes_applied'].values())
                
            except Exception as e:
                logger.error(f"Error fixing invalid data: {e}")
                results['errors'].append(str(e))
        
        return results


class DataExporter:
    """Export data from the rental ML system"""
    
    def __init__(self, connection_manager: DatabaseConnectionManager):
        self.connection_manager = connection_manager
        
    async def export_properties_to_json(self, file_path: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """Export properties to JSON file"""
        logger.info(f"Exporting properties to {file_path}")
        
        result = {
            'success': False,
            'exported_count': 0,
            'file_path': file_path,
            'error': None
        }
        
        try:
            async with self.connection_manager.get_connection() as conn:
                # Build query
                query = """
                    SELECT id, title, description, price, location, bedrooms, bathrooms,
                           square_feet, amenities, contact_info, images, property_type,
                           created_at, updated_at, status
                    FROM properties
                    ORDER BY created_at DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                rows = await conn.fetch(query)
                
                # Convert to JSON-serializable format
                properties = []
                for row in rows:
                    property_data = {
                        'id': str(row['id']),
                        'title': row['title'],
                        'description': row['description'],
                        'price': float(row['price']) if row['price'] else None,
                        'location': row['location'],
                        'bedrooms': row['bedrooms'],
                        'bathrooms': float(row['bathrooms']) if row['bathrooms'] else None,
                        'square_feet': row['square_feet'],
                        'amenities': row['amenities'] or [],
                        'contact_info': row['contact_info'] or {},
                        'images': row['images'] or [],
                        'property_type': row['property_type'],
                        'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                        'updated_at': row['updated_at'].isoformat() if row['updated_at'] else None,
                        'status': row['status']
                    }
                    properties.append(property_data)
                
                # Write to file
                with open(file_path, 'w') as f:
                    json.dump(properties, f, indent=2)
                
                result['success'] = True
                result['exported_count'] = len(properties)
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            result['error'] = str(e)
        
        return result
    
    async def export_properties_to_csv(self, file_path: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """Export properties to CSV file"""
        logger.info(f"Exporting properties to CSV: {file_path}")
        
        result = {
            'success': False,
            'exported_count': 0,
            'file_path': file_path,
            'error': None
        }
        
        try:
            async with self.connection_manager.get_connection() as conn:
                query = """
                    SELECT id, title, description, price, location, bedrooms, bathrooms,
                           square_feet, property_type, created_at, status
                    FROM properties
                    ORDER BY created_at DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                rows = await conn.fetch(query)
                
                # Write to CSV
                with open(file_path, 'w', newline='') as csvfile:
                    fieldnames = ['id', 'title', 'description', 'price', 'location', 
                                'bedrooms', 'bathrooms', 'square_feet', 'property_type', 
                                'created_at', 'status']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for row in rows:
                        writer.writerow({
                            'id': str(row['id']),
                            'title': row['title'],
                            'description': row['description'],
                            'price': row['price'],
                            'location': row['location'],
                            'bedrooms': row['bedrooms'],
                            'bathrooms': row['bathrooms'],
                            'square_feet': row['square_feet'],
                            'property_type': row['property_type'],
                            'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                            'status': row['status']
                        })
                
                result['success'] = True
                result['exported_count'] = len(rows)
                
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            result['error'] = str(e)
        
        return result
    
    async def export_data_statistics(self, file_path: str) -> Dict[str, Any]:
        """Export comprehensive data statistics"""
        logger.info(f"Exporting data statistics to {file_path}")
        
        result = {
            'success': False,
            'file_path': file_path,
            'error': None
        }
        
        try:
            # Gather statistics
            analyzer = DataQualityAnalyzer(self.connection_manager)
            
            statistics = {
                'export_timestamp': datetime.now().isoformat(),
                'property_analysis': await analyzer.analyze_property_quality(),
                'user_analysis': await analyzer.analyze_user_quality(),
                'interaction_analysis': await analyzer.analyze_interaction_quality()
            }
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(statistics, f, indent=2)
            
            result['success'] = True
            
        except Exception as e:
            logger.error(f"Statistics export failed: {e}")
            result['error'] = str(e)
        
        return result


class DataManagementUtilities:
    """Main class for data management utilities"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection_manager = DatabaseConnectionManager(database_url)
        self.analyzer = None
        self.cleaner = None
        self.exporter = None
        
    async def initialize(self):
        """Initialize connection and utilities"""
        await self.connection_manager.initialize()
        self.analyzer = DataQualityAnalyzer(self.connection_manager)
        self.cleaner = DataCleaner(self.connection_manager)
        self.exporter = DataExporter(self.connection_manager)
        logger.info("Data management utilities initialized")
        
    async def close(self):
        """Close connections"""
        await self.connection_manager.close()
        logger.info("Data management utilities closed")
    
    async def run_quality_check(self) -> Dict[str, Any]:
        """Run comprehensive data quality check"""
        return {
            'property_quality': await self.analyzer.analyze_property_quality(),
            'user_quality': await self.analyzer.analyze_user_quality(),
            'interaction_quality': await self.analyzer.analyze_interaction_quality()
        }
    
    async def run_data_cleanup(self) -> Dict[str, Any]:
        """Run comprehensive data cleanup"""
        return {
            'orphaned_cleanup': await self.cleaner.clean_orphaned_records(),
            'location_standardization': await self.cleaner.standardize_locations(),
            'data_fixes': await self.cleaner.fix_invalid_data()
        }
    
    async def export_all_data(self, output_dir: str) -> Dict[str, Any]:
        """Export all data to specified directory"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        return {
            'properties_json': await self.exporter.export_properties_to_json(
                str(output_path / 'properties.json')
            ),
            'properties_csv': await self.exporter.export_properties_to_csv(
                str(output_path / 'properties.csv')
            ),
            'statistics': await self.exporter.export_data_statistics(
                str(output_path / 'data_statistics.json')
            )
        }


async def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Management Utilities for Rental ML System')
    parser.add_argument('--command', choices=['quality', 'cleanup', 'export', 'all'], 
                       default='quality', help='Command to execute')
    parser.add_argument('--output-dir', default='./data_exports', 
                       help='Output directory for exports')
    
    args = parser.parse_args()
    
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rental_ml")
    
    utilities = DataManagementUtilities(database_url)
    
    try:
        await utilities.initialize()
        
        print("="*60)
        print("RENTAL ML SYSTEM - DATA MANAGEMENT UTILITIES")
        print("="*60)
        
        if args.command in ['quality', 'all']:
            print("\n📊 RUNNING DATA QUALITY CHECK...")
            quality_results = await utilities.run_quality_check()
            
            print("\n" + "-"*40)
            print("PROPERTY QUALITY ANALYSIS")
            print("-"*40)
            prop_analysis = quality_results['property_quality']
            print(f"Total Properties: {prop_analysis['total_properties']}")
            print(f"Quality Issues: {sum(prop_analysis['quality_issues'].values())} total")
            print(f"Data Completeness:")
            for field, completion in prop_analysis['data_completeness'].items():
                print(f"  {field}: {completion}%")
            
            if prop_analysis['recommendations']:
                print("\nRecommendations:")
                for rec in prop_analysis['recommendations']:
                    print(f"  - {rec}")
            
            print("\n" + "-"*40)
            print("USER QUALITY ANALYSIS")
            print("-"*40)
            user_analysis = quality_results['user_quality']
            print(f"Total Users: {user_analysis['total_users']}")
            print(f"Invalid Emails: {user_analysis['email_analysis']['invalid_emails']}")
            print(f"Active Users: {user_analysis['activity_analysis']['active_percentage']}%")
        
        if args.command in ['cleanup', 'all']:
            print("\n🧹 RUNNING DATA CLEANUP...")
            cleanup_results = await utilities.run_data_cleanup()
            
            print("\n" + "-"*40)
            print("CLEANUP RESULTS")
            print("-"*40)
            orphaned = cleanup_results['orphaned_cleanup']
            print(f"Orphaned Records Cleaned: {orphaned['total_deleted']}")
            
            location_std = cleanup_results['location_standardization']
            print(f"Locations Standardized: {location_std['total_updated']}")
            
            fixes = cleanup_results['data_fixes']
            print(f"Data Issues Fixed: {fixes['total_fixed']}")
        
        if args.command in ['export', 'all']:
            print("\n📤 EXPORTING DATA...")
            export_results = await utilities.export_all_data(args.output_dir)
            
            print("\n" + "-"*40)
            print("EXPORT RESULTS")
            print("-"*40)
            for export_type, result in export_results.items():
                success = result.get('success', False)
                count = result.get('exported_count', 'N/A')
                print(f"{export_type}: {'SUCCESS' if success else 'FAILED'} ({count} records)")
                if not success and result.get('error'):
                    print(f"  Error: {result['error']}")
    
    finally:
        await utilities.close()
    
    print("\n" + "="*60)
    print("DATA MANAGEMENT UTILITIES COMPLETED")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())