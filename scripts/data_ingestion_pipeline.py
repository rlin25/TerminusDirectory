#!/usr/bin/env python3
"""
Data Ingestion Pipeline for Rental ML System
Provides simple data ingestion functionality for initial property data
"""

import asyncio
import logging
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load production environment
load_dotenv(project_root / ".env.production")

from src.infrastructure.data.repository_factory import DatabaseConnectionManager
from src.domain.entities.property import Property

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleDataValidator:
    """Simple data validation for ingested properties"""
    
    @staticmethod
    def validate_property_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean property data"""
        cleaned_data = {}
        errors = []
        
        # Required fields
        required_fields = ['title', 'price', 'location', 'bedrooms', 'bathrooms']
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")
            else:
                cleaned_data[field] = data[field]
        
        # Validate price
        if 'price' in data:
            try:
                price = float(data['price'])
                if price < 0:
                    errors.append("Price cannot be negative")
                elif price > 100000:
                    errors.append("Price seems unrealistic (>$100,000)")
                else:
                    cleaned_data['price'] = price
            except (ValueError, TypeError):
                errors.append("Invalid price format")
        
        # Validate bedrooms
        if 'bedrooms' in data:
            try:
                bedrooms = int(data['bedrooms'])
                if bedrooms < 0 or bedrooms > 10:
                    errors.append("Invalid bedroom count")
                else:
                    cleaned_data['bedrooms'] = bedrooms
            except (ValueError, TypeError):
                errors.append("Invalid bedroom format")
        
        # Validate bathrooms
        if 'bathrooms' in data:
            try:
                bathrooms = float(data['bathrooms'])
                if bathrooms < 0 or bathrooms > 20:
                    errors.append("Invalid bathroom count")
                else:
                    cleaned_data['bathrooms'] = bathrooms
            except (ValueError, TypeError):
                errors.append("Invalid bathroom format")
        
        # Optional fields with defaults
        optional_fields = {
            'description': '',
            'square_feet': None,
            'amenities': [],
            'contact_info': {},
            'images': [],
            'property_type': 'apartment'
        }
        
        for field, default_value in optional_fields.items():
            cleaned_data[field] = data.get(field, default_value)
        
        return {
            'cleaned_data': cleaned_data,
            'errors': errors,
            'is_valid': len(errors) == 0
        }


class DataIngestionPipeline:
    """Simple data ingestion pipeline for property data"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection_manager = DatabaseConnectionManager(database_url)
        self.validator = SimpleDataValidator()
        
    async def initialize(self):
        """Initialize database connection"""
        await self.connection_manager.initialize()
        logger.info("Data ingestion pipeline initialized")
        
    async def close(self):
        """Close database connections"""
        await self.connection_manager.close()
        logger.info("Data ingestion pipeline closed")
        
    async def ingest_from_json_file(self, file_path: str) -> Dict[str, Any]:
        """Ingest property data from JSON file"""
        logger.info(f"Ingesting data from {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read JSON file: {e}")
            return {'success': False, 'error': str(e)}
        
        if not isinstance(data, list):
            data = [data]  # Wrap single property in list
        
        return await self.ingest_properties(data)
    
    async def ingest_properties(self, properties_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ingest a list of property data"""
        results = {
            'total_processed': 0,
            'successful_inserts': 0,
            'validation_errors': 0,
            'database_errors': 0,
            'errors': [],
            'success': False
        }
        
        async with self.connection_manager.get_connection() as conn:
            for i, property_data in enumerate(properties_data):
                results['total_processed'] += 1
                
                # Validate data
                validation_result = self.validator.validate_property_data(property_data)
                
                if not validation_result['is_valid']:
                    results['validation_errors'] += 1
                    results['errors'].append({
                        'index': i,
                        'type': 'validation',
                        'errors': validation_result['errors']
                    })
                    continue
                
                # Create property entity
                try:
                    clean_data = validation_result['cleaned_data']
                    
                    # Create property using domain entity
                    property_obj = Property.create(
                        title=clean_data['title'],
                        description=clean_data['description'],
                        price=clean_data['price'],
                        location=clean_data['location'],
                        bedrooms=clean_data['bedrooms'],
                        bathrooms=clean_data['bathrooms'],
                        square_feet=clean_data.get('square_feet'),
                        amenities=clean_data.get('amenities', []),
                        contact_info=clean_data.get('contact_info', {}),
                        images=clean_data.get('images', []),
                        property_type=clean_data.get('property_type', 'apartment')
                    )
                    
                    # Insert into database
                    await self._insert_property(conn, property_obj)
                    results['successful_inserts'] += 1
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"Processed {i + 1}/{len(properties_data)} properties")
                    
                except Exception as e:
                    results['database_errors'] += 1
                    results['errors'].append({
                        'index': i,
                        'type': 'database',
                        'error': str(e)
                    })
                    logger.error(f"Failed to insert property {i}: {e}")
        
        results['success'] = results['successful_inserts'] > 0
        
        logger.info(
            f"Ingestion completed: {results['successful_inserts']}/{results['total_processed']} "
            f"successful, {results['validation_errors']} validation errors, "
            f"{results['database_errors']} database errors"
        )
        
        return results
    
    async def _insert_property(self, conn, property_obj: Property):
        """Insert property into database"""
        await conn.execute("""
            INSERT INTO properties (
                id, title, description, price, location, bedrooms, bathrooms,
                square_feet, amenities, contact_info, images, scraped_at,
                is_active, property_type
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        """, 
            str(property_obj.id), property_obj.title, property_obj.description,
            float(property_obj.price), property_obj.location, property_obj.bedrooms,
            float(property_obj.bathrooms), property_obj.square_feet, property_obj.amenities,
            json.dumps(property_obj.contact_info) if property_obj.contact_info else '{}', 
            property_obj.images, property_obj.scraped_at or datetime.now(),
            property_obj.is_active, property_obj.property_type
        )
    
    async def ingest_sample_properties(self, count: int = 50) -> Dict[str, Any]:
        """Ingest sample properties for testing"""
        logger.info(f"Generating and ingesting {count} sample properties")
        
        sample_properties = []
        
        locations = ["Downtown", "Midtown", "Uptown", "Suburban Heights", "Riverside"]
        property_types = ["apartment", "house", "condo", "studio", "townhouse"]
        amenities_pool = ["parking", "gym", "pool", "laundry", "pet-friendly", "balcony"]
        
        import random
        
        for i in range(count):
            property_data = {
                "title": f"Sample Property {i+1}",
                "description": f"Beautiful sample property {i+1} with modern amenities",
                "price": random.randint(800, 4000),
                "location": random.choice(locations),
                "bedrooms": random.randint(0, 4),
                "bathrooms": random.choice([1.0, 1.5, 2.0, 2.5, 3.0]),
                "square_feet": random.randint(400, 2000),
                "property_type": random.choice(property_types),
                "amenities": random.sample(amenities_pool, random.randint(1, 4)),
                "contact_info": {
                    "phone": f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}",
                    "email": f"contact{i+1}@example.com"
                },
                "images": [f"https://picsum.photos/800/600?random={i+j}" for j in range(random.randint(1, 5))]
            }
            sample_properties.append(property_data)
        
        return await self.ingest_properties(sample_properties)
    
    async def check_database_status(self) -> Dict[str, Any]:
        """Check current database status"""
        logger.info("Checking database status")
        
        status = {
            'connected': False,
            'tables_exist': False,
            'property_count': 0,
            'user_count': 0,
            'interaction_count': 0,
            'errors': []
        }
        
        try:
            async with self.connection_manager.get_connection() as conn:
                status['connected'] = True
                
                # Check if tables exist
                tables_result = await conn.fetch("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name IN ('properties', 'users', 'user_interactions')
                """)
                
                existing_tables = [row['table_name'] for row in tables_result]
                status['tables_exist'] = len(existing_tables) >= 3
                status['existing_tables'] = existing_tables
                
                # Get record counts
                if 'properties' in existing_tables:
                    status['property_count'] = await conn.fetchval("SELECT COUNT(*) FROM properties")
                
                if 'users' in existing_tables:
                    status['user_count'] = await conn.fetchval("SELECT COUNT(*) FROM users")
                
                if 'user_interactions' in existing_tables:
                    status['interaction_count'] = await conn.fetchval("SELECT COUNT(*) FROM user_interactions")
                
        except Exception as e:
            status['errors'].append(str(e))
            logger.error(f"Database status check failed: {e}")
        
        return status
    
    async def cleanup_data(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Clean up data from specified table or all tables"""
        logger.info(f"Cleaning up data from {table_name or 'all tables'}")
        
        results = {
            'success': False,
            'tables_cleared': [],
            'errors': []
        }
        
        try:
            async with self.connection_manager.get_connection() as conn:
                if table_name:
                    tables_to_clear = [table_name]
                else:
                    # Clear in dependency order
                    tables_to_clear = [
                        'user_interactions',
                        'search_queries',
                        'training_metrics',
                        'ml_models',
                        'embeddings',
                        'properties',
                        'users'
                    ]
                
                for table in tables_to_clear:
                    try:
                        await conn.execute(f"TRUNCATE TABLE {table} CASCADE")
                        results['tables_cleared'].append(table)
                        logger.info(f"Cleared table: {table}")
                    except Exception as e:
                        results['errors'].append(f"Failed to clear {table}: {e}")
                        logger.warning(f"Could not clear table {table}: {e}")
                
                results['success'] = len(results['tables_cleared']) > 0
                
        except Exception as e:
            results['errors'].append(str(e))
            logger.error(f"Data cleanup failed: {e}")
        
        return results


async def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Ingestion Pipeline for Rental ML System')
    parser.add_argument('--command', choices=['status', 'sample', 'cleanup', 'ingest'], 
                       default='status', help='Command to execute')
    parser.add_argument('--file', help='JSON file to ingest (for ingest command)')
    parser.add_argument('--count', type=int, default=50, help='Number of sample properties (for sample command)')
    parser.add_argument('--table', help='Specific table to cleanup (for cleanup command)')
    
    args = parser.parse_args()
    
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rental_ml")
    
    pipeline = DataIngestionPipeline(database_url)
    
    try:
        await pipeline.initialize()
        
        if args.command == 'status':
            status = await pipeline.check_database_status()
            print("\n" + "="*50)
            print("DATABASE STATUS")
            print("="*50)
            print(f"Connected: {status['connected']}")
            print(f"Tables exist: {status['tables_exist']}")
            print(f"Properties: {status['property_count']}")
            print(f"Users: {status['user_count']}")
            print(f"Interactions: {status['interaction_count']}")
            
            if status['errors']:
                print("\nErrors:")
                for error in status['errors']:
                    print(f"  - {error}")
        
        elif args.command == 'sample':
            results = await pipeline.ingest_sample_properties(args.count)
            print("\n" + "="*50)
            print("SAMPLE DATA INGESTION RESULTS")
            print("="*50)
            print(f"Success: {results['success']}")
            print(f"Total processed: {results['total_processed']}")
            print(f"Successful inserts: {results['successful_inserts']}")
            print(f"Validation errors: {results['validation_errors']}")
            print(f"Database errors: {results['database_errors']}")
        
        elif args.command == 'cleanup':
            results = await pipeline.cleanup_data(args.table)
            print("\n" + "="*50)
            print("DATA CLEANUP RESULTS")
            print("="*50)
            print(f"Success: {results['success']}")
            print(f"Tables cleared: {', '.join(results['tables_cleared'])}")
            
            if results['errors']:
                print("\nErrors:")
                for error in results['errors']:
                    print(f"  - {error}")
        
        elif args.command == 'ingest':
            if not args.file:
                print("Error: --file parameter required for ingest command")
                return
            
            results = await pipeline.ingest_from_json_file(args.file)
            print("\n" + "="*50)
            print("DATA INGESTION RESULTS")
            print("="*50)
            print(f"Success: {results['success']}")
            
            if 'total_processed' in results:
                print(f"Total processed: {results['total_processed']}")
                print(f"Successful inserts: {results['successful_inserts']}")
                print(f"Validation errors: {results['validation_errors']}")
                print(f"Database errors: {results['database_errors']}")
    
    finally:
        await pipeline.close()
    
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())