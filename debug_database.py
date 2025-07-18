#!/usr/bin/env python3
"""
Debug database connection and schema
"""

import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment
load_dotenv(Path(__file__).parent / ".env.production")

from scripts.data_ingestion_pipeline import DataIngestionPipeline

async def debug_database():
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rental_ml")
    print(f"Database URL: {database_url}")
    
    pipeline = DataIngestionPipeline(database_url)
    
    try:
        await pipeline.initialize()
        
        # Check table structure
        async with pipeline.connection_manager.get_connection() as conn:
            # List all tables
            tables = await conn.fetch("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            
            print("Available tables:")
            for table in tables:
                print(f"  - {table['table_name']}")
            
            # Check properties table structure if it exists
            properties_columns = await conn.fetch("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = 'properties' AND table_schema = 'public'
                ORDER BY ordinal_position
            """)
            
            print("\nProperties table columns:")
            for col in properties_columns:
                nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                print(f"  - {col['column_name']}: {col['data_type']} {nullable}")
            
            # Check users table structure if it exists
            users_columns = await conn.fetch("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = 'users' AND table_schema = 'public'
                ORDER BY ordinal_position
            """)
            
            print("\nUsers table columns:")
            for col in users_columns:
                nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                print(f"  - {col['column_name']}: {col['data_type']} {nullable}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await pipeline.close()

if __name__ == "__main__":
    asyncio.run(debug_database())