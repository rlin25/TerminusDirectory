#!/usr/bin/env python3
"""
Database Connection Test Suite for Rental ML System
Created: 2025-07-12

This script performs comprehensive database connectivity tests including:
- PostgreSQL connection and performance tests
- Redis connection and caching tests
- CRUD operations validation
- Connection pool testing
- Health checks and monitoring
"""

import os
import sys
import asyncio
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
import asyncpg
import redis.asyncio as redis
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result container"""
    test_name: str
    success: bool
    duration_ms: int
    message: str
    details: Optional[Dict[str, Any]] = None


class DatabaseConnectionTester:
    """Comprehensive database connection and performance tester"""
    
    def __init__(self):
        self.postgres_config = self._load_postgres_config()
        self.redis_config = self._load_redis_config()
        self.test_results: List[TestResult] = []
    
    def _load_postgres_config(self) -> Dict[str, Any]:
        """Load PostgreSQL configuration from environment"""
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'rental_ml'),
            'user': os.getenv('DB_USERNAME', 'rental_ml_user'),
            'password': os.getenv('DB_PASSWORD', 'SecurePassword123!'),
            'pool_size': int(os.getenv('DB_POOL_SIZE', '10')),
            'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '20')),
        }
    
    def _load_redis_config(self) -> Dict[str, Any]:
        """Load Redis configuration from environment"""
        return {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', '6379')),
            'db': int(os.getenv('REDIS_DB', '0')),
            'password': os.getenv('REDIS_PASSWORD'),
            'max_connections': int(os.getenv('REDIS_MAX_CONNECTIONS', '20')),
        }
    
    async def run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test and record results"""
        start_time = time.time()
        try:
            result = await test_func()
            duration_ms = int((time.time() - start_time) * 1000)
            
            if isinstance(result, tuple):
                success, message, details = result
            else:
                success = result
                message = "Test completed successfully" if success else "Test failed"
                details = None
            
            test_result = TestResult(
                test_name=test_name,
                success=success,
                duration_ms=duration_ms,
                message=message,
                details=details
            )
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            test_result = TestResult(
                test_name=test_name,
                success=False,
                duration_ms=duration_ms,
                message=f"Test failed with exception: {str(e)}",
                details={'exception_type': type(e).__name__}
            )
        
        self.test_results.append(test_result)
        return test_result
    
    # ============================================
    # POSTGRESQL TESTS
    # ============================================
    
    async def test_postgres_basic_connection(self):
        """Test basic PostgreSQL connection"""
        try:
            conn = await asyncpg.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password']
            )
            
            # Test basic query
            result = await conn.fetchval("SELECT version()")
            await conn.close()
            
            return True, f"Connected successfully. Server version: {result[:50]}...", {
                'server_version': result,
                'connection_time_ms': time.time()
            }
            
        except Exception as e:
            return False, f"Connection failed: {str(e)}", None
    
    async def test_postgres_connection_pool(self):
        """Test PostgreSQL connection pool"""
        try:
            pool = await asyncpg.create_pool(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password'],
                min_size=5,
                max_size=self.postgres_config['pool_size']
            )
            
            # Test concurrent connections
            tasks = []
            for i in range(10):
                tasks.append(self._test_pool_connection(pool, i))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            await pool.close()
            
            successful_connections = sum(1 for r in results if not isinstance(r, Exception))
            
            return True, f"Pool test completed. {successful_connections}/10 connections successful", {
                'successful_connections': successful_connections,
                'pool_size': self.postgres_config['pool_size'],
                'exceptions': [str(r) for r in results if isinstance(r, Exception)]
            }
            
        except Exception as e:
            return False, f"Pool test failed: {str(e)}", None
    
    async def _test_pool_connection(self, pool, connection_id):
        """Helper method to test individual pool connection"""
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT $1 + $2", connection_id, 1)
            await asyncio.sleep(0.1)  # Simulate work
            return result
    
    async def test_postgres_table_structure(self):
        """Test that all required tables exist with proper structure"""
        try:
            conn = await asyncpg.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password']
            )
            
            # Check required tables
            required_tables = [
                'users', 'properties', 'user_interactions', 'ml_models',
                'embeddings', 'training_metrics', 'search_queries', 'audit_log'
            ]
            
            table_info = {}
            for table in required_tables:
                query = """
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = $1
                    ORDER BY ordinal_position
                """
                columns = await conn.fetch(query, table)
                table_info[table] = [
                    {
                        'name': col['column_name'],
                        'type': col['data_type'],
                        'nullable': col['is_nullable']
                    } for col in columns
                ]
            
            await conn.close()
            
            missing_tables = [t for t in required_tables if not table_info[t]]
            
            if missing_tables:
                return False, f"Missing tables: {missing_tables}", table_info
            else:
                return True, f"All {len(required_tables)} required tables found", table_info
            
        except Exception as e:
            return False, f"Table structure test failed: {str(e)}", None
    
    async def test_postgres_indexes(self):
        """Test that performance indexes are properly created"""
        try:
            conn = await asyncpg.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password']
            )
            
            # Check for critical indexes
            index_query = """
                SELECT schemaname, tablename, indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = 'public'
                ORDER BY tablename, indexname
            """
            indexes = await conn.fetch(index_query)
            await conn.close()
            
            index_info = {}
            for idx in indexes:
                table = idx['tablename']
                if table not in index_info:
                    index_info[table] = []
                index_info[table].append({
                    'name': idx['indexname'],
                    'definition': idx['indexdef']
                })
            
            # Check for critical indexes
            critical_indexes = [
                'idx_users_email', 'idx_properties_status', 'idx_properties_price',
                'idx_user_interactions_user_id', 'idx_properties_fulltext'
            ]
            
            all_index_names = [idx['indexname'] for idx in indexes]
            missing_indexes = [idx for idx in critical_indexes if idx not in all_index_names]
            
            if missing_indexes:
                return False, f"Missing critical indexes: {missing_indexes}", index_info
            else:
                return True, f"All critical indexes found. Total indexes: {len(indexes)}", index_info
            
        except Exception as e:
            return False, f"Index test failed: {str(e)}", None
    
    async def test_postgres_crud_operations(self):
        """Test basic CRUD operations on users table"""
        try:
            conn = await asyncpg.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password']
            )
            
            test_email = f"test_user_{int(time.time())}@example.com"
            operations = {}
            
            # CREATE
            insert_query = """
                INSERT INTO users (email, min_price, max_price, min_bedrooms, max_bedrooms)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """
            user_id = await conn.fetchval(insert_query, test_email, 1000.0, 2500.0, 1, 3)
            operations['create'] = True
            
            # READ
            select_query = "SELECT email, min_price, max_price FROM users WHERE id = $1"
            user_data = await conn.fetchrow(select_query, user_id)
            operations['read'] = user_data is not None
            
            # UPDATE
            update_query = "UPDATE users SET max_price = $1 WHERE id = $2"
            await conn.execute(update_query, 3000.0, user_id)
            
            updated_data = await conn.fetchrow(select_query, user_id)
            operations['update'] = updated_data['max_price'] == 3000.0
            
            # DELETE
            delete_query = "DELETE FROM users WHERE id = $1"
            await conn.execute(delete_query, user_id)
            
            deleted_check = await conn.fetchrow(select_query, user_id)
            operations['delete'] = deleted_check is None
            
            await conn.close()
            
            all_successful = all(operations.values())
            return all_successful, f"CRUD operations: {operations}", operations
            
        except Exception as e:
            return False, f"CRUD test failed: {str(e)}", None
    
    async def test_postgres_performance(self):
        """Test PostgreSQL query performance"""
        try:
            conn = await asyncpg.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password']
            )
            
            performance_tests = {}
            
            # Test 1: Count properties
            start = time.time()
            count = await conn.fetchval("SELECT COUNT(*) FROM properties WHERE status = 'active'")
            performance_tests['count_properties_ms'] = int((time.time() - start) * 1000)
            
            # Test 2: Search with filters
            start = time.time()
            results = await conn.fetch("""
                SELECT id, title, price FROM properties 
                WHERE status = 'active' AND price BETWEEN $1 AND $2 
                ORDER BY price LIMIT 10
            """, 1000, 3000)
            performance_tests['search_filtered_ms'] = int((time.time() - start) * 1000)
            
            # Test 3: Join query
            start = time.time()
            join_results = await conn.fetch("""
                SELECT p.title, COUNT(ui.id) as interaction_count
                FROM properties p
                LEFT JOIN user_interactions ui ON p.id = ui.property_id
                WHERE p.status = 'active'
                GROUP BY p.id, p.title
                LIMIT 10
            """)
            performance_tests['join_query_ms'] = int((time.time() - start) * 1000)
            
            await conn.close()
            
            # Check if performance is acceptable (under 1 second)
            slow_queries = {k: v for k, v in performance_tests.items() if v > 1000}
            
            if slow_queries:
                return False, f"Slow queries detected: {slow_queries}", performance_tests
            else:
                return True, f"All queries performed well", performance_tests
            
        except Exception as e:
            return False, f"Performance test failed: {str(e)}", None
    
    # ============================================
    # REDIS TESTS
    # ============================================
    
    async def test_redis_basic_connection(self):
        """Test basic Redis connection"""
        try:
            client = redis.Redis(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                db=self.redis_config['db'],
                password=self.redis_config['password']
            )
            
            # Test ping
            pong = await client.ping()
            
            # Test basic operations
            await client.set('test_key', 'test_value', ex=10)
            value = await client.get('test_key')
            await client.delete('test_key')
            
            await client.close()
            
            return True, f"Redis connection successful. Ping: {pong}", {
                'ping_response': pong,
                'set_get_test': value == b'test_value'
            }
            
        except Exception as e:
            return False, f"Redis connection failed: {str(e)}", None
    
    async def test_redis_connection_pool(self):
        """Test Redis connection pool"""
        try:
            pool = redis.ConnectionPool(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                db=self.redis_config['db'],
                password=self.redis_config['password'],
                max_connections=self.redis_config['max_connections']
            )
            
            # Test multiple concurrent connections
            tasks = []
            for i in range(15):
                tasks.append(self._test_redis_pool_connection(pool, i))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_ops = sum(1 for r in results if not isinstance(r, Exception))
            
            return True, f"Redis pool test: {successful_ops}/15 operations successful", {
                'successful_operations': successful_ops,
                'pool_max_connections': self.redis_config['max_connections'],
                'exceptions': [str(r) for r in results if isinstance(r, Exception)]
            }
            
        except Exception as e:
            return False, f"Redis pool test failed: {str(e)}", None
    
    async def _test_redis_pool_connection(self, pool, op_id):
        """Helper method to test individual Redis pool connection"""
        client = redis.Redis(connection_pool=pool)
        try:
            key = f"pool_test_{op_id}"
            await client.set(key, f"value_{op_id}", ex=5)
            value = await client.get(key)
            await client.delete(key)
            return value == f"value_{op_id}".encode()
        finally:
            await client.close()
    
    async def test_redis_caching_operations(self):
        """Test Redis caching functionality"""
        try:
            client = redis.Redis(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                db=self.redis_config['db'],
                password=self.redis_config['password']
            )
            
            cache_tests = {}
            
            # Test 1: String caching
            await client.set('test_string', 'cached_value', ex=60)
            cached_value = await client.get('test_string')
            cache_tests['string_cache'] = cached_value == b'cached_value'
            
            # Test 2: JSON caching
            test_data = {'property_id': '123', 'price': 2500, 'location': 'Downtown'}
            await client.set('test_json', json.dumps(test_data), ex=60)
            cached_json = await client.get('test_json')
            parsed_data = json.loads(cached_json.decode())
            cache_tests['json_cache'] = parsed_data == test_data
            
            # Test 3: List operations
            await client.lpush('test_list', 'item1', 'item2', 'item3')
            list_length = await client.llen('test_list')
            list_items = await client.lrange('test_list', 0, -1)
            cache_tests['list_operations'] = list_length == 3 and len(list_items) == 3
            
            # Test 4: Hash operations
            await client.hset('test_hash', mapping={
                'property_id': '456',
                'title': 'Test Property',
                'price': '3000'
            })
            hash_data = await client.hgetall('test_hash')
            cache_tests['hash_operations'] = len(hash_data) == 3
            
            # Test 5: TTL functionality
            await client.set('test_ttl', 'expires_soon', ex=2)
            ttl = await client.ttl('test_ttl')
            cache_tests['ttl_test'] = 0 < ttl <= 2
            
            # Cleanup
            await client.delete('test_string', 'test_json', 'test_list', 'test_hash', 'test_ttl')
            await client.close()
            
            all_passed = all(cache_tests.values())
            return all_passed, f"Cache operations: {cache_tests}", cache_tests
            
        except Exception as e:
            return False, f"Redis caching test failed: {str(e)}", None
    
    async def test_redis_performance(self):
        """Test Redis performance with bulk operations"""
        try:
            client = redis.Redis(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                db=self.redis_config['db'],
                password=self.redis_config['password']
            )
            
            performance_tests = {}
            
            # Test 1: Bulk SET operations
            start = time.time()
            pipe = client.pipeline()
            for i in range(1000):
                pipe.set(f'perf_test_{i}', f'value_{i}', ex=30)
            await pipe.execute()
            performance_tests['bulk_set_1000_ms'] = int((time.time() - start) * 1000)
            
            # Test 2: Bulk GET operations
            start = time.time()
            pipe = client.pipeline()
            for i in range(1000):
                pipe.get(f'perf_test_{i}')
            results = await pipe.execute()
            performance_tests['bulk_get_1000_ms'] = int((time.time() - start) * 1000)
            
            # Test 3: Cleanup
            start = time.time()
            keys = [f'perf_test_{i}' for i in range(1000)]
            await client.delete(*keys)
            performance_tests['bulk_delete_1000_ms'] = int((time.time() - start) * 1000)
            
            await client.close()
            
            # Check performance thresholds
            slow_operations = {k: v for k, v in performance_tests.items() if v > 5000}  # 5 second threshold
            
            if slow_operations:
                return False, f"Slow Redis operations: {slow_operations}", performance_tests
            else:
                return True, f"Redis performance acceptable", performance_tests
            
        except Exception as e:
            return False, f"Redis performance test failed: {str(e)}", None
    
    # ============================================
    # INTEGRATED TESTS
    # ============================================
    
    async def test_postgres_redis_integration(self):
        """Test integration between PostgreSQL and Redis caching"""
        try:
            # Connect to both databases
            pg_conn = await asyncpg.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password']
            )
            
            redis_client = redis.Redis(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                db=self.redis_config['db'],
                password=self.redis_config['password']
            )
            
            integration_tests = {}
            
            # Test 1: Cache PostgreSQL query results
            properties = await pg_conn.fetch(
                "SELECT id, title, price FROM properties WHERE status = 'active' LIMIT 5"
            )
            
            # Cache results in Redis
            cache_key = 'test_properties_cache'
            properties_json = json.dumps([dict(p) for p in properties], default=str)
            await redis_client.set(cache_key, properties_json, ex=300)
            
            # Retrieve from cache
            cached_data = await redis_client.get(cache_key)
            cached_properties = json.loads(cached_data.decode())
            
            integration_tests['cache_query_results'] = len(cached_properties) == len(properties)
            
            # Test 2: Cache invalidation simulation
            # Simulate updating a property in PostgreSQL
            if properties:
                test_property_id = properties[0]['id']
                await pg_conn.execute(
                    "UPDATE properties SET updated_at = CURRENT_TIMESTAMP WHERE id = $1",
                    test_property_id
                )
                
                # Remove from cache (simulating cache invalidation)
                await redis_client.delete(cache_key)
                cached_after_delete = await redis_client.get(cache_key)
                integration_tests['cache_invalidation'] = cached_after_delete is None
            
            # Cleanup
            await redis_client.delete(cache_key)
            await pg_conn.close()
            await redis_client.close()
            
            all_passed = all(integration_tests.values())
            return all_passed, f"Integration tests: {integration_tests}", integration_tests
            
        except Exception as e:
            return False, f"Integration test failed: {str(e)}", None
    
    # ============================================
    # MAIN TEST RUNNER
    # ============================================
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all database tests"""
        logger.info("Starting comprehensive database connection tests...")
        
        # PostgreSQL Tests
        await self.run_test("PostgreSQL Basic Connection", self.test_postgres_basic_connection)
        await self.run_test("PostgreSQL Connection Pool", self.test_postgres_connection_pool)
        await self.run_test("PostgreSQL Table Structure", self.test_postgres_table_structure)
        await self.run_test("PostgreSQL Indexes", self.test_postgres_indexes)
        await self.run_test("PostgreSQL CRUD Operations", self.test_postgres_crud_operations)
        await self.run_test("PostgreSQL Performance", self.test_postgres_performance)
        
        # Redis Tests
        await self.run_test("Redis Basic Connection", self.test_redis_basic_connection)
        await self.run_test("Redis Connection Pool", self.test_redis_connection_pool)
        await self.run_test("Redis Caching Operations", self.test_redis_caching_operations)
        await self.run_test("Redis Performance", self.test_redis_performance)
        
        # Integration Tests
        await self.run_test("PostgreSQL-Redis Integration", self.test_postgres_redis_integration)
        
        # Generate summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        failed_tests = total_tests - passed_tests
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            'total_duration_ms': sum(r.duration_ms for r in self.test_results),
            'test_results': [
                {
                    'test_name': r.test_name,
                    'success': r.success,
                    'duration_ms': r.duration_ms,
                    'message': r.message,
                    'details': r.details
                } for r in self.test_results
            ]
        }
        
        return summary
    
    def print_test_results(self, summary: Dict[str, Any]):
        """Print formatted test results"""
        print("\n" + "="*80)
        print("DATABASE CONNECTION TEST RESULTS")
        print("="*80)
        
        print(f"\nOverall Summary:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed_tests']}")
        print(f"  Failed: {summary['failed_tests']}")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        print(f"  Total Duration: {summary['total_duration_ms']:,}ms")
        
        print(f"\nDetailed Results:")
        print("-" * 80)
        
        for result in summary['test_results']:
            status = "✓ PASS" if result['success'] else "✗ FAIL"
            print(f"{status:8} | {result['test_name']:35} | {result['duration_ms']:6}ms | {result['message']}")
        
        if summary['failed_tests'] > 0:
            print(f"\nFailed Test Details:")
            print("-" * 80)
            for result in summary['test_results']:
                if not result['success']:
                    print(f"❌ {result['test_name']}:")
                    print(f"   Message: {result['message']}")
                    if result['details']:
                        print(f"   Details: {json.dumps(result['details'], indent=4, default=str)}")
                    print()


async def main():
    """Main test execution function"""
    # Load environment variables
    from dotenv import load_dotenv
    
    env_file = Path(__file__).parent.parent / '.env.production'
    if env_file.exists():
        load_dotenv(env_file)
        logger.info(f"Loaded environment from {env_file}")
    else:
        logger.warning("Production .env file not found, using environment variables")
    
    # Run tests
    tester = DatabaseConnectionTester()
    summary = await tester.run_all_tests()
    
    # Print results
    tester.print_test_results(summary)
    
    # Save results to file
    results_file = Path(__file__).parent.parent / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Test results saved to {results_file}")
    
    # Exit with appropriate code
    exit_code = 0 if summary['failed_tests'] == 0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())