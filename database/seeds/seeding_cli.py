#!/usr/bin/env python3
"""
Production Data Seeding CLI Tool
Command-line interface for managing database seeding operations
"""

import asyncio
import click
import logging
import sys
import json
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from database.seeds.production_data_seeder import (
    ProductionDataSeeder, SeedingMode, DataQuality
)
from config.database.connection_manager import get_connection_manager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('seeding.log')
    ]
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """Production Database Seeding Tool"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    ctx.ensure_object(dict)


@cli.command()
@click.option('--mode', '-m', 
              type=click.Choice(['development', 'staging', 'performance_testing', 'load_testing']),
              default='development',
              help='Seeding mode')
@click.option('--tables', '-t', multiple=True, help='Specific tables to seed')
@click.option('--incremental', is_flag=True, help='Add to existing data (do not clear)')
@click.option('--dry-run', is_flag=True, help='Show what would be seeded without executing')
@click.pass_context
async def seed(ctx, mode, tables, incremental, dry_run):
    """Seed database with realistic data"""
    try:
        # Convert mode string to enum
        seeding_mode = SeedingMode(mode)
        
        connection_manager = await get_connection_manager()
        seeder = ProductionDataSeeder(connection_manager)
        
        # Show seeding plan
        config = seeder.seeding_configs[seeding_mode]
        
        click.echo(f"\n🌱 Database Seeding Plan:")
        click.echo(f"  Mode: {mode}")
        click.echo(f"  Quality: {config.data_quality.value}")
        click.echo(f"  Batch size: {config.batch_size}")
        click.echo(f"  Max workers: {config.max_workers}")
        click.echo(f"  Incremental: {incremental}")
        
        if tables:
            click.echo(f"  Tables: {', '.join(tables)}")
        else:
            click.echo(f"  Tables: all configured tables")
            
        click.echo(f"\n📊 Data Distribution:")
        tables_to_show = tables if tables else config.data_distribution.keys()
        total_records = 0
        
        for table in tables_to_show:
            if table in config.data_distribution:
                count = config.data_distribution[table]
                click.echo(f"    {table}: {count:,} records")
                total_records += count
                
        click.echo(f"  Total: {total_records:,} records")
        
        if dry_run:
            click.echo("\n🧪 Dry run mode - no data will be created")
            return
            
        # Confirm execution
        if not click.confirm(f"\nProceed with seeding {total_records:,} records in {mode} mode?"):
            click.echo("Seeding cancelled")
            return
            
        # Execute seeding
        click.echo(f"\n🚀 Starting database seeding...")
        start_time = datetime.now()
        
        results = await seeder.seed_database(
            mode=seeding_mode,
            tables=list(tables) if tables else None,
            incremental=incremental
        )
        
        # Display results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        click.echo(f"\n✅ Seeding completed in {duration:.2f} seconds")
        click.echo(f"   Total records created: {results['total_records_created']:,}")
        
        if results.get('performance_metrics'):
            metrics = results['performance_metrics']
            click.echo(f"   Records per second: {metrics.get('total_records_per_second', 0):,.0f}")
            click.echo(f"   Database size: {metrics.get('database_size_after_mb', 0):.1f} MB")
            
        # Show table breakdown
        if results.get('tables_seeded'):
            click.echo(f"\n📋 Table Results:")
            for table, stats in results['tables_seeded'].items():
                rps = stats.get('records_per_second', 0)
                click.echo(f"    {table}: {stats['records_created']:,} records ({rps:,.0f} rec/s)")
                
        # Show errors if any
        if results.get('errors'):
            click.echo(f"\n❌ Errors ({len(results['errors'])}):")
            for error in results['errors']:
                click.echo(f"    {error}")
                
        # Show validation results
        if results.get('validation'):
            validation = results['validation']
            if validation['passed']:
                click.echo(f"\n✅ Data integrity validation passed")
            else:
                click.echo(f"\n❌ Data integrity validation failed:")
                for error in validation.get('errors', []):
                    click.echo(f"    {error}")
                    
    except Exception as e:
        click.echo(f"❌ Seeding failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
async def status(ctx):
    """Show seeding progress and status"""
    try:
        connection_manager = await get_connection_manager()
        seeder = ProductionDataSeeder(connection_manager)
        
        # Get database statistics
        async with connection_manager.get_connection(analytics=True) as conn:
            stats = await conn.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    n_live_tup as row_count,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                FROM pg_stat_user_tables 
                WHERE schemaname = 'public'
                ORDER BY n_live_tup DESC
            """)
            
        click.echo(f"\n📊 Database Status:")
        total_rows = 0
        
        for stat in stats:
            row_count = stat['row_count'] or 0
            total_rows += row_count
            click.echo(f"  {stat['tablename']}: {row_count:,} rows ({stat['size']})")
            
        click.echo(f"  Total: {total_rows:,} rows")
        
        # Get current seeding progress if any
        progress = await seeder.get_seeding_progress()
        
        if progress:
            click.echo(f"\n🔄 Active Seeding Operations:")
            for table, prog in progress.items():
                if prog['progress_percent'] < 100:
                    click.echo(f"  {table}: {prog['progress_percent']:.1f}% "
                             f"({prog['completed_records']:,}/{prog['total_records']:,})")
        else:
            click.echo(f"\n✅ No active seeding operations")
            
    except Exception as e:
        click.echo(f"❌ Failed to get status: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--tables', '-t', multiple=True, help='Specific tables to clear')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
async def clear(ctx, tables, confirm):
    """Clear seeded data from database"""
    try:
        connection_manager = await get_connection_manager()
        
        # Show what will be cleared
        if tables:
            tables_list = list(tables)
        else:
            tables_list = ['users', 'properties', 'user_interactions', 'search_queries', 
                          'ml_models', 'training_metrics', 'embeddings', 'audit_log']
            
        # Get current counts
        async with connection_manager.get_connection(analytics=True) as conn:
            counts = {}
            for table in tables_list:
                try:
                    count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                    counts[table] = count
                except:
                    counts[table] = 0
                    
        click.echo(f"\n🗑️  Data to be cleared:")
        total_records = 0
        for table in tables_list:
            count = counts.get(table, 0)
            total_records += count
            click.echo(f"    {table}: {count:,} records")
            
        click.echo(f"  Total: {total_records:,} records")
        
        if total_records == 0:
            click.echo("✅ No data to clear")
            return
            
        # Confirm deletion
        if not confirm:
            click.echo(f"\n⚠️  WARNING: This will permanently delete {total_records:,} records!")
            if not click.confirm("Are you sure you want to proceed?"):
                click.echo("Clear operation cancelled")
                return
                
        # Clear data
        click.echo(f"\n🗑️  Clearing data...")
        
        async with connection_manager.get_connection() as conn:
            # Disable triggers for faster deletion
            await conn.execute("SET session_replication_role = replica")
            
            try:
                for table in reversed(tables_list):  # Reverse order for dependencies
                    try:
                        await conn.execute(f"TRUNCATE TABLE {table} CASCADE")
                        click.echo(f"  ✅ Cleared {table}")
                    except Exception as e:
                        click.echo(f"  ❌ Failed to clear {table}: {e}")
                        
            finally:
                # Re-enable triggers
                await conn.execute("SET session_replication_role = DEFAULT")
                
        click.echo(f"\n✅ Data clearing completed")
        
    except Exception as e:
        click.echo(f"❌ Clear operation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
async def validate(ctx):
    """Validate database integrity and data quality"""
    try:
        connection_manager = await get_connection_manager()
        seeder = ProductionDataSeeder(connection_manager)
        
        click.echo(f"\n🔍 Validating database integrity...")
        
        # Check all main tables
        tables = ['users', 'properties', 'user_interactions', 'search_queries', 
                 'ml_models', 'training_metrics', 'embeddings']
        
        validation_results = await seeder._validate_data_integrity(tables)
        
        # Display results
        click.echo(f"\n📊 Table Counts:")
        for table, count in validation_results['table_counts'].items():
            click.echo(f"  {table}: {count:,} records")
            
        click.echo(f"\n🔗 Integrity Checks:")
        all_passed = True
        
        for check, result in validation_results['integrity_checks'].items():
            if result == 0:
                click.echo(f"  ✅ {check}: passed")
            else:
                click.echo(f"  ❌ {check}: {result} issues found")
                all_passed = False
                
        # Additional checks
        async with connection_manager.get_connection(analytics=True) as conn:
            # Check for null values in critical fields
            critical_checks = [
                ("users", "email", "SELECT COUNT(*) FROM users WHERE email IS NULL"),
                ("properties", "price", "SELECT COUNT(*) FROM properties WHERE price IS NULL OR price <= 0"),
                ("user_interactions", "timestamp", "SELECT COUNT(*) FROM user_interactions WHERE timestamp IS NULL"),
            ]
            
            click.echo(f"\n🎯 Data Quality Checks:")
            for table, field, check_sql in critical_checks:
                try:
                    issues = await conn.fetchval(check_sql)
                    if issues == 0:
                        click.echo(f"  ✅ {table}.{field}: no issues")
                    else:
                        click.echo(f"  ❌ {table}.{field}: {issues} issues found")
                        all_passed = False
                except Exception as e:
                    click.echo(f"  ⚠️  {table}.{field}: check failed ({e})")
                    
        # Overall result
        if all_passed and validation_results['passed']:
            click.echo(f"\n✅ All validation checks passed")
        else:
            click.echo(f"\n❌ Validation issues found:")
            for error in validation_results.get('errors', []):
                click.echo(f"    {error}")
                
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', help='Output file for statistics')
@click.pass_context
async def stats(ctx, output):
    """Show detailed database statistics"""
    try:
        connection_manager = await get_connection_manager()
        
        stats_data = {
            'generated_at': datetime.now().isoformat(),
            'table_statistics': {},
            'database_size': {},
            'index_statistics': {},
            'performance_metrics': {}
        }
        
        async with connection_manager.get_connection(analytics=True) as conn:
            # Table statistics
            table_stats = await conn.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    n_live_tup as row_count,
                    n_dead_tup as dead_rows,
                    n_tup_ins as inserts,
                    n_tup_upd as updates,
                    n_tup_del as deletes,
                    last_vacuum,
                    last_autovacuum,
                    last_analyze,
                    last_autoanalyze,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
                    pg_total_relation_size(schemaname||'.'||tablename) as total_size_bytes
                FROM pg_stat_user_tables 
                WHERE schemaname = 'public'
                ORDER BY n_live_tup DESC
            """)
            
            for stat in table_stats:
                table_name = stat['tablename']
                stats_data['table_statistics'][table_name] = {
                    'row_count': stat['row_count'] or 0,
                    'dead_rows': stat['dead_rows'] or 0,
                    'total_operations': (stat['inserts'] or 0) + (stat['updates'] or 0) + (stat['deletes'] or 0),
                    'total_size': stat['total_size'],
                    'total_size_bytes': stat['total_size_bytes'] or 0,
                    'last_vacuum': stat['last_vacuum'].isoformat() if stat['last_vacuum'] else None,
                    'last_analyze': stat['last_analyze'].isoformat() if stat['last_analyze'] else None
                }
                
            # Database size
            db_size = await conn.fetchrow("""
                SELECT 
                    pg_database_size(current_database()) as db_size_bytes,
                    pg_size_pretty(pg_database_size(current_database())) as db_size
            """)
            
            stats_data['database_size'] = {
                'total_size': db_size['db_size'],
                'total_size_bytes': db_size['db_size_bytes']
            }
            
            # Index statistics
            index_stats = await conn.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch,
                    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
                FROM pg_stat_user_indexes
                ORDER BY idx_scan DESC
                LIMIT 20
            """)
            
            for idx_stat in index_stats:
                index_name = idx_stat['indexname']
                stats_data['index_statistics'][index_name] = {
                    'table': idx_stat['tablename'],
                    'scans': idx_stat['idx_scan'] or 0,
                    'tuples_read': idx_stat['idx_tup_read'] or 0,
                    'tuples_fetched': idx_stat['idx_tup_fetch'] or 0,
                    'size': idx_stat['index_size']
                }
                
        # Display or save statistics
        if output:
            with open(output, 'w') as f:
                json.dump(stats_data, f, indent=2)
            click.echo(f"✅ Statistics saved to {output}")
        else:
            # Display summary
            click.echo(f"\n📊 Database Statistics:")
            click.echo(f"  Database size: {stats_data['database_size']['total_size']}")
            click.echo(f"  Tables: {len(stats_data['table_statistics'])}")
            
            click.echo(f"\n📋 Top Tables by Size:")
            sorted_tables = sorted(
                stats_data['table_statistics'].items(),
                key=lambda x: x[1]['total_size_bytes'],
                reverse=True
            )
            
            for table_name, table_stats in sorted_tables[:10]:
                click.echo(f"  {table_name}: {table_stats['row_count']:,} rows "
                         f"({table_stats['total_size']})")
                         
            click.echo(f"\n🔍 Most Used Indexes:")
            sorted_indexes = sorted(
                stats_data['index_statistics'].items(),
                key=lambda x: x[1]['scans'],
                reverse=True
            )
            
            for index_name, index_stats in sorted_indexes[:10]:
                click.echo(f"  {index_name}: {index_stats['scans']:,} scans "
                         f"({index_stats['size']})")
                         
    except Exception as e:
        click.echo(f"❌ Failed to get statistics: {e}", err=True)
        sys.exit(1)


def run_async_command(coro):
    """Helper to run async commands"""
    try:
        asyncio.run(coro)
    except KeyboardInterrupt:
        click.echo("\n⚠️  Operation cancelled by user")
        sys.exit(1)


# Wrap async commands
for command_name in ['seed', 'status', 'clear', 'validate', 'stats']:
    command = cli.commands[command_name]
    original_callback = command.callback
    
    def make_wrapper(original):
        def wrapper(*args, **kwargs):
            return run_async_command(original(*args, **kwargs))
        return wrapper
    
    command.callback = make_wrapper(original_callback)


if __name__ == '__main__':
    cli()