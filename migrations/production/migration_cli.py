#!/usr/bin/env python3
"""
Production Migration CLI Tool
Command-line interface for managing database migrations safely in production
"""

import asyncio
import click
import logging
import sys
import os
import json
from datetime import datetime
from typing import Optional, List
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from migrations.production.migration_framework import (
    ProductionMigrationManager, MigrationStatus, MigrationSafety
)
from config.database.connection_manager import get_connection_manager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('migration.log')
    ]
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', help='Migration config file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Production Database Migration Manager"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    ctx.ensure_object(dict)
    ctx.obj['config'] = config


@cli.command()
@click.pass_context
async def init(ctx):
    """Initialize migration system"""
    try:
        connection_manager = await get_connection_manager()
        migration_manager = ProductionMigrationManager(
            connection_manager, ctx.obj.get('config')
        )
        
        await migration_manager.initialize()
        click.echo("✅ Migration system initialized successfully")
        
    except Exception as e:
        click.echo(f"❌ Failed to initialize migration system: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
async def list(ctx):
    """List all available migrations"""
    try:
        connection_manager = await get_connection_manager()
        migration_manager = ProductionMigrationManager(
            connection_manager, ctx.obj.get('config')
        )
        
        await migration_manager.initialize()
        
        # Get all migrations
        all_migrations = migration_manager.discover_migrations()
        pending_migrations = await migration_manager.get_pending_migrations()
        
        # Display summary
        click.echo("\n📋 Migration Summary:")
        click.echo(f"  Total migrations: {len(all_migrations)}")
        click.echo(f"  Pending migrations: {len(pending_migrations)}")
        click.echo(f"  Applied migrations: {len(all_migrations) - len(pending_migrations)}")
        
        if pending_migrations:
            click.echo("\n⏳ Pending Migrations:")
            for migration in pending_migrations:
                safety_emoji = {
                    MigrationSafety.SAFE: "🟢",
                    MigrationSafety.CAUTION: "🟡", 
                    MigrationSafety.DANGEROUS: "🔴"
                }
                
                click.echo(f"  {safety_emoji[migration.safety_level]} {migration.version}")
                click.echo(f"    Name: {migration.name}")
                click.echo(f"    Safety: {migration.safety_level.value}")
                click.echo(f"    Duration: ~{migration.estimated_duration}s")
                if migration.dependencies:
                    click.echo(f"    Dependencies: {', '.join(migration.dependencies)}")
                click.echo()
        else:
            click.echo("\n✅ All migrations are up to date")
            
    except Exception as e:
        click.echo(f"❌ Failed to list migrations: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('version', required=False)
@click.option('--dry-run', is_flag=True, help='Perform dry run without executing')
@click.option('--force', is_flag=True, help='Force execution of dangerous migrations')
@click.option('--auto-approve', is_flag=True, help='Skip interactive approval')
@click.pass_context
async def migrate(ctx, version, dry_run, force, auto_approve):
    """Run pending migrations or specific migration"""
    try:
        connection_manager = await get_connection_manager()
        migration_manager = ProductionMigrationManager(
            connection_manager, ctx.obj.get('config')
        )
        
        await migration_manager.initialize()
        
        # Determine which migrations to run
        if version:
            # Run specific migration
            all_migrations = migration_manager.discover_migrations()
            target_migration = None
            for m in all_migrations:
                if m.version == version:
                    target_migration = m
                    break
                    
            if not target_migration:
                click.echo(f"❌ Migration {version} not found", err=True)
                sys.exit(1)
                
            migrations_to_run = [target_migration]
        else:
            # Run all pending migrations
            migrations_to_run = await migration_manager.get_pending_migrations()
            
        if not migrations_to_run:
            click.echo("✅ No migrations to run")
            return
            
        # Display migration plan
        click.echo("\n📋 Migration Plan:")
        total_duration = 0
        has_dangerous = False
        
        for migration in migrations_to_run:
            safety_emoji = {
                MigrationSafety.SAFE: "🟢",
                MigrationSafety.CAUTION: "🟡",
                MigrationSafety.DANGEROUS: "🔴"
            }
            
            if migration.safety_level == MigrationSafety.DANGEROUS:
                has_dangerous = True
                
            click.echo(f"  {safety_emoji[migration.safety_level]} {migration.version} - {migration.name}")
            click.echo(f"    Duration: ~{migration.estimated_duration}s")
            if migration.requires_downtime:
                click.echo("    ⚠️  Requires downtime")
            total_duration += migration.estimated_duration
            
        click.echo(f"\nTotal estimated duration: {total_duration}s")
        
        # Check for dangerous migrations
        if has_dangerous and not force:
            click.echo("\n🔴 Dangerous migrations detected!")
            click.echo("Use --force to proceed with dangerous migrations")
            sys.exit(1)
            
        # Dry run mode
        if dry_run:
            click.echo("\n🧪 Dry run mode - no changes will be made")
            
        # Interactive approval
        if not auto_approve and not dry_run:
            if not click.confirm(f"\nProceed with {len(migrations_to_run)} migration(s)?"):
                click.echo("Migration cancelled")
                return
                
        # Execute migrations
        click.echo(f"\n🚀 {'Dry run:' if dry_run else 'Executing'} {len(migrations_to_run)} migration(s)...")
        
        for i, migration in enumerate(migrations_to_run, 1):
            click.echo(f"\n[{i}/{len(migrations_to_run)}] Running {migration.version}...")
            
            try:
                # Validate before execution
                valid, issues = await migration_manager.validate_migration(migration)
                if not valid:
                    click.echo(f"❌ Validation failed: {', '.join(issues)}", err=True)
                    continue
                    
                # Execute migration
                execution = await migration_manager.execute_migration(migration, dry_run=dry_run)
                
                if execution.status == MigrationStatus.COMPLETED:
                    duration = execution.duration_seconds or 0
                    click.echo(f"✅ Completed in {duration:.2f}s")
                    if execution.affected_rows:
                        click.echo(f"   Affected rows: {execution.affected_rows}")
                else:
                    click.echo(f"❌ Failed: {execution.error_message}", err=True)
                    
            except Exception as e:
                click.echo(f"❌ Migration {migration.version} failed: {e}", err=True)
                
        click.echo("\n🎉 Migration process completed")
        
    except Exception as e:
        click.echo(f"❌ Migration failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('version')
@click.option('--auto-approve', is_flag=True, help='Skip interactive approval')
@click.pass_context
async def rollback(ctx, version, auto_approve):
    """Rollback a specific migration"""
    try:
        connection_manager = await get_connection_manager()
        migration_manager = ProductionMigrationManager(
            connection_manager, ctx.obj.get('config')
        )
        
        await migration_manager.initialize()
        
        # Check if migration exists and is applied
        applied_migrations = [exec for exec in migration_manager.migration_history
                            if exec.status == MigrationStatus.COMPLETED]
        
        target_migration = None
        for exec in applied_migrations:
            if exec.version == version:
                target_migration = exec
                break
                
        if not target_migration:
            click.echo(f"❌ Migration {version} not found or not applied", err=True)
            sys.exit(1)
            
        if target_migration.rollback_executed:
            click.echo(f"❌ Migration {version} already rolled back", err=True)
            sys.exit(1)
            
        # Check if rollback file exists
        rollback_file = Path(f"migrations/production/scripts/{version}_rollback.sql")
        if not rollback_file.exists():
            click.echo(f"❌ Rollback file not found: {rollback_file}", err=True)
            sys.exit(1)
            
        click.echo(f"\n⚠️  Rollback Plan:")
        click.echo(f"  Migration: {version}")
        click.echo(f"  Applied at: {target_migration.started_at}")
        click.echo(f"  Duration: {target_migration.duration_seconds:.2f}s")
        click.echo(f"  Affected rows: {target_migration.affected_rows}")
        
        # Interactive approval
        if not auto_approve:
            click.echo("\n🚨 WARNING: Rollback operations can cause data loss!")
            if not click.confirm("Are you sure you want to proceed with rollback?"):
                click.echo("Rollback cancelled")
                return
                
        # Execute rollback
        click.echo(f"\n🔄 Rolling back migration {version}...")
        
        execution = await migration_manager.rollback_migration(version)
        
        if execution.status == MigrationStatus.COMPLETED:
            duration = execution.duration_seconds or 0
            click.echo(f"✅ Rollback completed in {duration:.2f}s")
        else:
            click.echo(f"❌ Rollback failed: {execution.error_message}", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Rollback failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
async def status(ctx):
    """Show migration system status"""
    try:
        connection_manager = await get_connection_manager()
        migration_manager = ProductionMigrationManager(
            connection_manager, ctx.obj.get('config')
        )
        
        await migration_manager.initialize()
        
        status = await migration_manager.get_migration_status()
        
        click.echo("\n📊 Migration System Status:")
        click.echo(f"  Total migrations: {status['total_migrations']}")
        click.echo(f"  Completed: {status['completed_migrations']}")
        click.echo(f"  Failed: {status['failed_migrations']}")
        click.echo(f"  Rolled back: {status['rolled_back_migrations']}")
        click.echo(f"  Pending: {status['pending_migrations']}")
        
        if status['currently_running']:
            click.echo(f"  Currently running: {status['currently_running']}")
            for running in status['running_migrations']:
                click.echo(f"    - {running}")
                
        if status['last_migration']:
            click.echo(f"  Last migration: {status['last_migration']}")
            
        if status['pending_migration_versions']:
            click.echo(f"\n⏳ Pending migrations:")
            for version in status['pending_migration_versions']:
                click.echo(f"    - {version}")
                
    except Exception as e:
        click.echo(f"❌ Failed to get status: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('version')
@click.pass_context
async def validate(ctx, version):
    """Validate a specific migration"""
    try:
        connection_manager = await get_connection_manager()
        migration_manager = ProductionMigrationManager(
            connection_manager, ctx.obj.get('config')
        )
        
        await migration_manager.initialize()
        
        # Find migration
        all_migrations = migration_manager.discover_migrations()
        target_migration = None
        for m in all_migrations:
            if m.version == version:
                target_migration = m
                break
                
        if not target_migration:
            click.echo(f"❌ Migration {version} not found", err=True)
            sys.exit(1)
            
        # Validate migration
        click.echo(f"\n🔍 Validating migration {version}...")
        
        valid, issues = await migration_manager.validate_migration(target_migration)
        
        if valid:
            click.echo("✅ Migration validation passed")
        else:
            click.echo("❌ Migration validation failed:")
            for issue in issues:
                click.echo(f"  - {issue}")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', help='Output file for history export')
@click.pass_context
async def history(ctx, output):
    """Show migration history"""
    try:
        connection_manager = await get_connection_manager()
        migration_manager = ProductionMigrationManager(
            connection_manager, ctx.obj.get('config')
        )
        
        await migration_manager.initialize()
        
        history = migration_manager.migration_history
        
        if output:
            # Export to file
            history_data = []
            for exec in history:
                history_data.append({
                    'version': exec.version,
                    'status': exec.status.value,
                    'started_at': exec.started_at.isoformat() if exec.started_at else None,
                    'completed_at': exec.completed_at.isoformat() if exec.completed_at else None,
                    'duration_seconds': exec.duration_seconds,
                    'error_message': exec.error_message,
                    'rollback_executed': exec.rollback_executed,
                    'affected_rows': exec.affected_rows
                })
                
            with open(output, 'w') as f:
                json.dump(history_data, f, indent=2)
                
            click.echo(f"✅ Migration history exported to {output}")
        else:
            # Display in terminal
            click.echo("\n📜 Migration History:")
            
            if not history:
                click.echo("  No migrations in history")
                return
                
            for exec in sorted(history, key=lambda x: x.started_at or datetime.min):
                status_emoji = {
                    MigrationStatus.COMPLETED: "✅",
                    MigrationStatus.FAILED: "❌",
                    MigrationStatus.ROLLED_BACK: "🔄"
                }
                
                emoji = status_emoji.get(exec.status, "❓")
                
                click.echo(f"  {emoji} {exec.version}")
                if exec.started_at:
                    click.echo(f"    Started: {exec.started_at}")
                if exec.duration_seconds:
                    click.echo(f"    Duration: {exec.duration_seconds:.2f}s")
                if exec.affected_rows:
                    click.echo(f"    Rows affected: {exec.affected_rows}")
                if exec.error_message:
                    click.echo(f"    Error: {exec.error_message}")
                if exec.rollback_executed:
                    click.echo("    🔄 Rolled back")
                click.echo()
                
    except Exception as e:
        click.echo(f"❌ Failed to get history: {e}", err=True)
        sys.exit(1)


def run_async_command(coro):
    """Helper to run async commands"""
    try:
        asyncio.run(coro)
    except KeyboardInterrupt:
        click.echo("\n⚠️  Operation cancelled by user")
        sys.exit(1)


# Wrap async commands
for command_name in ['init', 'list', 'migrate', 'rollback', 'status', 'validate', 'history']:
    command = cli.commands[command_name]
    original_callback = command.callback
    
    def make_wrapper(original):
        def wrapper(*args, **kwargs):
            return run_async_command(original(*args, **kwargs))
        return wrapper
    
    command.callback = make_wrapper(original_callback)


if __name__ == '__main__':
    cli()