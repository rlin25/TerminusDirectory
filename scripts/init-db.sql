-- =============================================================================
-- PostgreSQL Database Initialization Script
-- =============================================================================
-- This script sets up the initial database structure and configurations

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create application schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS public;

-- Set default search path
SET search_path = public;

-- =============================================================================
-- Database Configuration
-- =============================================================================

-- Set timezone
SET timezone = 'UTC';

-- Configure connection limits
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Apply configuration changes
SELECT pg_reload_conf();

-- =============================================================================
-- Create Application User (if running as superuser)
-- =============================================================================

DO $$
BEGIN
    -- Create application user if it doesn't exist
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'rental_ml_app') THEN
        CREATE ROLE rental_ml_app WITH LOGIN PASSWORD 'app_user_password';
    END IF;
    
    -- Grant necessary permissions
    GRANT CONNECT ON DATABASE rental_ml TO rental_ml_app;
    GRANT USAGE ON SCHEMA public TO rental_ml_app;
    GRANT CREATE ON SCHEMA public TO rental_ml_app;
    
    -- Grant permissions on existing tables
    GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO rental_ml_app;
    GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO rental_ml_app;
    
    -- Grant permissions on future tables
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO rental_ml_app;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO rental_ml_app;
    
EXCEPTION WHEN insufficient_privilege THEN
    -- If we don't have superuser privileges, just continue
    RAISE NOTICE 'Could not create application user (insufficient privileges)';
END $$;

-- =============================================================================
-- Logging Configuration
-- =============================================================================

-- Enable query logging for slow queries
ALTER SYSTEM SET log_min_duration_statement = '1s';
ALTER SYSTEM SET log_statement = 'mod';
ALTER SYSTEM SET log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h ';
ALTER SYSTEM SET log_checkpoints = on;
ALTER SYSTEM SET log_connections = on;
ALTER SYSTEM SET log_disconnections = on;
ALTER SYSTEM SET log_lock_waits = on;

-- =============================================================================
-- Performance Monitoring
-- =============================================================================

-- Create extension for query statistics (if available)
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- =============================================================================
-- Initial Database Comments
-- =============================================================================

COMMENT ON DATABASE rental_ml IS 'Rental ML System - Machine Learning Powered Rental Property Platform';

-- =============================================================================
-- Success Message
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '==================================================';
    RAISE NOTICE 'Database initialization completed successfully!';
    RAISE NOTICE 'Database: rental_ml';
    RAISE NOTICE 'Extensions: uuid-ossp, pg_trgm, btree_gin';
    RAISE NOTICE 'Application ready for table creation and data loading';
    RAISE NOTICE '==================================================';
    RAISE NOTICE '';
END $$;