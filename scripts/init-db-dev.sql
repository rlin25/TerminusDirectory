-- =============================================================================
-- PostgreSQL Development Database Initialization Script
-- =============================================================================
-- This script sets up the development database with additional debugging features

-- Include base initialization
\i /docker-entrypoint-initdb.d/init-db.sql

-- =============================================================================
-- Development-Specific Configuration
-- =============================================================================

-- Enable more verbose logging for development
ALTER SYSTEM SET log_min_duration_statement = '100ms';
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_min_messages = 'info';
ALTER SYSTEM SET log_error_verbosity = 'verbose';

-- Enable auto-explain for development
CREATE EXTENSION IF NOT EXISTS auto_explain;
ALTER SYSTEM SET session_preload_libraries = 'auto_explain';
ALTER SYSTEM SET auto_explain.log_min_duration = '500ms';
ALTER SYSTEM SET auto_explain.log_analyze = true;
ALTER SYSTEM SET auto_explain.log_buffers = true;
ALTER SYSTEM SET auto_explain.log_timing = true;
ALTER SYSTEM SET auto_explain.log_triggers = true;
ALTER SYSTEM SET auto_explain.log_verbose = true;

-- Apply configuration
SELECT pg_reload_conf();

-- =============================================================================
-- Development User and Permissions
-- =============================================================================

-- Create development user with more permissions
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'dev_user') THEN
        CREATE ROLE dev_user WITH LOGIN PASSWORD 'dev_password' SUPERUSER;
    END IF;
    
    -- Grant all permissions for development
    GRANT ALL PRIVILEGES ON DATABASE rental_ml_dev TO dev_user;
    
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Could not create development user: %', SQLERRM;
END $$;

-- =============================================================================
-- Development Logging
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '==================================================';
    RAISE NOTICE 'Development database initialization completed!';
    RAISE NOTICE 'Database: rental_ml_dev';
    RAISE NOTICE 'Development features enabled:';
    RAISE NOTICE '  - Verbose logging';
    RAISE NOTICE '  - Auto-explain for slow queries';
    RAISE NOTICE '  - Extended debugging information';
    RAISE NOTICE '==================================================';
    RAISE NOTICE '';
END $$;