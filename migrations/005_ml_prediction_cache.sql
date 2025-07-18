-- Migration: 005_ml_prediction_cache.sql
-- Description: Add prediction cache table and enhanced ML model storage
-- Created: 2025-07-14
-- Author: ML System Enhancement

-- ============================================
-- PREDICTION CACHE TABLE
-- ============================================
-- Create prediction cache table for ML model predictions
CREATE TABLE prediction_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cache_key VARCHAR(512) NOT NULL UNIQUE,
    predictions BYTEA NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    cached_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT prediction_cache_expires_after_cached CHECK (expires_at > cached_at),
    CONSTRAINT prediction_cache_access_count_positive CHECK (access_count >= 0)
);

-- ============================================
-- INDEXES FOR PREDICTION CACHE
-- ============================================
-- Primary access pattern: lookup by cache key
CREATE INDEX idx_prediction_cache_key ON prediction_cache(cache_key);

-- Cleanup expired entries efficiently
CREATE INDEX idx_prediction_cache_expires ON prediction_cache(expires_at);

-- Monitor access patterns
CREATE INDEX idx_prediction_cache_accessed ON prediction_cache(last_accessed);

-- Composite index for cache management
CREATE INDEX idx_prediction_cache_expires_accessed ON prediction_cache(expires_at, last_accessed);

-- ============================================
-- ENHANCED ML MODELS TABLE INDEXES
-- ============================================
-- Add additional indexes to existing ml_models table if it exists
DO $$ 
BEGIN
    -- Check if ml_models table exists and add indexes
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'ml_models') THEN
        -- Composite index for model name and version lookups
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_models_name_version_active 
        ON ml_models(model_name, version, is_active);
        
        -- Index for active models ordered by creation date
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_models_active_created 
        ON ml_models(is_active, created_at DESC) WHERE is_active = true;
        
        -- GIN index for metadata searching
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_models_metadata_gin 
        ON ml_models USING gin (metadata);
    END IF;
END $$;

-- ============================================
-- ENHANCED EMBEDDINGS TABLE INDEXES
-- ============================================
-- Add additional indexes to existing embeddings table if it exists
DO $$ 
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'embeddings') THEN
        -- Composite index for entity type and dimension
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_type_dimension 
        ON embeddings(entity_type, dimension);
        
        -- Index for updated_at for efficient cleanup
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_updated_at 
        ON embeddings(updated_at);
    END IF;
END $$;

-- ============================================
-- ENHANCED TRAINING METRICS TABLE INDEXES
-- ============================================
-- Add additional indexes to existing training_metrics table if it exists
DO $$ 
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'training_metrics') THEN
        -- Composite index for model name and version
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_training_metrics_model_version 
        ON training_metrics(model_name, version);
        
        -- Index for training date
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_training_metrics_date 
        ON training_metrics(training_date);
        
        -- GIN index for metrics searching
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_training_metrics_gin 
        ON training_metrics USING gin (metrics);
    END IF;
END $$;

-- ============================================
-- CACHE CLEANUP FUNCTION
-- ============================================
-- Function to clean up expired cache entries
CREATE OR REPLACE FUNCTION cleanup_expired_predictions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM prediction_cache 
    WHERE expires_at <= CURRENT_TIMESTAMP;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Log the cleanup operation
    RAISE NOTICE 'Cleaned up % expired prediction cache entries', deleted_count;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- SCHEDULED CACHE CLEANUP
-- ============================================
-- Create a scheduled job to clean up expired cache entries
-- This requires pg_cron extension (optional)
-- SELECT cron.schedule('cleanup-expired-predictions', '0 */6 * * *', 'SELECT cleanup_expired_predictions();');

-- ============================================
-- PERFORMANCE MONITORING VIEWS
-- ============================================
-- View for cache performance monitoring
CREATE OR REPLACE VIEW cache_performance_stats AS
SELECT 
    COUNT(*) as total_entries,
    COUNT(*) FILTER (WHERE expires_at > CURRENT_TIMESTAMP) as active_entries,
    COUNT(*) FILTER (WHERE expires_at <= CURRENT_TIMESTAMP) as expired_entries,
    AVG(access_count) as avg_access_count,
    MAX(access_count) as max_access_count,
    SUM(octet_length(predictions)) as total_size_bytes,
    AVG(octet_length(predictions)) as avg_size_bytes,
    MAX(octet_length(predictions)) as max_size_bytes,
    DATE_TRUNC('hour', cached_at) as cache_hour
FROM prediction_cache
GROUP BY DATE_TRUNC('hour', cached_at)
ORDER BY cache_hour DESC;

-- View for model storage statistics
CREATE OR REPLACE VIEW model_storage_stats AS
SELECT 
    model_name,
    COUNT(*) as total_versions,
    COUNT(*) FILTER (WHERE is_active = true) as active_versions,
    SUM(octet_length(model_data)) as total_size_bytes,
    AVG(octet_length(model_data)) as avg_size_bytes,
    MAX(created_at) as latest_version_date,
    MIN(created_at) as first_version_date
FROM ml_models
GROUP BY model_name
ORDER BY total_size_bytes DESC;

-- ============================================
-- GRANTS AND PERMISSIONS
-- ============================================
-- Grant appropriate permissions (adjust role names as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON prediction_cache TO ml_service_role;
-- GRANT SELECT ON cache_performance_stats TO monitoring_role;
-- GRANT SELECT ON model_storage_stats TO monitoring_role;

-- ============================================
-- COMMENTS
-- ============================================
COMMENT ON TABLE prediction_cache IS 'Cache for ML model predictions with TTL support';
COMMENT ON COLUMN prediction_cache.cache_key IS 'Unique identifier for cached predictions';
COMMENT ON COLUMN prediction_cache.predictions IS 'Serialized prediction data';
COMMENT ON COLUMN prediction_cache.metadata IS 'Additional metadata about the cached predictions';
COMMENT ON COLUMN prediction_cache.expires_at IS 'When the cache entry expires';
COMMENT ON COLUMN prediction_cache.access_count IS 'Number of times this cache entry has been accessed';

COMMENT ON FUNCTION cleanup_expired_predictions() IS 'Removes expired entries from prediction cache';
COMMENT ON VIEW cache_performance_stats IS 'Performance statistics for prediction cache';
COMMENT ON VIEW model_storage_stats IS 'Storage statistics for ML models';