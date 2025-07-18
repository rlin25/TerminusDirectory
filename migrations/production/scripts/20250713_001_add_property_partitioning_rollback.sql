-- Rollback Migration: Remove Property Partitioning
-- Version: 20250713_001_rollback
-- Description: Rollback property partitioning and restore original table structure

-- Start transaction for atomic rollback
BEGIN;

-- Drop scheduled cron jobs if they exist
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') THEN
        -- Remove scheduled jobs
        PERFORM cron.unschedule('create-property-partitions');
        PERFORM cron.unschedule('cleanup-old-property-partitions');
        RAISE NOTICE 'Removed scheduled partition management jobs';
    END IF;
END $$;

-- Drop partition management functions
DROP FUNCTION IF EXISTS create_monthly_property_partition(INTEGER, INTEGER);
DROP FUNCTION IF EXISTS drop_old_property_partitions(INTEGER);
DROP FUNCTION IF EXISTS get_property_partition_for_date(DATE);

-- Verify backup table exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'properties_backup_20250713') THEN
        RAISE EXCEPTION 'Backup table properties_backup_20250713 not found. Cannot proceed with rollback.';
    END IF;
END $$;

-- Create temporary table to hold all partitioned data
CREATE TEMP TABLE properties_rollback_temp AS
SELECT * FROM properties;

-- Get count for verification
DO $$
DECLARE
    temp_count INTEGER;
    backup_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO temp_count FROM properties_rollback_temp;
    SELECT COUNT(*) INTO backup_count FROM properties_backup_20250713;
    
    RAISE NOTICE 'Temporary table has % rows, backup table has % rows', temp_count, backup_count;
END $$;

-- Drop partitioned table and all partitions
DROP TABLE IF EXISTS properties CASCADE;

-- Restore original table from backup
ALTER TABLE properties_backup_20250713 RENAME TO properties;

-- Merge any new data that was added after partitioning
INSERT INTO properties 
SELECT * FROM properties_rollback_temp 
WHERE id NOT IN (SELECT id FROM properties);

-- Update sequence ownership
ALTER SEQUENCE IF EXISTS properties_id_seq OWNED BY properties.id;

-- Recreate original indexes that might have been lost
CREATE INDEX IF NOT EXISTS idx_properties_status ON properties(status);
CREATE INDEX IF NOT EXISTS idx_properties_location ON properties(location);
CREATE INDEX IF NOT EXISTS idx_properties_location_trgm ON properties USING GIN(location gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_properties_price ON properties(price);
CREATE INDEX IF NOT EXISTS idx_properties_bedrooms ON properties(bedrooms);
CREATE INDEX IF NOT EXISTS idx_properties_bathrooms ON properties(bathrooms);
CREATE INDEX IF NOT EXISTS idx_properties_square_feet ON properties(square_feet);
CREATE INDEX IF NOT EXISTS idx_properties_type ON properties(property_type);
CREATE INDEX IF NOT EXISTS idx_properties_created_at ON properties(created_at);
CREATE INDEX IF NOT EXISTS idx_properties_scraped_at ON properties(scraped_at);
CREATE INDEX IF NOT EXISTS idx_properties_price_per_sqft ON properties(price_per_sqft);
CREATE INDEX IF NOT EXISTS idx_properties_amenities_gin ON properties USING GIN(amenities);
CREATE INDEX IF NOT EXISTS idx_properties_external_id ON properties(external_id);
CREATE INDEX IF NOT EXISTS idx_properties_slug ON properties(slug);
CREATE INDEX IF NOT EXISTS idx_properties_coordinates ON properties(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_properties_data_quality ON properties(data_quality_score);

-- Recreate full-text search index
CREATE INDEX IF NOT EXISTS idx_properties_fulltext ON properties USING GIN(to_tsvector('english', full_text_search));

-- Recreate composite indexes
CREATE INDEX IF NOT EXISTS idx_properties_status_price ON properties(status, price);
CREATE INDEX IF NOT EXISTS idx_properties_status_bedrooms ON properties(status, bedrooms);
CREATE INDEX IF NOT EXISTS idx_properties_status_location ON properties(status, location);
CREATE INDEX IF NOT EXISTS idx_properties_price_bedrooms_bathrooms ON properties(price, bedrooms, bathrooms);

-- Recreate views that depend on properties table
DROP VIEW IF EXISTS property_stats;
CREATE VIEW property_stats AS
SELECT 
    COUNT(*) as total_properties,
    COUNT(*) FILTER (WHERE status = 'active') as active_properties,
    AVG(price) as avg_price,
    MIN(price) as min_price,
    MAX(price) as max_price,
    AVG(bedrooms) as avg_bedrooms,
    AVG(bathrooms) as avg_bathrooms,
    AVG(square_feet) as avg_square_feet,
    COUNT(DISTINCT location) as unique_locations,
    AVG(amenity_count) as avg_amenity_count
FROM properties;

DROP VIEW IF EXISTS property_popularity;
CREATE VIEW property_popularity AS
SELECT 
    p.id as property_id,
    p.title,
    p.price,
    p.location,
    COUNT(ui.id) as total_interactions,
    COUNT(ui.id) FILTER (WHERE ui.interaction_type = 'view') as views,
    COUNT(ui.id) FILTER (WHERE ui.interaction_type = 'like') as likes,
    COUNT(ui.id) FILTER (WHERE ui.interaction_type = 'inquiry') as inquiries,
    AVG(ui.interaction_strength) as avg_interaction_strength,
    MAX(ui.timestamp) as last_interaction
FROM properties p
LEFT JOIN user_interactions ui ON p.id = ui.property_id
WHERE p.status = 'active'
GROUP BY p.id, p.title, p.price, p.location;

-- Verify data integrity after rollback
DO $$
DECLARE
    final_count INTEGER;
    temp_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO final_count FROM properties;
    SELECT COUNT(*) INTO temp_count FROM properties_rollback_temp;
    
    IF final_count < temp_count THEN
        RAISE EXCEPTION 'Data loss detected during rollback: final=%, expected=%', final_count, temp_count;
    END IF;
    
    RAISE NOTICE 'Rollback verification passed: % rows in restored table', final_count;
END $$;

-- Update table statistics
ANALYZE properties;

-- Clean up temporary table
DROP TABLE properties_rollback_temp;

-- Log rollback completion
INSERT INTO migration_execution_log (migration_version, execution_step, status, message)
VALUES ('20250713_001_rollback', 'rollback_complete', 'completed', 
        'Properties table partitioning successfully rolled back');

-- Update table comment
COMMENT ON TABLE properties IS 'Rental property listings (partitioning removed)';

COMMIT;

-- Final verification outside transaction
DO $$
DECLARE
    table_exists BOOLEAN;
    is_partitioned BOOLEAN;
BEGIN
    -- Check if table exists and is not partitioned
    SELECT EXISTS (
        SELECT 1 FROM pg_tables 
        WHERE tablename = 'properties' AND schemaname = 'public'
    ) INTO table_exists;
    
    SELECT EXISTS (
        SELECT 1 FROM pg_partitioned_table 
        WHERE partrelid = 'properties'::regclass
    ) INTO is_partitioned;
    
    IF NOT table_exists THEN
        RAISE EXCEPTION 'Properties table does not exist after rollback';
    END IF;
    
    IF is_partitioned THEN
        RAISE EXCEPTION 'Properties table is still partitioned after rollback';
    END IF;
    
    RAISE NOTICE 'Rollback verification complete: properties table restored to original state';
END $$;