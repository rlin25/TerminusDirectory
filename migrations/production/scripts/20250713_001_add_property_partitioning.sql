-- Migration: Add Property Partitioning
-- Version: 20250713_001
-- Name: Add Property Partitioning by Date
-- Description: Partition properties table by scraped_at date for better performance on large datasets
-- Author: Production Database Team
-- Created At: 2025-07-13T00:00:00Z
-- Safety Level: caution
-- Estimated Duration: 300
-- Requires Downtime: false
-- Dependencies: 001_initial_schema
-- Affects Tables: properties
-- Rollback Strategy: automated
-- Validation Queries: SELECT COUNT(*) > 0 FROM properties_2025_07; SELECT COUNT(*) > 0 FROM properties_2025_08
-- Pre Checks: SELECT COUNT(*) FROM properties; SELECT schemaname, tablename FROM pg_tables WHERE tablename = 'properties'
-- Post Checks: SELECT schemaname, tablename FROM pg_tables WHERE tablename LIKE 'properties_202%'

-- Create partitioned properties table
CREATE TABLE properties_partitioned (
    LIKE properties INCLUDING ALL
) PARTITION BY RANGE (scraped_at);

-- Create initial partitions for current and next few months
CREATE TABLE properties_2025_07 PARTITION OF properties_partitioned
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');

CREATE TABLE properties_2025_08 PARTITION OF properties_partitioned
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');

CREATE TABLE properties_2025_09 PARTITION OF properties_partitioned
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');

CREATE TABLE properties_2025_10 PARTITION OF properties_partitioned
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');

CREATE TABLE properties_2025_11 PARTITION OF properties_partitioned
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

CREATE TABLE properties_2025_12 PARTITION OF properties_partitioned
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- Create default partition for future dates
CREATE TABLE properties_default PARTITION OF properties_partitioned DEFAULT;

-- Copy existing data to partitioned table
INSERT INTO properties_partitioned SELECT * FROM properties;

-- Verify data integrity
DO $$
DECLARE
    original_count INTEGER;
    partitioned_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO original_count FROM properties;
    SELECT COUNT(*) INTO partitioned_count FROM properties_partitioned;
    
    IF original_count != partitioned_count THEN
        RAISE EXCEPTION 'Data count mismatch: original=%, partitioned=%', original_count, partitioned_count;
    END IF;
    
    RAISE NOTICE 'Data migration verified: % rows copied successfully', original_count;
END $$;

-- Drop old table and rename partitioned table
BEGIN;
    -- Rename original table as backup
    ALTER TABLE properties RENAME TO properties_backup_20250713;
    
    -- Rename partitioned table to original name
    ALTER TABLE properties_partitioned RENAME TO properties;
    
    -- Update sequence ownership
    ALTER SEQUENCE IF EXISTS properties_id_seq OWNED BY properties.id;
    
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
COMMIT;

-- Create indexes on partitions for better performance
CREATE INDEX CONCURRENTLY idx_properties_2025_07_status_price ON properties_2025_07(status, price);
CREATE INDEX CONCURRENTLY idx_properties_2025_07_location_gin ON properties_2025_07 USING GIN(location gin_trgm_ops);
CREATE INDEX CONCURRENTLY idx_properties_2025_07_bedrooms ON properties_2025_07(bedrooms);

CREATE INDEX CONCURRENTLY idx_properties_2025_08_status_price ON properties_2025_08(status, price);
CREATE INDEX CONCURRENTLY idx_properties_2025_08_location_gin ON properties_2025_08 USING GIN(location gin_trgm_ops);
CREATE INDEX CONCURRENTLY idx_properties_2025_08_bedrooms ON properties_2025_08(bedrooms);

CREATE INDEX CONCURRENTLY idx_properties_2025_09_status_price ON properties_2025_09(status, price);
CREATE INDEX CONCURRENTLY idx_properties_2025_09_location_gin ON properties_2025_09 USING GIN(location gin_trgm_ops);
CREATE INDEX CONCURRENTLY idx_properties_2025_09_bedrooms ON properties_2025_09(bedrooms);

-- Create function to automatically create monthly partitions
CREATE OR REPLACE FUNCTION create_monthly_property_partition(year INTEGER, month INTEGER)
RETURNS VOID AS $$
DECLARE
    table_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    table_name := 'properties_' || year || '_' || LPAD(month::TEXT, 2, '0');
    start_date := DATE(year || '-' || month || '-01');
    end_date := start_date + INTERVAL '1 month';
    
    EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF properties
                    FOR VALUES FROM (%L) TO (%L)',
                   table_name, start_date, end_date);
                   
    -- Create indexes on new partition
    EXECUTE format('CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_%s_status_price ON %I(status, price)', 
                   table_name, table_name);
    EXECUTE format('CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_%s_location_gin ON %I USING GIN(location gin_trgm_ops)', 
                   table_name, table_name);
    EXECUTE format('CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_%s_bedrooms ON %I(bedrooms)', 
                   table_name, table_name);
                   
    RAISE NOTICE 'Created partition % for date range % to %', table_name, start_date, end_date;
END;
$$ LANGUAGE plpgsql;

-- Create function to drop old partitions (for data retention)
CREATE OR REPLACE FUNCTION drop_old_property_partitions(retention_months INTEGER DEFAULT 24)
RETURNS VOID AS $$
DECLARE
    partition_record RECORD;
    cutoff_date DATE;
BEGIN
    cutoff_date := DATE_TRUNC('month', CURRENT_DATE - (retention_months || ' months')::INTERVAL);
    
    FOR partition_record IN
        SELECT schemaname, tablename 
        FROM pg_tables 
        WHERE tablename ~ '^properties_\d{4}_\d{2}$'
        AND schemaname = 'public'
    LOOP
        -- Extract date from partition name and compare
        DECLARE
            partition_date DATE;
            year_part INTEGER;
            month_part INTEGER;
        BEGIN
            year_part := (regexp_matches(partition_record.tablename, 'properties_(\d{4})_(\d{2})'))[1]::INTEGER;
            month_part := (regexp_matches(partition_record.tablename, 'properties_(\d{4})_(\d{2})'))[2]::INTEGER;
            partition_date := DATE(year_part || '-' || month_part || '-01');
            
            IF partition_date < cutoff_date THEN
                EXECUTE format('DROP TABLE IF EXISTS %I', partition_record.tablename);
                RAISE NOTICE 'Dropped old partition %', partition_record.tablename;
            END IF;
        END;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Schedule automatic partition creation using pg_cron (if available)
-- This will create partitions for the next 3 months
DO $$
BEGIN
    -- Check if pg_cron is available
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') THEN
        -- Schedule monthly partition creation
        PERFORM cron.schedule('create-property-partitions', '0 0 1 * *', 
            'SELECT create_monthly_property_partition(EXTRACT(year FROM CURRENT_DATE + INTERVAL ''2 months'')::INTEGER, 
                                                     EXTRACT(month FROM CURRENT_DATE + INTERVAL ''2 months'')::INTEGER)');
        
        -- Schedule quarterly cleanup of old partitions
        PERFORM cron.schedule('cleanup-old-property-partitions', '0 2 1 */3 *', 
            'SELECT drop_old_property_partitions(24)');
            
        RAISE NOTICE 'Scheduled automatic partition management';
    ELSE
        RAISE NOTICE 'pg_cron not available, manual partition management required';
    END IF;
END $$;

-- Create constraint exclusion helper function
CREATE OR REPLACE FUNCTION get_property_partition_for_date(target_date DATE)
RETURNS TEXT AS $$
DECLARE
    year_part INTEGER;
    month_part INTEGER;
    table_name TEXT;
BEGIN
    year_part := EXTRACT(year FROM target_date);
    month_part := EXTRACT(month FROM target_date);
    table_name := 'properties_' || year_part || '_' || LPAD(month_part::TEXT, 2, '0');
    
    -- Check if partition exists
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = table_name) THEN
        RETURN table_name;
    ELSE
        RETURN 'properties_default';
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Update table statistics
ANALYZE properties;

-- Log migration completion
INSERT INTO migration_execution_log (migration_version, execution_step, status, message)
VALUES ('20250713_001', 'partitioning_complete', 'completed', 
        'Properties table successfully partitioned by scraped_at date');

COMMENT ON TABLE properties IS 'Rental property listings (partitioned by scraped_at date)';
COMMENT ON FUNCTION create_monthly_property_partition IS 'Creates monthly partition for properties table';
COMMENT ON FUNCTION drop_old_property_partitions IS 'Drops old property partitions based on retention policy';
COMMENT ON FUNCTION get_property_partition_for_date IS 'Returns partition name for a given date';