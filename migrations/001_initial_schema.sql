-- Migration: 001_initial_schema.sql
-- Description: Create initial database schema for rental ML system
-- Created: 2025-07-12
-- Author: Production Database Setup

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create enum types for better data integrity
CREATE TYPE interaction_type AS ENUM ('view', 'like', 'inquiry', 'save', 'contact', 'favorite');
CREATE TYPE property_type AS ENUM ('apartment', 'house', 'condo', 'townhouse', 'studio', 'loft');
CREATE TYPE user_status AS ENUM ('active', 'inactive', 'suspended', 'deleted');
CREATE TYPE property_status AS ENUM ('active', 'inactive', 'rented', 'pending', 'deleted');

-- ============================================
-- USERS TABLE
-- ============================================
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status user_status DEFAULT 'active',
    
    -- User preferences (denormalized for performance)
    min_price DECIMAL(10,2),
    max_price DECIMAL(10,2),
    min_bedrooms INTEGER,
    max_bedrooms INTEGER,
    min_bathrooms DECIMAL(3,1),
    max_bathrooms DECIMAL(3,1),
    preferred_locations TEXT[],
    required_amenities TEXT[],
    property_types property_type[],
    
    -- Additional user metadata
    last_login TIMESTAMP WITH TIME ZONE,
    login_count INTEGER DEFAULT 0,
    preference_updated_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT users_email_check CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    CONSTRAINT users_price_check CHECK (min_price IS NULL OR max_price IS NULL OR min_price <= max_price),
    CONSTRAINT users_bedrooms_check CHECK (min_bedrooms IS NULL OR max_bedrooms IS NULL OR min_bedrooms <= max_bedrooms),
    CONSTRAINT users_bathrooms_check CHECK (min_bathrooms IS NULL OR max_bathrooms IS NULL OR min_bathrooms <= max_bathrooms)
);

-- ============================================
-- PROPERTIES TABLE
-- ============================================
CREATE TABLE properties (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    location VARCHAR(500) NOT NULL,
    bedrooms INTEGER NOT NULL,
    bathrooms DECIMAL(3,1) NOT NULL,
    square_feet INTEGER,
    amenities TEXT[] DEFAULT '{}',
    contact_info JSONB DEFAULT '{}',
    images TEXT[] DEFAULT '{}',
    scraped_at TIMESTAMP WITH TIME ZONE NOT NULL,
    status property_status DEFAULT 'active',
    property_type property_type DEFAULT 'apartment',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Computed/derived fields for optimization
    full_text_search TEXT,
    price_per_sqft DECIMAL(8,2),
    amenity_count INTEGER GENERATED ALWAYS AS (array_length(amenities, 1)) STORED,
    image_count INTEGER GENERATED ALWAYS AS (array_length(images, 1)) STORED,
    
    -- Location data (for future geospatial queries)
    latitude DECIMAL(10,8),
    longitude DECIMAL(11,8),
    
    -- SEO and search optimization
    slug VARCHAR(255),
    external_id VARCHAR(255), -- ID from external source (apartments.com, etc.)
    external_url TEXT,
    
    -- Data quality tracking
    data_quality_score DECIMAL(3,2) DEFAULT 1.0,
    last_verified TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT properties_price_check CHECK (price > 0),
    CONSTRAINT properties_bedrooms_check CHECK (bedrooms >= 0),
    CONSTRAINT properties_bathrooms_check CHECK (bathrooms >= 0),
    CONSTRAINT properties_square_feet_check CHECK (square_feet IS NULL OR square_feet > 0),
    CONSTRAINT properties_price_per_sqft_check CHECK (price_per_sqft IS NULL OR price_per_sqft > 0),
    CONSTRAINT properties_data_quality_check CHECK (data_quality_score >= 0 AND data_quality_score <= 1),
    CONSTRAINT properties_coordinates_check CHECK (
        (latitude IS NULL AND longitude IS NULL) OR 
        (latitude IS NOT NULL AND longitude IS NOT NULL AND 
         latitude >= -90 AND latitude <= 90 AND 
         longitude >= -180 AND longitude <= 180)
    )
);

-- ============================================
-- USER INTERACTIONS TABLE
-- ============================================
CREATE TABLE user_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    property_id UUID NOT NULL,
    interaction_type interaction_type NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    duration_seconds INTEGER,
    
    -- Additional interaction metadata
    session_id VARCHAR(255),
    user_agent TEXT,
    ip_address INET,
    referrer TEXT,
    
    -- ML features
    interaction_strength DECIMAL(3,2) DEFAULT 1.0, -- Weighted importance of interaction
    
    -- Constraints
    CONSTRAINT user_interactions_duration_check CHECK (duration_seconds IS NULL OR duration_seconds >= 0),
    CONSTRAINT user_interactions_strength_check CHECK (interaction_strength >= 0 AND interaction_strength <= 1)
);

-- ============================================
-- ML MODELS TABLE
-- ============================================
CREATE TABLE ml_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_data BYTEA, -- For small models; use file storage for large ones
    model_file_path TEXT, -- Path to model file for large models
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Model performance metrics
    training_accuracy DECIMAL(5,4),
    validation_accuracy DECIMAL(5,4),
    training_time_seconds INTEGER,
    model_size_bytes BIGINT,
    
    -- Model versioning
    parent_model_id UUID REFERENCES ml_models(id),
    
    -- Constraints
    CONSTRAINT ml_models_unique_active_version UNIQUE (model_name, version),
    CONSTRAINT ml_models_accuracy_check CHECK (
        training_accuracy IS NULL OR (training_accuracy >= 0 AND training_accuracy <= 1)
    ),
    CONSTRAINT ml_models_validation_accuracy_check CHECK (
        validation_accuracy IS NULL OR (validation_accuracy >= 0 AND validation_accuracy <= 1)
    ),
    CONSTRAINT ml_models_training_time_check CHECK (training_time_seconds IS NULL OR training_time_seconds > 0),
    CONSTRAINT ml_models_size_check CHECK (model_size_bytes IS NULL OR model_size_bytes > 0)
);

-- ============================================
-- EMBEDDINGS TABLE
-- ============================================
CREATE TABLE embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type VARCHAR(50) NOT NULL, -- 'user', 'property', etc.
    entity_id VARCHAR(255) NOT NULL,
    embeddings BYTEA NOT NULL, -- Serialized numpy array or similar
    dimension INTEGER NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Embedding metadata
    norm DECIMAL(10,6), -- L2 norm of the embedding
    
    -- Constraints
    CONSTRAINT embeddings_unique_entity_model UNIQUE (entity_type, entity_id, model_version),
    CONSTRAINT embeddings_dimension_check CHECK (dimension > 0),
    CONSTRAINT embeddings_norm_check CHECK (norm IS NULL OR norm > 0)
);

-- ============================================
-- TRAINING METRICS TABLE
-- ============================================
CREATE TABLE training_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    metrics JSONB NOT NULL,
    training_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Training job metadata
    job_id VARCHAR(255),
    training_duration_seconds INTEGER,
    dataset_size INTEGER,
    hyperparameters JSONB,
    
    -- Performance tracking
    cpu_usage_percent DECIMAL(5,2),
    memory_usage_mb INTEGER,
    gpu_usage_percent DECIMAL(5,2),
    
    -- Constraints
    CONSTRAINT training_metrics_duration_check CHECK (training_duration_seconds IS NULL OR training_duration_seconds > 0),
    CONSTRAINT training_metrics_dataset_size_check CHECK (dataset_size IS NULL OR dataset_size > 0),
    CONSTRAINT training_metrics_cpu_check CHECK (cpu_usage_percent IS NULL OR (cpu_usage_percent >= 0 AND cpu_usage_percent <= 100)),
    CONSTRAINT training_metrics_memory_check CHECK (memory_usage_mb IS NULL OR memory_usage_mb > 0),
    CONSTRAINT training_metrics_gpu_check CHECK (gpu_usage_percent IS NULL OR (gpu_usage_percent >= 0 AND gpu_usage_percent <= 100))
);

-- ============================================
-- SEARCH QUERIES TABLE (for analytics)
-- ============================================
CREATE TABLE search_queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    query_text TEXT NOT NULL,
    filters JSONB DEFAULT '{}',
    results_count INTEGER,
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Query metadata
    session_id VARCHAR(255),
    page_number INTEGER DEFAULT 1,
    page_size INTEGER DEFAULT 50,
    sort_by VARCHAR(100),
    
    -- Constraints
    CONSTRAINT search_queries_results_count_check CHECK (results_count IS NULL OR results_count >= 0),
    CONSTRAINT search_queries_execution_time_check CHECK (execution_time_ms IS NULL OR execution_time_ms >= 0),
    CONSTRAINT search_queries_page_check CHECK (page_number > 0 AND page_size > 0)
);

-- ============================================
-- AUDIT LOG TABLE
-- ============================================
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name VARCHAR(100) NOT NULL,
    operation VARCHAR(10) NOT NULL, -- INSERT, UPDATE, DELETE
    row_id UUID NOT NULL,
    old_values JSONB,
    new_values JSONB,
    changed_by UUID, -- Can reference users(id) if needed
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Additional context
    session_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    
    -- Constraints
    CONSTRAINT audit_log_operation_check CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE'))
);

-- ============================================
-- CREATE INDEXES FOR PERFORMANCE
-- ============================================

-- Users table indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_created_at ON users(created_at);
CREATE INDEX idx_users_last_login ON users(last_login);
CREATE INDEX idx_users_price_range ON users(min_price, max_price);
CREATE INDEX idx_users_bedroom_range ON users(min_bedrooms, max_bedrooms);
CREATE INDEX idx_users_locations_gin ON users USING GIN(preferred_locations);
CREATE INDEX idx_users_amenities_gin ON users USING GIN(required_amenities);
CREATE INDEX idx_users_property_types_gin ON users USING GIN(property_types);

-- Properties table indexes
CREATE INDEX idx_properties_status ON properties(status);
CREATE INDEX idx_properties_location ON properties(location);
CREATE INDEX idx_properties_location_trgm ON properties USING GIN(location gin_trgm_ops);
CREATE INDEX idx_properties_price ON properties(price);
CREATE INDEX idx_properties_bedrooms ON properties(bedrooms);
CREATE INDEX idx_properties_bathrooms ON properties(bathrooms);
CREATE INDEX idx_properties_square_feet ON properties(square_feet);
CREATE INDEX idx_properties_type ON properties(property_type);
CREATE INDEX idx_properties_created_at ON properties(created_at);
CREATE INDEX idx_properties_scraped_at ON properties(scraped_at);
CREATE INDEX idx_properties_price_per_sqft ON properties(price_per_sqft);
CREATE INDEX idx_properties_amenities_gin ON properties USING GIN(amenities);
CREATE INDEX idx_properties_external_id ON properties(external_id);
CREATE INDEX idx_properties_slug ON properties(slug);
CREATE INDEX idx_properties_coordinates ON properties(latitude, longitude);
CREATE INDEX idx_properties_data_quality ON properties(data_quality_score);

-- Full-text search index
CREATE INDEX idx_properties_fulltext ON properties USING GIN(to_tsvector('english', full_text_search));

-- Composite indexes for common queries
CREATE INDEX idx_properties_status_price ON properties(status, price);
CREATE INDEX idx_properties_status_bedrooms ON properties(status, bedrooms);
CREATE INDEX idx_properties_status_location ON properties(status, location);
CREATE INDEX idx_properties_price_bedrooms_bathrooms ON properties(price, bedrooms, bathrooms);

-- User interactions indexes
CREATE INDEX idx_user_interactions_user_id ON user_interactions(user_id);
CREATE INDEX idx_user_interactions_property_id ON user_interactions(property_id);
CREATE INDEX idx_user_interactions_type ON user_interactions(interaction_type);
CREATE INDEX idx_user_interactions_timestamp ON user_interactions(timestamp);
CREATE INDEX idx_user_interactions_session ON user_interactions(session_id);
CREATE INDEX idx_user_interactions_user_timestamp ON user_interactions(user_id, timestamp);
CREATE INDEX idx_user_interactions_property_timestamp ON user_interactions(property_id, timestamp);
CREATE INDEX idx_user_interactions_strength ON user_interactions(interaction_strength);

-- ML models indexes
CREATE INDEX idx_ml_models_name_version ON ml_models(model_name, version);
CREATE INDEX idx_ml_models_active ON ml_models(is_active);
CREATE INDEX idx_ml_models_created_at ON ml_models(created_at);
CREATE INDEX idx_ml_models_parent ON ml_models(parent_model_id);

-- Embeddings indexes
CREATE INDEX idx_embeddings_entity ON embeddings(entity_type, entity_id);
CREATE INDEX idx_embeddings_model_version ON embeddings(model_version);
CREATE INDEX idx_embeddings_created_at ON embeddings(created_at);
CREATE INDEX idx_embeddings_dimension ON embeddings(dimension);

-- Training metrics indexes
CREATE INDEX idx_training_metrics_model ON training_metrics(model_name, version);
CREATE INDEX idx_training_metrics_date ON training_metrics(training_date);
CREATE INDEX idx_training_metrics_job ON training_metrics(job_id);

-- Search queries indexes
CREATE INDEX idx_search_queries_user_id ON search_queries(user_id);
CREATE INDEX idx_search_queries_created_at ON search_queries(created_at);
CREATE INDEX idx_search_queries_session ON search_queries(session_id);
CREATE INDEX idx_search_queries_text_trgm ON search_queries USING GIN(query_text gin_trgm_ops);

-- Audit log indexes
CREATE INDEX idx_audit_log_table_name ON audit_log(table_name);
CREATE INDEX idx_audit_log_operation ON audit_log(operation);
CREATE INDEX idx_audit_log_row_id ON audit_log(row_id);
CREATE INDEX idx_audit_log_changed_at ON audit_log(changed_at);
CREATE INDEX idx_audit_log_changed_by ON audit_log(changed_by);

-- ============================================
-- UPDATE TRIGGERS
-- ============================================

-- Function to update the updated_at column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add update triggers to tables with updated_at columns
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_properties_updated_at BEFORE UPDATE ON properties
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_embeddings_updated_at BEFORE UPDATE ON embeddings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to automatically update computed fields in properties
CREATE OR REPLACE FUNCTION update_property_computed_fields()
RETURNS TRIGGER AS $$
BEGIN
    -- Update full text search field
    NEW.full_text_search = NEW.title || ' ' || NEW.description || ' ' || NEW.location || ' ' || 
                           COALESCE(array_to_string(NEW.amenities, ' '), '');
    
    -- Update price per square foot
    IF NEW.square_feet IS NOT NULL AND NEW.square_feet > 0 THEN
        NEW.price_per_sqft = NEW.price / NEW.square_feet;
    ELSE
        NEW.price_per_sqft = NULL;
    END IF;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add trigger for property computed fields
CREATE TRIGGER update_property_computed_fields_trigger 
    BEFORE INSERT OR UPDATE ON properties
    FOR EACH ROW EXECUTE FUNCTION update_property_computed_fields();

-- ============================================
-- PERFORMANCE MONITORING VIEWS
-- ============================================

-- View for property statistics
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

-- View for user interaction summary
CREATE VIEW user_interaction_summary AS
SELECT 
    u.id as user_id,
    u.email,
    COUNT(ui.id) as total_interactions,
    COUNT(ui.id) FILTER (WHERE ui.interaction_type = 'view') as views,
    COUNT(ui.id) FILTER (WHERE ui.interaction_type = 'like') as likes,
    COUNT(ui.id) FILTER (WHERE ui.interaction_type = 'inquiry') as inquiries,
    COUNT(ui.id) FILTER (WHERE ui.interaction_type = 'save') as saves,
    MAX(ui.timestamp) as last_interaction,
    AVG(ui.duration_seconds) as avg_duration
FROM users u
LEFT JOIN user_interactions ui ON u.id = ui.user_id
WHERE u.status = 'active'
GROUP BY u.id, u.email;

-- View for property popularity
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

-- ============================================
-- SECURITY AND DATA INTEGRITY
-- ============================================

-- Row Level Security (RLS) setup for multi-tenant scenarios (if needed in future)
-- ALTER TABLE users ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE user_interactions ENABLE ROW LEVEL SECURITY;

-- Grant appropriate permissions to application user
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO rental_ml_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO rental_ml_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO rental_ml_user;

-- Create read-only user for analytics/reporting
-- CREATE USER rental_ml_readonly WITH PASSWORD 'ReadOnlyPassword123!';
-- GRANT CONNECT ON DATABASE rental_ml TO rental_ml_readonly;
-- GRANT USAGE ON SCHEMA public TO rental_ml_readonly;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO rental_ml_readonly;

COMMENT ON DATABASE rental_ml IS 'Rental ML System Production Database';
COMMENT ON TABLE users IS 'User accounts and preferences';
COMMENT ON TABLE properties IS 'Rental property listings';
COMMENT ON TABLE user_interactions IS 'User interaction events for ML training';
COMMENT ON TABLE ml_models IS 'Machine learning model storage and metadata';
COMMENT ON TABLE embeddings IS 'Vector embeddings for recommendations';
COMMENT ON TABLE training_metrics IS 'ML model training performance metrics';
COMMENT ON TABLE search_queries IS 'Search query analytics';
COMMENT ON TABLE audit_log IS 'System audit trail';