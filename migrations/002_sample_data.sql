-- Migration: 002_sample_data.sql
-- Description: Insert sample data for testing and development
-- Created: 2025-07-12
-- Author: Production Database Setup

-- ============================================
-- SAMPLE USERS
-- ============================================
INSERT INTO users (id, email, min_price, max_price, min_bedrooms, max_bedrooms, min_bathrooms, max_bathrooms, preferred_locations, required_amenities, property_types) VALUES
(uuid_generate_v4(), 'john.doe@email.com', 1000.00, 2500.00, 1, 3, 1.0, 2.0, ARRAY['Downtown', 'Midtown'], ARRAY['parking', 'gym'], ARRAY['apartment', 'condo']),
(uuid_generate_v4(), 'jane.smith@email.com', 1500.00, 3000.00, 2, 4, 1.5, 3.0, ARRAY['Suburbs', 'North End'], ARRAY['pool', 'balcony'], ARRAY['house', 'townhouse']),
(uuid_generate_v4(), 'alice.johnson@email.com', 800.00, 1800.00, 0, 2, 1.0, 1.5, ARRAY['University District'], ARRAY['wifi', 'laundry'], ARRAY['studio', 'apartment']),
(uuid_generate_v4(), 'bob.wilson@email.com', 2000.00, 4000.00, 2, 5, 2.0, 3.5, ARRAY['Uptown', 'Riverfront'], ARRAY['garage', 'security'], ARRAY['house', 'loft']),
(uuid_generate_v4(), 'carol.brown@email.com', 1200.00, 2200.00, 1, 2, 1.0, 2.0, ARRAY['City Center'], ARRAY['elevator', 'dishwasher'], ARRAY['apartment', 'condo']);

-- ============================================
-- SAMPLE PROPERTIES
-- ============================================
INSERT INTO properties (
    id, title, description, price, location, bedrooms, bathrooms, square_feet, 
    amenities, contact_info, images, scraped_at, property_type, latitude, longitude, 
    external_id, data_quality_score
) VALUES
-- Downtown Apartments
(uuid_generate_v4(), 'Modern Downtown Studio', 
 'Beautiful studio apartment in the heart of downtown with city views. Recently renovated with modern fixtures and appliances. Perfect for young professionals.',
 1400.00, 'Downtown', 0, 1.0, 650, 
 ARRAY['wifi', 'laundry', 'gym', 'elevator', 'security'],
 '{"phone": "555-0101", "email": "contact@downtown-apartments.com"}',
 ARRAY['https://example.com/img1.jpg', 'https://example.com/img2.jpg'],
 CURRENT_TIMESTAMP - INTERVAL '1 day', 'studio', 40.7589, -73.9851, 'EXT001', 0.95),

(uuid_generate_v4(), 'Luxury 2BR Downtown Apartment', 
 'Spacious 2-bedroom, 2-bathroom apartment with premium finishes. Floor-to-ceiling windows, in-unit washer/dryer, and stunning city views.',
 2800.00, 'Downtown', 2, 2.0, 1200, 
 ARRAY['wifi', 'laundry', 'gym', 'elevator', 'security', 'parking', 'balcony', 'dishwasher'],
 '{"phone": "555-0102", "email": "luxury@downtown-living.com"}',
 ARRAY['https://example.com/img3.jpg', 'https://example.com/img4.jpg', 'https://example.com/img5.jpg'],
 CURRENT_TIMESTAMP - INTERVAL '2 hours', 'apartment', 40.7614, -73.9776, 'EXT002', 0.98),

-- Midtown Properties
(uuid_generate_v4(), 'Charming Midtown 1BR', 
 'Cozy 1-bedroom apartment in vibrant Midtown. Close to restaurants, shopping, and public transportation. Pet-friendly building.',
 1800.00, 'Midtown', 1, 1.0, 800, 
 ARRAY['wifi', 'laundry', 'gym', 'elevator', 'pets-allowed'],
 '{"phone": "555-0201", "email": "info@midtown-residences.com"}',
 ARRAY['https://example.com/img6.jpg', 'https://example.com/img7.jpg'],
 CURRENT_TIMESTAMP - INTERVAL '5 hours', 'apartment', 40.7505, -73.9934, 'EXT003', 0.92),

-- Suburban Houses
(uuid_generate_v4(), 'Spacious Family Home in Suburbs', 
 'Beautiful 4-bedroom, 3-bathroom house with a large backyard. Perfect for families. Recently updated kitchen and bathrooms.',
 3200.00, 'Suburbs', 4, 3.0, 2400, 
 ARRAY['wifi', 'laundry', 'garage', 'yard', 'security', 'dishwasher', 'fireplace'],
 '{"phone": "555-0301", "email": "suburban@homes-realty.com"}',
 ARRAY['https://example.com/img8.jpg', 'https://example.com/img9.jpg', 'https://example.com/img10.jpg'],
 CURRENT_TIMESTAMP - INTERVAL '1 day', 'house', 40.8176, -73.9782, 'EXT004', 0.96),

(uuid_generate_v4(), 'Cozy 2BR Townhouse', 
 'Newly renovated townhouse with modern amenities. Private entrance, small patio, and one car garage. Great for couples or small families.',
 2400.00, 'North End', 2, 2.5, 1400, 
 ARRAY['wifi', 'laundry', 'garage', 'patio', 'security', 'dishwasher'],
 '{"phone": "555-0302", "email": "townhouse@north-living.com"}',
 ARRAY['https://example.com/img11.jpg', 'https://example.com/img12.jpg'],
 CURRENT_TIMESTAMP - INTERVAL '3 hours', 'townhouse', 40.8448, -73.9242, 'EXT005', 0.94),

-- University District
(uuid_generate_v4(), 'Student-Friendly 1BR Near Campus', 
 'Affordable 1-bedroom apartment close to university campus. Utilities included. Perfect for graduate students or young professionals.',
 1200.00, 'University District', 1, 1.0, 600, 
 ARRAY['wifi', 'laundry', 'utilities-included', 'study-room'],
 '{"phone": "555-0401", "email": "student@campus-housing.com"}',
 ARRAY['https://example.com/img13.jpg'],
 CURRENT_TIMESTAMP - INTERVAL '6 hours', 'apartment', 40.8075, -73.9626, 'EXT006', 0.88),

-- Uptown Luxury
(uuid_generate_v4(), 'Luxury 3BR Penthouse', 
 'Stunning penthouse with panoramic city views. 3 bedrooms, 3 bathrooms, gourmet kitchen, and private terrace. Doorman building.',
 4500.00, 'Uptown', 3, 3.0, 2000, 
 ARRAY['wifi', 'laundry', 'gym', 'elevator', 'security', 'parking', 'balcony', 'dishwasher', 'doorman', 'terrace'],
 '{"phone": "555-0501", "email": "luxury@uptown-penthouses.com"}',
 ARRAY['https://example.com/img14.jpg', 'https://example.com/img15.jpg', 'https://example.com/img16.jpg', 'https://example.com/img17.jpg'],
 CURRENT_TIMESTAMP - INTERVAL '4 hours', 'apartment', 40.7831, -73.9712, 'EXT007', 0.99),

-- Riverfront
(uuid_generate_v4(), 'Waterfront Loft with River Views', 
 'Industrial-style loft with exposed brick and stunning river views. Open floor plan, high ceilings, and modern appliances.',
 3500.00, 'Riverfront', 2, 2.0, 1800, 
 ARRAY['wifi', 'laundry', 'gym', 'elevator', 'security', 'parking', 'river-view', 'dishwasher', 'exposed-brick'],
 '{"phone": "555-0601", "email": "loft@riverfront-living.com"}',
 ARRAY['https://example.com/img18.jpg', 'https://example.com/img19.jpg', 'https://example.com/img20.jpg'],
 CURRENT_TIMESTAMP - INTERVAL '8 hours', 'loft', 40.7282, -74.0776, 'EXT008', 0.97),

-- City Center
(uuid_generate_v4(), 'Modern 1BR in City Center', 
 'Contemporary 1-bedroom apartment with sleek design. Walking distance to major attractions and business district.',
 2000.00, 'City Center', 1, 1.0, 900, 
 ARRAY['wifi', 'laundry', 'gym', 'elevator', 'security', 'dishwasher', 'concierge'],
 '{"phone": "555-0701", "email": "modern@citycenter-apts.com"}',
 ARRAY['https://example.com/img21.jpg', 'https://example.com/img22.jpg'],
 CURRENT_TIMESTAMP - INTERVAL '2 days', 'apartment', 40.7505, -73.9934, 'EXT009', 0.93),

-- Additional variety properties
(uuid_generate_v4(), 'Affordable Studio Near Transit', 
 'Budget-friendly studio apartment with easy access to public transportation. Basic amenities included.',
 950.00, 'Transit District', 0, 1.0, 450, 
 ARRAY['wifi', 'laundry', 'transit-access'],
 '{"phone": "555-0801", "email": "budget@transit-living.com"}',
 ARRAY['https://example.com/img23.jpg'],
 CURRENT_TIMESTAMP - INTERVAL '12 hours', 'studio', 40.7282, -73.9942, 'EXT010', 0.85);

-- ============================================
-- SAMPLE USER INTERACTIONS
-- ============================================
-- Get some user and property IDs for interactions
DO $$
DECLARE
    user_ids UUID[];
    property_ids UUID[];
    user_id UUID;
    property_id UUID;
    i INTEGER;
BEGIN
    -- Get user IDs
    SELECT ARRAY(SELECT id FROM users LIMIT 5) INTO user_ids;
    -- Get property IDs  
    SELECT ARRAY(SELECT id FROM properties LIMIT 10) INTO property_ids;
    
    -- Generate sample interactions
    FOR i IN 1..50 LOOP
        user_id := user_ids[1 + (i % array_length(user_ids, 1))];
        property_id := property_ids[1 + (i % array_length(property_ids, 1))];
        
        INSERT INTO user_interactions (
            user_id, property_id, interaction_type, timestamp, duration_seconds, 
            session_id, interaction_strength
        ) VALUES (
            user_id,
            property_id,
            (ARRAY['view', 'like', 'inquiry', 'save', 'contact'])[1 + (i % 5)],
            CURRENT_TIMESTAMP - (INTERVAL '1 minute' * (i * 10)),
            30 + (i % 300),
            'session_' || (i % 10),
            0.5 + (i % 5) * 0.1
        );
    END LOOP;
END $$;

-- ============================================
-- SAMPLE ML MODELS
-- ============================================
INSERT INTO ml_models (id, model_name, version, metadata, training_accuracy, validation_accuracy, training_time_seconds, model_size_bytes) VALUES
(uuid_generate_v4(), 'collaborative_filter', 'v1.0', 
 '{"algorithm": "matrix_factorization", "factors": 50, "regularization": 0.01}',
 0.8542, 0.8234, 3600, 1048576),

(uuid_generate_v4(), 'content_recommender', 'v1.0', 
 '{"algorithm": "cosine_similarity", "features": ["price", "bedrooms", "bathrooms", "amenities"]}',
 0.7891, 0.7654, 1800, 524288),

(uuid_generate_v4(), 'hybrid_recommender', 'v1.0', 
 '{"collaborative_weight": 0.7, "content_weight": 0.3, "min_interactions": 5}',
 0.8934, 0.8756, 5400, 2097152),

(uuid_generate_v4(), 'search_ranker', 'v1.0', 
 '{"algorithm": "learning_to_rank", "features": ["relevance", "popularity", "freshness"]}',
 0.8123, 0.7987, 2700, 786432);

-- ============================================
-- SAMPLE TRAINING METRICS
-- ============================================
INSERT INTO training_metrics (
    model_name, version, metrics, job_id, training_duration_seconds, 
    dataset_size, hyperparameters, cpu_usage_percent, memory_usage_mb
) VALUES
('collaborative_filter', 'v1.0', 
 '{"rmse": 0.8542, "mae": 0.6734, "precision_at_10": 0.234, "recall_at_10": 0.145}',
 'job_001', 3600, 10000, 
 '{"factors": 50, "learning_rate": 0.01, "regularization": 0.01, "epochs": 100}',
 85.5, 2048),

('content_recommender', 'v1.0',
 '{"accuracy": 0.7891, "f1_score": 0.7432, "precision": 0.8123, "recall": 0.6834}',
 'job_002', 1800, 8500,
 '{"similarity_threshold": 0.7, "feature_weights": [0.3, 0.2, 0.2, 0.3]}',
 72.3, 1536),

('hybrid_recommender', 'v1.0',
 '{"ndcg_at_10": 0.8934, "map_at_10": 0.7654, "precision_at_5": 0.345, "recall_at_20": 0.567}',
 'job_003', 5400, 12000,
 '{"collaborative_weight": 0.7, "content_weight": 0.3, "alpha": 0.5}',
 91.2, 4096),

('search_ranker', 'v1.0',
 '{"ndcg": 0.8123, "map": 0.7456, "mrr": 0.8234, "auc": 0.8567}',
 'job_004', 2700, 15000,
 '{"learning_rate": 0.1, "n_estimators": 100, "max_depth": 6}',
 78.9, 2560);

-- ============================================
-- SAMPLE SEARCH QUERIES
-- ============================================
DO $$
DECLARE
    user_ids UUID[];
    user_id UUID;
    i INTEGER;
BEGIN
    SELECT ARRAY(SELECT id FROM users LIMIT 5) INTO user_ids;
    
    FOR i IN 1..20 LOOP
        user_id := CASE WHEN i % 3 = 0 THEN NULL ELSE user_ids[1 + (i % array_length(user_ids, 1))] END;
        
        INSERT INTO search_queries (
            user_id, query_text, filters, results_count, execution_time_ms, 
            session_id, page_number, page_size, sort_by
        ) VALUES (
            user_id,
            (ARRAY['downtown apartment', '2 bedroom house', 'luxury condo', 'studio near campus', 'pet friendly'])[1 + (i % 5)],
            ('{"min_price": ' || (1000 + (i % 5) * 500) || ', "max_price": ' || (2000 + (i % 5) * 1000) || '}')::jsonb,
            5 + (i % 15),
            50 + (i % 200),
            'session_' || (i % 8),
            1,
            20,
            (ARRAY['relevance', 'price_asc', 'price_desc', 'date_new'])[1 + (i % 4)]
        );
    END LOOP;
END $$;

-- ============================================
-- UPDATE STATISTICS
-- ============================================
-- Analyze tables for query optimization
ANALYZE users;
ANALYZE properties;
ANALYZE user_interactions;
ANALYZE ml_models;
ANALYZE embeddings;
ANALYZE training_metrics;
ANALYZE search_queries;
ANALYZE audit_log;

-- ============================================
-- VERIFY DATA INTEGRITY
-- ============================================
-- Check that all foreign key constraints are satisfied
DO $$
DECLARE
    constraint_violations INTEGER;
BEGIN
    SELECT COUNT(*) INTO constraint_violations
    FROM user_interactions ui
    LEFT JOIN users u ON ui.user_id = u.id
    WHERE u.id IS NULL;
    
    IF constraint_violations > 0 THEN
        RAISE EXCEPTION 'Found % orphaned user interactions', constraint_violations;
    END IF;
    
    RAISE NOTICE 'Data integrity verification completed successfully';
END $$;

-- ============================================
-- PERFORMANCE BASELINE
-- ============================================
-- Log current table sizes for monitoring
INSERT INTO audit_log (table_name, operation, row_id, new_values) VALUES
('migration_stats', 'INSERT', uuid_generate_v4(), 
 jsonb_build_object(
     'migration', '002_sample_data',
     'users_count', (SELECT COUNT(*) FROM users),
     'properties_count', (SELECT COUNT(*) FROM properties),
     'interactions_count', (SELECT COUNT(*) FROM user_interactions),
     'models_count', (SELECT COUNT(*) FROM ml_models),
     'completed_at', CURRENT_TIMESTAMP
 ));

COMMIT;