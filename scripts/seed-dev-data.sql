-- =============================================================================
-- Development Database Seed Data
-- =============================================================================
-- This script populates the development database with test data

-- Wait for tables to be created by the application
-- This script assumes tables have been created by the application startup

-- =============================================================================
-- Test Users
-- =============================================================================

INSERT INTO users (id, email, min_price, max_price, min_bedrooms, max_bedrooms, min_bathrooms, max_bathrooms, preferred_locations, required_amenities, property_types)
VALUES 
    (gen_random_uuid(), 'john.doe@example.com', 1000, 2500, 1, 3, 1, 2, ARRAY['Downtown', 'Midtown'], ARRAY['parking', 'gym'], ARRAY['apartment']),
    (gen_random_uuid(), 'jane.smith@example.com', 1500, 3000, 2, 4, 1.5, 2.5, ARRAY['Suburbs', 'Uptown'], ARRAY['pool', 'laundry'], ARRAY['house', 'townhouse']),
    (gen_random_uuid(), 'mike.johnson@example.com', 800, 1800, 1, 2, 1, 1.5, ARRAY['University District'], ARRAY['internet', 'furnished'], ARRAY['apartment', 'studio']),
    (gen_random_uuid(), 'sarah.williams@example.com', 2000, 4000, 2, 5, 2, 3, ARRAY['Luxury District', 'Waterfront'], ARRAY['concierge', 'gym', 'pool'], ARRAY['luxury apartment', 'penthouse']),
    (gen_random_uuid(), 'dev.user@example.com', 500, 5000, 1, 6, 1, 4, ARRAY['Any'], ARRAY[], ARRAY['apartment', 'house', 'condo'])
ON CONFLICT (email) DO NOTHING;

-- =============================================================================
-- Test Properties
-- =============================================================================

INSERT INTO properties (id, title, description, price, location, bedrooms, bathrooms, square_feet, amenities, contact_info, images, property_type, full_text_search, price_per_sqft, scraped_at)
VALUES 
    (gen_random_uuid(), 
     'Modern Downtown Apartment', 
     'Beautiful modern apartment in the heart of downtown with stunning city views. Features include hardwood floors, stainless steel appliances, and in-unit laundry.',
     1800, 
     'Downtown', 
     2, 
     1.5, 
     900,
     ARRAY['parking', 'gym', 'rooftop', 'laundry'],
     '{"phone": "555-0101", "email": "downtown@example.com"}',
     ARRAY['https://example.com/image1.jpg', 'https://example.com/image2.jpg'],
     'apartment',
     'Modern Downtown Apartment Beautiful modern apartment in the heart of downtown with stunning city views. Features include hardwood floors, stainless steel appliances, and in-unit laundry.',
     2.00,
     NOW()),
    
    (gen_random_uuid(),
     'Cozy Studio Near University',
     'Perfect for students! This cozy studio apartment is just 5 minutes walk from the university campus. Fully furnished with modern amenities.',
     1200,
     'University District',
     1,
     1,
     450,
     ARRAY['furnished', 'internet', 'study area'],
     '{"phone": "555-0102", "email": "university@example.com"}',
     ARRAY['https://example.com/studio1.jpg'],
     'studio',
     'Cozy Studio Near University Perfect for students! This cozy studio apartment is just 5 minutes walk from the university campus. Fully furnished with modern amenities.',
     2.67,
     NOW()),
    
    (gen_random_uuid(),
     'Luxury Waterfront Condo',
     'Stunning 3-bedroom luxury condominium with panoramic water views. Features marble countertops, high-end appliances, and access to exclusive amenities.',
     3500,
     'Waterfront',
     3,
     2.5,
     1400,
     ARRAY['concierge', 'gym', 'pool', 'parking', 'balcony'],
     '{"phone": "555-0103", "email": "luxury@example.com"}',
     ARRAY['https://example.com/luxury1.jpg', 'https://example.com/luxury2.jpg', 'https://example.com/luxury3.jpg'],
     'luxury apartment',
     'Luxury Waterfront Condo Stunning 3-bedroom luxury condominium with panoramic water views. Features marble countertops, high-end appliances, and access to exclusive amenities.',
     2.50,
     NOW()),
    
    (gen_random_uuid(),
     'Suburban Family House',
     'Spacious 4-bedroom family house in quiet suburban neighborhood. Large backyard, 2-car garage, and excellent school district.',
     2800,
     'Suburbs',
     4,
     3,
     2200,
     ARRAY['garage', 'yard', 'dishwasher', 'fireplace'],
     '{"phone": "555-0104", "email": "suburban@example.com"}',
     ARRAY['https://example.com/house1.jpg', 'https://example.com/house2.jpg'],
     'house',
     'Suburban Family House Spacious 4-bedroom family house in quiet suburban neighborhood. Large backyard, 2-car garage, and excellent school district.',
     1.27,
     NOW()),
    
    (gen_random_uuid(),
     'Modern Midtown Loft',
     'Industrial-chic loft in trendy Midtown district. Exposed brick walls, high ceilings, and modern amenities. Perfect for young professionals.',
     2200,
     'Midtown',
     2,
     2,
     1100,
     ARRAY['exposed brick', 'high ceilings', 'modern kitchen', 'parking'],
     '{"phone": "555-0105", "email": "midtown@example.com"}',
     ARRAY['https://example.com/loft1.jpg'],
     'loft',
     'Modern Midtown Loft Industrial-chic loft in trendy Midtown district. Exposed brick walls, high ceilings, and modern amenities. Perfect for young professionals.',
     2.00,
     NOW()),
    
    (gen_random_uuid(),
     'Budget-Friendly Apartment',
     'Affordable 1-bedroom apartment in safe neighborhood. Basic amenities, recently renovated, great for first-time renters.',
     950,
     'East Side',
     1,
     1,
     550,
     ARRAY['laundry', 'parking'],
     '{"phone": "555-0106", "email": "budget@example.com"}',
     ARRAY['https://example.com/budget1.jpg'],
     'apartment',
     'Budget-Friendly Apartment Affordable 1-bedroom apartment in safe neighborhood. Basic amenities, recently renovated, great for first-time renters.',
     1.73,
     NOW()),
    
    (gen_random_uuid(),
     'Upscale Townhouse',
     'Beautiful 3-bedroom townhouse with private entrance and small patio. Modern finishes throughout, in-unit washer/dryer.',
     2600,
     'Uptown',
     3,
     2.5,
     1350,
     ARRAY['private entrance', 'patio', 'laundry', 'dishwasher'],
     '{"phone": "555-0107", "email": "uptown@example.com"}',
     ARRAY['https://example.com/townhouse1.jpg', 'https://example.com/townhouse2.jpg'],
     'townhouse',
     'Upscale Townhouse Beautiful 3-bedroom townhouse with private entrance and small patio. Modern finishes throughout, in-unit washer/dryer.',
     1.93,
     NOW()),
    
    (gen_random_uuid(),
     'Pet-Friendly Garden Apartment',
     'Ground floor apartment with direct access to garden area. Perfect for pet owners! Spacious rooms and pet washing station.',
     1600,
     'Garden District',
     2,
     1.5,
     800,
     ARRAY['pet friendly', 'garden access', 'pet wash station', 'parking'],
     '{"phone": "555-0108", "email": "garden@example.com"}',
     ARRAY['https://example.com/garden1.jpg'],
     'apartment',
     'Pet-Friendly Garden Apartment Ground floor apartment with direct access to garden area. Perfect for pet owners! Spacious rooms and pet washing station.',
     2.00,
     NOW());

-- =============================================================================
-- Test User Interactions
-- =============================================================================

-- Get user IDs for creating interactions
DO $$
DECLARE
    user1_id UUID;
    user2_id UUID;
    user3_id UUID;
    prop1_id UUID;
    prop2_id UUID;
    prop3_id UUID;
    prop4_id UUID;
BEGIN
    -- Get some user IDs
    SELECT id INTO user1_id FROM users WHERE email = 'john.doe@example.com';
    SELECT id INTO user2_id FROM users WHERE email = 'jane.smith@example.com';
    SELECT id INTO user3_id FROM users WHERE email = 'mike.johnson@example.com';
    
    -- Get some property IDs
    SELECT id INTO prop1_id FROM properties WHERE title = 'Modern Downtown Apartment';
    SELECT id INTO prop2_id FROM properties WHERE title = 'Cozy Studio Near University';
    SELECT id INTO prop3_id FROM properties WHERE title = 'Luxury Waterfront Condo';
    SELECT id INTO prop4_id FROM properties WHERE title = 'Suburban Family House';
    
    -- Create user interactions
    INSERT INTO user_interactions (user_id, property_id, interaction_type, timestamp, duration_seconds)
    VALUES 
        -- John's interactions
        (user1_id, prop1_id, 'view', NOW() - INTERVAL '1 day', 45),
        (user1_id, prop1_id, 'favorite', NOW() - INTERVAL '1 day', NULL),
        (user1_id, prop2_id, 'view', NOW() - INTERVAL '2 days', 30),
        (user1_id, prop3_id, 'view', NOW() - INTERVAL '3 days', 120),
        
        -- Jane's interactions
        (user2_id, prop3_id, 'view', NOW() - INTERVAL '1 hour', 90),
        (user2_id, prop4_id, 'view', NOW() - INTERVAL '2 hours', 180),
        (user2_id, prop4_id, 'favorite', NOW() - INTERVAL '2 hours', NULL),
        (user2_id, prop4_id, 'contact', NOW() - INTERVAL '1 hour', NULL),
        
        -- Mike's interactions
        (user3_id, prop2_id, 'view', NOW() - INTERVAL '30 minutes', 240),
        (user3_id, prop2_id, 'favorite', NOW() - INTERVAL '30 minutes', NULL),
        (user3_id, prop1_id, 'view', NOW() - INTERVAL '1 hour', 60);
        
END $$;

-- =============================================================================
-- Test ML Models (Empty placeholders)
-- =============================================================================

INSERT INTO ml_models (model_name, version, model_data, metadata)
VALUES 
    ('collaborative_filter', 'v1.0.0', E'\\x', '{"accuracy": 0.85, "training_date": "2024-01-15", "status": "placeholder"}'),
    ('content_recommender', 'v1.0.0', E'\\x', '{"accuracy": 0.78, "training_date": "2024-01-15", "status": "placeholder"}'),
    ('hybrid_recommender', 'v1.0.0', E'\\x', '{"accuracy": 0.90, "training_date": "2024-01-15", "status": "placeholder"}'),
    ('search_ranker', 'v1.0.0', E'\\x', '{"ndcg": 0.82, "training_date": "2024-01-15", "status": "placeholder"}');

-- =============================================================================
-- Success Message
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '==================================================';
    RAISE NOTICE 'Development seed data loaded successfully!';
    RAISE NOTICE 'Created:';
    RAISE NOTICE '  - 5 test users';
    RAISE NOTICE '  - 8 test properties';
    RAISE NOTICE '  - Sample user interactions';
    RAISE NOTICE '  - Placeholder ML models';
    RAISE NOTICE '==================================================';
    RAISE NOTICE '';
END $$;