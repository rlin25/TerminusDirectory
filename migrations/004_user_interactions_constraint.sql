-- Migration: 004_user_interactions_constraint.sql
-- Description: Add unique constraint for user interactions to prevent duplicates
-- Created: 2025-07-14
-- Author: Production Database Enhancement

-- Add unique constraint for user interactions to prevent duplicate entries
-- This supports the ON CONFLICT clause in the repository implementation
ALTER TABLE user_interactions 
ADD CONSTRAINT user_interactions_unique_interaction 
UNIQUE (user_id, property_id, interaction_type, timestamp);

-- Create additional indexes for better performance
CREATE INDEX IF NOT EXISTS idx_user_interactions_user_property 
ON user_interactions(user_id, property_id);

CREATE INDEX IF NOT EXISTS idx_user_interactions_type_timestamp 
ON user_interactions(interaction_type, timestamp);

-- Add comment to document the constraint
COMMENT ON CONSTRAINT user_interactions_unique_interaction ON user_interactions IS 
'Prevents duplicate user interactions for the same property at the same timestamp';