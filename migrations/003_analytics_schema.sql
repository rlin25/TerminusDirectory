-- Migration: 003_analytics_schema.sql
-- Description: Create comprehensive analytics and business intelligence schema
-- Created: 2025-07-13
-- Author: Claude Code

-- Enable necessary extensions for analytics
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_cron";

-- Create custom types for analytics
CREATE TYPE metric_type AS ENUM ('gauge', 'counter', 'histogram', 'summary');
CREATE TYPE event_type AS ENUM ('user_action', 'system_event', 'business_event', 'ml_event');
CREATE TYPE aggregation_level AS ENUM ('minute', 'hour', 'day', 'week', 'month', 'year');
CREATE TYPE report_status AS ENUM ('pending', 'running', 'completed', 'failed', 'cancelled');
CREATE TYPE alert_severity AS ENUM ('info', 'warning', 'error', 'critical');

-- ============================================
-- ANALYTICS EVENTS TABLE (Time-series)
-- ============================================
CREATE TABLE analytics_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event_type event_type NOT NULL,
    event_name VARCHAR(255) NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    session_id VARCHAR(255),
    
    -- Event properties as JSONB for flexibility
    properties JSONB DEFAULT '{}',
    
    -- Contextual information
    ip_address INET,
    user_agent TEXT,
    referrer TEXT,
    
    -- Geographic information
    country VARCHAR(2),
    region VARCHAR(100),
    city VARCHAR(100),
    
    -- Device and browser information
    device_type VARCHAR(50),
    browser_name VARCHAR(50),
    browser_version VARCHAR(50),
    os_name VARCHAR(50),
    os_version VARCHAR(50),
    
    -- Business context
    property_id UUID,
    search_id VARCHAR(255),
    recommendation_id VARCHAR(255),
    
    -- ML model context
    model_name VARCHAR(255),
    model_version VARCHAR(50),
    prediction_score DECIMAL(5,4),
    
    -- Processing metadata
    processed_at TIMESTAMP WITH TIME ZONE,
    batch_id UUID,
    
    -- Constraints
    CONSTRAINT analytics_events_prediction_score_check CHECK (prediction_score IS NULL OR (prediction_score >= 0 AND prediction_score <= 1))
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('analytics_events', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- ============================================
-- BUSINESS METRICS TABLE
-- ============================================
CREATE TABLE business_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(255) NOT NULL,
    metric_type metric_type NOT NULL DEFAULT 'gauge',
    metric_value DECIMAL(15,4) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Metric dimensions for segmentation
    dimensions JSONB DEFAULT '{}',
    
    -- Metadata
    source VARCHAR(100),
    tags TEXT[],
    description TEXT,
    
    -- Data quality tracking
    confidence_score DECIMAL(3,2) DEFAULT 1.0,
    data_source_count INTEGER DEFAULT 1,
    
    -- Constraints
    CONSTRAINT business_metrics_confidence_check CHECK (confidence_score >= 0 AND confidence_score <= 1),
    CONSTRAINT business_metrics_data_source_check CHECK (data_source_count > 0)
);

-- Convert to hypertable
SELECT create_hypertable('business_metrics', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- ============================================
-- KPI DEFINITIONS TABLE
-- ============================================
CREATE TABLE kpi_definitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    kpi_name VARCHAR(255) NOT NULL UNIQUE,
    kpi_description TEXT,
    calculation_formula TEXT NOT NULL,
    target_value DECIMAL(15,4),
    warning_threshold DECIMAL(15,4),
    critical_threshold DECIMAL(15,4),
    
    -- Thresholds configuration
    higher_is_better BOOLEAN DEFAULT TRUE,
    unit VARCHAR(50),
    category VARCHAR(100),
    
    -- Calculation metadata
    calculation_window INTERVAL DEFAULT INTERVAL '1 day',
    update_frequency INTERVAL DEFAULT INTERVAL '1 hour',
    
    -- Status and ownership
    is_active BOOLEAN DEFAULT TRUE,
    owner_team VARCHAR(100),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- KPI VALUES TABLE (Time-series)
-- ============================================
CREATE TABLE kpi_values (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    kpi_id UUID NOT NULL REFERENCES kpi_definitions(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    value DECIMAL(15,4) NOT NULL,
    
    -- Context and segmentation
    segment_dimensions JSONB DEFAULT '{}',
    
    -- Calculation metadata
    calculation_duration_ms INTEGER,
    data_points_count INTEGER,
    confidence_score DECIMAL(3,2) DEFAULT 1.0,
    
    -- Anomaly detection
    is_anomaly BOOLEAN DEFAULT FALSE,
    anomaly_score DECIMAL(5,4),
    
    -- Constraints
    CONSTRAINT kpi_values_confidence_check CHECK (confidence_score >= 0 AND confidence_score <= 1),
    CONSTRAINT kpi_values_anomaly_score_check CHECK (anomaly_score IS NULL OR (anomaly_score >= 0 AND anomaly_score <= 1))
);

-- Convert to hypertable
SELECT create_hypertable('kpi_values', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- ============================================
-- USER BEHAVIOR ANALYTICS TABLE
-- ============================================
CREATE TABLE user_behavior_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Session metrics
    session_duration_seconds INTEGER,
    page_views INTEGER DEFAULT 0,
    property_views INTEGER DEFAULT 0,
    search_queries INTEGER DEFAULT 0,
    interactions INTEGER DEFAULT 0,
    
    -- Engagement metrics
    bounce_rate DECIMAL(5,4),
    engagement_score DECIMAL(5,4),
    conversion_score DECIMAL(5,4),
    
    -- User journey
    entry_page VARCHAR(500),
    exit_page VARCHAR(500),
    referrer_type VARCHAR(100),
    traffic_source VARCHAR(100),
    campaign_id VARCHAR(255),
    
    -- Device and context
    device_category VARCHAR(50),
    browser_category VARCHAR(50),
    location_category VARCHAR(100),
    
    -- Behavioral segments
    user_segment VARCHAR(100),
    predicted_intent VARCHAR(100),
    risk_score DECIMAL(5,4),
    
    -- Constraints
    CONSTRAINT user_behavior_bounce_rate_check CHECK (bounce_rate IS NULL OR (bounce_rate >= 0 AND bounce_rate <= 1)),
    CONSTRAINT user_behavior_engagement_check CHECK (engagement_score IS NULL OR (engagement_score >= 0 AND engagement_score <= 1)),
    CONSTRAINT user_behavior_conversion_check CHECK (conversion_score IS NULL OR (conversion_score >= 0 AND conversion_score <= 1)),
    CONSTRAINT user_behavior_risk_check CHECK (risk_score IS NULL OR (risk_score >= 0 AND risk_score <= 1))
);

-- Convert to hypertable
SELECT create_hypertable('user_behavior_analytics', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- ============================================
-- PROPERTY ANALYTICS TABLE
-- ============================================
CREATE TABLE property_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    property_id UUID NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Performance metrics
    view_count INTEGER DEFAULT 0,
    unique_viewers INTEGER DEFAULT 0,
    inquiries_count INTEGER DEFAULT 0,
    favorites_count INTEGER DEFAULT 0,
    contact_requests INTEGER DEFAULT 0,
    
    -- Engagement metrics
    avg_view_duration DECIMAL(8,2),
    bounce_rate DECIMAL(5,4),
    conversion_rate DECIMAL(5,4),
    
    -- Recommendation metrics
    recommended_count INTEGER DEFAULT 0,
    recommendation_click_rate DECIMAL(5,4),
    recommendation_conversion_rate DECIMAL(5,4),
    
    -- Market position
    market_rank INTEGER,
    price_competitiveness_score DECIMAL(5,4),
    feature_score DECIMAL(5,4),
    location_score DECIMAL(5,4),
    
    -- Calculated metrics
    revenue_generated DECIMAL(12,2) DEFAULT 0,
    cost_per_lead DECIMAL(8,2),
    roi_score DECIMAL(5,4),
    
    -- Data aggregation metadata
    aggregation_period INTERVAL DEFAULT INTERVAL '1 hour',
    data_quality_score DECIMAL(3,2) DEFAULT 1.0,
    
    -- Constraints
    CONSTRAINT property_analytics_bounce_rate_check CHECK (bounce_rate IS NULL OR (bounce_rate >= 0 AND bounce_rate <= 1)),
    CONSTRAINT property_analytics_conversion_rate_check CHECK (conversion_rate IS NULL OR (conversion_rate >= 0 AND conversion_rate <= 1)),
    CONSTRAINT property_analytics_rec_click_rate_check CHECK (recommendation_click_rate IS NULL OR (recommendation_click_rate >= 0 AND recommendation_click_rate <= 1)),
    CONSTRAINT property_analytics_rec_conversion_rate_check CHECK (recommendation_conversion_rate IS NULL OR (recommendation_conversion_rate >= 0 AND recommendation_conversion_rate <= 1)),
    CONSTRAINT property_analytics_scores_check CHECK (
        (price_competitiveness_score IS NULL OR (price_competitiveness_score >= 0 AND price_competitiveness_score <= 1)) AND
        (feature_score IS NULL OR (feature_score >= 0 AND feature_score <= 1)) AND
        (location_score IS NULL OR (location_score >= 0 AND location_score <= 1)) AND
        (roi_score IS NULL OR (roi_score >= 0 AND roi_score <= 1))
    )
);

-- Convert to hypertable
SELECT create_hypertable('property_analytics', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- ============================================
-- ML MODEL PERFORMANCE TABLE
-- ============================================
CREATE TABLE ml_model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Performance metrics
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    auc_score DECIMAL(5,4),
    
    -- Inference metrics
    predictions_count INTEGER DEFAULT 0,
    avg_inference_time_ms DECIMAL(8,2),
    p95_inference_time_ms DECIMAL(8,2),
    p99_inference_time_ms DECIMAL(8,2),
    
    -- Business impact metrics
    recommendation_accuracy DECIMAL(5,4),
    click_through_rate DECIMAL(5,4),
    conversion_rate DECIMAL(5,4),
    revenue_impact DECIMAL(12,2),
    
    -- Model drift detection
    feature_drift_score DECIMAL(5,4),
    prediction_drift_score DECIMAL(5,4),
    data_quality_score DECIMAL(5,4),
    concept_drift_detected BOOLEAN DEFAULT FALSE,
    
    -- Resource utilization
    cpu_usage_percent DECIMAL(5,2),
    memory_usage_mb INTEGER,
    gpu_usage_percent DECIMAL(5,2),
    
    -- Model metadata
    training_dataset_size INTEGER,
    test_dataset_size INTEGER,
    feature_count INTEGER,
    
    -- Constraints
    CONSTRAINT ml_performance_accuracy_check CHECK (accuracy IS NULL OR (accuracy >= 0 AND accuracy <= 1)),
    CONSTRAINT ml_performance_precision_check CHECK (precision_score IS NULL OR (precision_score >= 0 AND precision_score <= 1)),
    CONSTRAINT ml_performance_recall_check CHECK (recall_score IS NULL OR (recall_score >= 0 AND recall_score <= 1)),
    CONSTRAINT ml_performance_f1_check CHECK (f1_score IS NULL OR (f1_score >= 0 AND f1_score <= 1)),
    CONSTRAINT ml_performance_auc_check CHECK (auc_score IS NULL OR (auc_score >= 0 AND auc_score <= 1)),
    CONSTRAINT ml_performance_rec_accuracy_check CHECK (recommendation_accuracy IS NULL OR (recommendation_accuracy >= 0 AND recommendation_accuracy <= 1)),
    CONSTRAINT ml_performance_ctr_check CHECK (click_through_rate IS NULL OR (click_through_rate >= 0 AND click_through_rate <= 1)),
    CONSTRAINT ml_performance_conversion_check CHECK (conversion_rate IS NULL OR (conversion_rate >= 0 AND conversion_rate <= 1)),
    CONSTRAINT ml_performance_drift_checks CHECK (
        (feature_drift_score IS NULL OR (feature_drift_score >= 0 AND feature_drift_score <= 1)) AND
        (prediction_drift_score IS NULL OR (prediction_drift_score >= 0 AND prediction_drift_score <= 1)) AND
        (data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 1))
    )
);

-- Convert to hypertable
SELECT create_hypertable('ml_model_performance', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- ============================================
-- REVENUE ANALYTICS TABLE
-- ============================================
CREATE TABLE revenue_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Revenue metrics
    total_revenue DECIMAL(15,2) NOT NULL DEFAULT 0,
    subscription_revenue DECIMAL(15,2) DEFAULT 0,
    commission_revenue DECIMAL(15,2) DEFAULT 0,
    advertising_revenue DECIMAL(15,2) DEFAULT 0,
    other_revenue DECIMAL(15,2) DEFAULT 0,
    
    -- Cost metrics
    acquisition_cost DECIMAL(15,2) DEFAULT 0,
    operational_cost DECIMAL(15,2) DEFAULT 0,
    technology_cost DECIMAL(15,2) DEFAULT 0,
    marketing_cost DECIMAL(15,2) DEFAULT 0,
    
    -- Calculated metrics
    gross_profit DECIMAL(15,2),
    net_profit DECIMAL(15,2),
    profit_margin DECIMAL(5,4),
    
    -- Customer metrics
    new_customers INTEGER DEFAULT 0,
    churned_customers INTEGER DEFAULT 0,
    customer_lifetime_value DECIMAL(10,2),
    customer_acquisition_cost DECIMAL(8,2),
    
    -- Geographic and segment breakdown
    region VARCHAR(100),
    customer_segment VARCHAR(100),
    product_category VARCHAR(100),
    
    -- Time aggregation
    aggregation_level aggregation_level NOT NULL DEFAULT 'hour',
    aggregation_period INTERVAL DEFAULT INTERVAL '1 hour',
    
    -- Data quality
    data_completeness DECIMAL(3,2) DEFAULT 1.0,
    
    -- Constraints
    CONSTRAINT revenue_analytics_profit_margin_check CHECK (profit_margin IS NULL OR (profit_margin >= -1 AND profit_margin <= 1)),
    CONSTRAINT revenue_analytics_completeness_check CHECK (data_completeness >= 0 AND data_completeness <= 1)
);

-- Convert to hypertable
SELECT create_hypertable('revenue_analytics', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- ============================================
-- MARKET ANALYTICS TABLE
-- ============================================
CREATE TABLE market_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Market identifiers
    region VARCHAR(100) NOT NULL,
    property_type VARCHAR(50),
    price_range VARCHAR(50),
    
    -- Market metrics
    avg_price DECIMAL(12,2),
    median_price DECIMAL(12,2),
    price_per_sqft DECIMAL(8,2),
    inventory_count INTEGER DEFAULT 0,
    new_listings INTEGER DEFAULT 0,
    expired_listings INTEGER DEFAULT 0,
    
    -- Market dynamics
    supply_demand_ratio DECIMAL(5,4),
    price_trend_direction VARCHAR(20), -- 'increasing', 'decreasing', 'stable'
    price_change_percent DECIMAL(5,4),
    velocity_score DECIMAL(5,4),
    
    -- Competition analysis
    competitor_count INTEGER DEFAULT 0,
    market_share DECIMAL(5,4),
    competitive_advantage_score DECIMAL(5,4),
    
    -- Demand indicators
    search_volume INTEGER DEFAULT 0,
    inquiry_rate DECIMAL(5,4),
    conversion_rate DECIMAL(5,4),
    
    -- Seasonal factors
    seasonality_factor DECIMAL(5,4),
    trend_strength DECIMAL(5,4),
    forecast_confidence DECIMAL(5,4),
    
    -- Aggregation metadata
    aggregation_level aggregation_level NOT NULL DEFAULT 'day',
    data_sources_count INTEGER DEFAULT 1,
    
    -- Constraints
    CONSTRAINT market_analytics_supply_demand_check CHECK (supply_demand_ratio IS NULL OR supply_demand_ratio >= 0),
    CONSTRAINT market_analytics_price_change_check CHECK (price_change_percent IS NULL OR (price_change_percent >= -1 AND price_change_percent <= 10)),
    CONSTRAINT market_analytics_velocity_check CHECK (velocity_score IS NULL OR (velocity_score >= 0 AND velocity_score <= 1)),
    CONSTRAINT market_analytics_market_share_check CHECK (market_share IS NULL OR (market_share >= 0 AND market_share <= 1)),
    CONSTRAINT market_analytics_scores_check CHECK (
        (competitive_advantage_score IS NULL OR (competitive_advantage_score >= 0 AND competitive_advantage_score <= 1)) AND
        (inquiry_rate IS NULL OR (inquiry_rate >= 0 AND inquiry_rate <= 1)) AND
        (conversion_rate IS NULL OR (conversion_rate >= 0 AND conversion_rate <= 1)) AND
        (seasonality_factor IS NULL OR (seasonality_factor >= 0 AND seasonality_factor <= 2)) AND
        (trend_strength IS NULL OR (trend_strength >= 0 AND trend_strength <= 1)) AND
        (forecast_confidence IS NULL OR (forecast_confidence >= 0 AND forecast_confidence <= 1))
    )
);

-- Convert to hypertable
SELECT create_hypertable('market_analytics', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- ============================================
-- REPORTS TABLE
-- ============================================
CREATE TABLE reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    report_name VARCHAR(255) NOT NULL,
    report_type VARCHAR(100) NOT NULL,
    report_config JSONB NOT NULL,
    
    -- Scheduling
    schedule_config JSONB,
    next_run_time TIMESTAMP WITH TIME ZONE,
    
    -- Execution tracking
    status report_status DEFAULT 'pending',
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Access control
    is_public BOOLEAN DEFAULT FALSE,
    allowed_users UUID[],
    allowed_roles TEXT[],
    
    -- Output configuration
    output_format VARCHAR(50) DEFAULT 'json',
    output_location TEXT,
    retention_days INTEGER DEFAULT 30,
    
    -- Performance tracking
    avg_execution_time_seconds INTEGER,
    last_success_at TIMESTAMP WITH TIME ZONE,
    last_failure_at TIMESTAMP WITH TIME ZONE,
    failure_count INTEGER DEFAULT 0,
    
    -- Data source tracking
    data_sources TEXT[],
    dependencies TEXT[]
);

-- ============================================
-- REPORT EXECUTIONS TABLE
-- ============================================
CREATE TABLE report_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    report_id UUID NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    execution_id VARCHAR(255) NOT NULL,
    
    -- Execution details
    status report_status NOT NULL DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Results
    output_data JSONB,
    output_file_path TEXT,
    output_size_bytes BIGINT,
    
    -- Performance metrics
    execution_time_seconds INTEGER,
    records_processed INTEGER,
    memory_usage_mb INTEGER,
    
    -- Error handling
    error_message TEXT,
    error_details JSONB,
    retry_count INTEGER DEFAULT 0,
    
    -- Metadata
    execution_parameters JSONB,
    data_range_start TIMESTAMP WITH TIME ZONE,
    data_range_end TIMESTAMP WITH TIME ZONE
);

-- ============================================
-- ALERTS TABLE
-- ============================================
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_name VARCHAR(255) NOT NULL,
    alert_type VARCHAR(100) NOT NULL,
    severity alert_severity NOT NULL DEFAULT 'warning',
    
    -- Alert condition
    condition_config JSONB NOT NULL,
    threshold_value DECIMAL(15,4),
    
    -- Notification settings
    notification_channels TEXT[],
    recipient_users UUID[],
    recipient_groups TEXT[],
    
    -- Status and timing
    is_active BOOLEAN DEFAULT TRUE,
    triggered_at TIMESTAMP WITH TIME ZONE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    last_check_at TIMESTAMP WITH TIME ZONE,
    
    -- Frequency control
    check_frequency INTERVAL DEFAULT INTERVAL '5 minutes',
    cooldown_period INTERVAL DEFAULT INTERVAL '1 hour',
    max_alerts_per_hour INTEGER DEFAULT 10,
    
    -- Metadata
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Context
    metric_name VARCHAR(255),
    entity_type VARCHAR(100),
    entity_id VARCHAR(255),
    
    -- Alert history tracking
    trigger_count INTEGER DEFAULT 0,
    false_positive_count INTEGER DEFAULT 0,
    accuracy_score DECIMAL(3,2)
);

-- ============================================
-- ALERT NOTIFICATIONS TABLE
-- ============================================
CREATE TABLE alert_notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_id UUID NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
    
    -- Notification details
    channel_type VARCHAR(50) NOT NULL, -- 'email', 'slack', 'webhook', 'sms'
    recipient VARCHAR(255) NOT NULL,
    subject VARCHAR(500),
    message TEXT NOT NULL,
    
    -- Status tracking
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'sent', 'failed', 'delivered'
    sent_at TIMESTAMP WITH TIME ZONE,
    delivered_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    notification_metadata JSONB,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Context
    alert_severity alert_severity,
    alert_value DECIMAL(15,4),
    threshold_value DECIMAL(15,4),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- DATA LINEAGE TABLE
-- ============================================
CREATE TABLE data_lineage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Source information
    source_table VARCHAR(255) NOT NULL,
    source_column VARCHAR(255),
    source_query_hash VARCHAR(64),
    
    -- Target information
    target_table VARCHAR(255) NOT NULL,
    target_column VARCHAR(255),
    
    -- Transformation details
    transformation_type VARCHAR(100),
    transformation_logic TEXT,
    transformation_hash VARCHAR(64),
    
    -- Dependencies
    dependencies JSONB,
    upstream_tables TEXT[],
    downstream_tables TEXT[],
    
    -- Quality and governance
    data_quality_score DECIMAL(3,2),
    business_criticality VARCHAR(50), -- 'low', 'medium', 'high', 'critical'
    compliance_requirements TEXT[],
    
    -- Timing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_validated_at TIMESTAMP WITH TIME ZONE,
    
    -- Ownership
    owner_team VARCHAR(100),
    steward_user UUID REFERENCES users(id) ON DELETE SET NULL
);

-- ============================================
-- CREATE INDEXES FOR ANALYTICS PERFORMANCE
-- ============================================

-- Analytics events indexes
CREATE INDEX idx_analytics_events_timestamp ON analytics_events(timestamp DESC);
CREATE INDEX idx_analytics_events_type ON analytics_events(event_type);
CREATE INDEX idx_analytics_events_name ON analytics_events(event_name);
CREATE INDEX idx_analytics_events_user_id ON analytics_events(user_id);
CREATE INDEX idx_analytics_events_session_id ON analytics_events(session_id);
CREATE INDEX idx_analytics_events_property_id ON analytics_events(property_id);
CREATE INDEX idx_analytics_events_model ON analytics_events(model_name, model_version);
CREATE INDEX idx_analytics_events_properties_gin ON analytics_events USING GIN(properties);
CREATE INDEX idx_analytics_events_user_timestamp ON analytics_events(user_id, timestamp DESC);
CREATE INDEX idx_analytics_events_type_timestamp ON analytics_events(event_type, timestamp DESC);

-- Business metrics indexes
CREATE INDEX idx_business_metrics_timestamp ON business_metrics(timestamp DESC);
CREATE INDEX idx_business_metrics_name ON business_metrics(metric_name);
CREATE INDEX idx_business_metrics_type ON business_metrics(metric_type);
CREATE INDEX idx_business_metrics_source ON business_metrics(source);
CREATE INDEX idx_business_metrics_name_timestamp ON business_metrics(metric_name, timestamp DESC);
CREATE INDEX idx_business_metrics_dimensions_gin ON business_metrics USING GIN(dimensions);

-- KPI indexes
CREATE INDEX idx_kpi_definitions_name ON kpi_definitions(kpi_name);
CREATE INDEX idx_kpi_definitions_category ON kpi_definitions(category);
CREATE INDEX idx_kpi_definitions_active ON kpi_definitions(is_active);
CREATE INDEX idx_kpi_values_kpi_id_timestamp ON kpi_values(kpi_id, timestamp DESC);
CREATE INDEX idx_kpi_values_timestamp ON kpi_values(timestamp DESC);
CREATE INDEX idx_kpi_values_anomaly ON kpi_values(is_anomaly, timestamp DESC);

-- User behavior analytics indexes
CREATE INDEX idx_user_behavior_user_id ON user_behavior_analytics(user_id);
CREATE INDEX idx_user_behavior_session_id ON user_behavior_analytics(session_id);
CREATE INDEX idx_user_behavior_timestamp ON user_behavior_analytics(timestamp DESC);
CREATE INDEX idx_user_behavior_user_timestamp ON user_behavior_analytics(user_id, timestamp DESC);
CREATE INDEX idx_user_behavior_segment ON user_behavior_analytics(user_segment);

-- Property analytics indexes
CREATE INDEX idx_property_analytics_property_id ON property_analytics(property_id);
CREATE INDEX idx_property_analytics_timestamp ON property_analytics(timestamp DESC);
CREATE INDEX idx_property_analytics_property_timestamp ON property_analytics(property_id, timestamp DESC);

-- ML model performance indexes
CREATE INDEX idx_ml_performance_model ON ml_model_performance(model_name, model_version);
CREATE INDEX idx_ml_performance_timestamp ON ml_model_performance(timestamp DESC);
CREATE INDEX idx_ml_performance_drift ON ml_model_performance(concept_drift_detected, timestamp DESC);

-- Revenue analytics indexes
CREATE INDEX idx_revenue_analytics_timestamp ON revenue_analytics(timestamp DESC);
CREATE INDEX idx_revenue_analytics_region ON revenue_analytics(region);
CREATE INDEX idx_revenue_analytics_segment ON revenue_analytics(customer_segment);
CREATE INDEX idx_revenue_analytics_aggregation ON revenue_analytics(aggregation_level, timestamp DESC);

-- Market analytics indexes
CREATE INDEX idx_market_analytics_region ON market_analytics(region);
CREATE INDEX idx_market_analytics_property_type ON market_analytics(property_type);
CREATE INDEX idx_market_analytics_timestamp ON market_analytics(timestamp DESC);
CREATE INDEX idx_market_analytics_region_timestamp ON market_analytics(region, timestamp DESC);

-- Reports indexes
CREATE INDEX idx_reports_type ON reports(report_type);
CREATE INDEX idx_reports_status ON reports(status);
CREATE INDEX idx_reports_created_by ON reports(created_by);
CREATE INDEX idx_reports_next_run ON reports(next_run_time);
CREATE INDEX idx_report_executions_report_id ON report_executions(report_id);
CREATE INDEX idx_report_executions_status ON report_executions(status);
CREATE INDEX idx_report_executions_started_at ON report_executions(started_at DESC);

-- Alerts indexes
CREATE INDEX idx_alerts_type ON alerts(alert_type);
CREATE INDEX idx_alerts_severity ON alerts(severity);
CREATE INDEX idx_alerts_active ON alerts(is_active);
CREATE INDEX idx_alerts_triggered_at ON alerts(triggered_at DESC);
CREATE INDEX idx_alerts_metric ON alerts(metric_name);
CREATE INDEX idx_alert_notifications_alert_id ON alert_notifications(alert_id);
CREATE INDEX idx_alert_notifications_status ON alert_notifications(status);
CREATE INDEX idx_alert_notifications_sent_at ON alert_notifications(sent_at DESC);

-- Data lineage indexes
CREATE INDEX idx_data_lineage_source_table ON data_lineage(source_table);
CREATE INDEX idx_data_lineage_target_table ON data_lineage(target_table);
CREATE INDEX idx_data_lineage_owner ON data_lineage(owner_team);
CREATE INDEX idx_data_lineage_criticality ON data_lineage(business_criticality);

-- ============================================
-- MATERIALIZED VIEWS FOR PERFORMANCE
-- ============================================

-- Real-time dashboard summary
CREATE MATERIALIZED VIEW dashboard_summary AS
SELECT 
    DATE_TRUNC('hour', timestamp) as hour,
    COUNT(DISTINCT user_id) as active_users,
    COUNT(*) FILTER (WHERE event_type = 'user_action') as user_actions,
    COUNT(*) FILTER (WHERE event_name = 'property_view') as property_views,
    COUNT(*) FILTER (WHERE event_name = 'search_query') as searches,
    COUNT(*) FILTER (WHERE event_name = 'recommendation_click') as recommendation_clicks,
    AVG(CASE WHEN properties->>'response_time_ms' IS NOT NULL 
        THEN (properties->>'response_time_ms')::numeric END) as avg_response_time
FROM analytics_events
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY hour DESC;

-- Property performance summary
CREATE MATERIALIZED VIEW property_performance_summary AS
SELECT 
    property_id,
    DATE_TRUNC('day', timestamp) as day,
    SUM(view_count) as total_views,
    SUM(unique_viewers) as total_unique_viewers,
    SUM(inquiries_count) as total_inquiries,
    AVG(conversion_rate) as avg_conversion_rate,
    AVG(engagement_score) as avg_engagement_score
FROM property_analytics
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY property_id, DATE_TRUNC('day', timestamp);

-- User segment analysis
CREATE MATERIALIZED VIEW user_segment_analysis AS
SELECT 
    user_segment,
    DATE_TRUNC('day', timestamp) as day,
    COUNT(DISTINCT user_id) as unique_users,
    AVG(session_duration_seconds) as avg_session_duration,
    AVG(engagement_score) as avg_engagement_score,
    AVG(conversion_score) as avg_conversion_score,
    SUM(property_views) as total_property_views
FROM user_behavior_analytics
WHERE timestamp >= NOW() - INTERVAL '30 days'
  AND user_segment IS NOT NULL
GROUP BY user_segment, DATE_TRUNC('day', timestamp);

-- ============================================
-- TRIGGERS FOR REAL-TIME UPDATES
-- ============================================

-- Function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_analytics_views()
RETURNS TRIGGER AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY dashboard_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY property_performance_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY user_segment_analysis;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Trigger to refresh views on data changes (with debouncing)
CREATE OR REPLACE FUNCTION schedule_view_refresh()
RETURNS TRIGGER AS $$
BEGIN
    -- Use pg_cron to schedule refresh every 5 minutes
    -- This prevents too frequent refreshes
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- CONTINUOUS AGGREGATES (TimescaleDB)
-- ============================================

-- Hourly analytics events aggregation
CREATE MATERIALIZED VIEW analytics_events_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', timestamp) as bucket,
    event_type,
    event_name,
    COUNT(*) as event_count,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(DISTINCT session_id) as unique_sessions
FROM analytics_events
GROUP BY bucket, event_type, event_name
WITH NO DATA;

-- Daily KPI aggregation
CREATE MATERIALIZED VIEW kpi_values_daily
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', timestamp) as bucket,
    kpi_id,
    AVG(value) as avg_value,
    MIN(value) as min_value,
    MAX(value) as max_value,
    COUNT(*) as data_points,
    STDDEV(value) as std_deviation
FROM kpi_values
GROUP BY bucket, kpi_id
WITH NO DATA;

-- Enable continuous aggregate policies
SELECT add_continuous_aggregate_policy('analytics_events_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('kpi_values_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 hour');

-- ============================================
-- DATA RETENTION POLICIES
-- ============================================

-- Retention policy for raw analytics events (90 days)
SELECT add_retention_policy('analytics_events', INTERVAL '90 days');

-- Retention policy for business metrics (1 year)
SELECT add_retention_policy('business_metrics', INTERVAL '1 year');

-- Retention policy for user behavior analytics (6 months)
SELECT add_retention_policy('user_behavior_analytics', INTERVAL '6 months');

-- ============================================
-- SECURITY AND PERMISSIONS
-- ============================================

-- Grant permissions to analytics user
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO rental_ml_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO rental_ml_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO rental_ml_user;

-- Create analytics read-only role
CREATE ROLE analytics_readonly;
GRANT CONNECT ON DATABASE rental_ml TO analytics_readonly;
GRANT USAGE ON SCHEMA public TO analytics_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analytics_readonly;
GRANT SELECT ON ALL MATERIALIZED VIEWS IN SCHEMA public TO analytics_readonly;

-- Create analytics admin role
CREATE ROLE analytics_admin;
GRANT CONNECT ON DATABASE rental_ml TO analytics_admin;
GRANT USAGE ON SCHEMA public TO analytics_admin;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO analytics_admin;
GRANT SELECT, REFRESH ON ALL MATERIALIZED VIEWS IN SCHEMA public TO analytics_admin;

-- Comments for documentation
COMMENT ON TABLE analytics_events IS 'Time-series table for all analytics events';
COMMENT ON TABLE business_metrics IS 'Business metrics and KPI values';
COMMENT ON TABLE kpi_definitions IS 'KPI definitions and thresholds';
COMMENT ON TABLE user_behavior_analytics IS 'User behavior and engagement analytics';
COMMENT ON TABLE property_analytics IS 'Property performance analytics';
COMMENT ON TABLE ml_model_performance IS 'ML model performance tracking';
COMMENT ON TABLE revenue_analytics IS 'Revenue and financial analytics';
COMMENT ON TABLE market_analytics IS 'Market analysis and trends';
COMMENT ON TABLE reports IS 'Report definitions and configurations';
COMMENT ON TABLE alerts IS 'Alert definitions and monitoring rules';
COMMENT ON TABLE data_lineage IS 'Data lineage and governance tracking';