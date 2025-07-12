"""
Rental ML System - Streamlit Demo Application

This is a comprehensive demo application showcasing all the features of the rental ML system:
- Property search with advanced filters
- ML-powered recommendation engine
- User preference management
- Data analytics and visualization
- System monitoring and performance metrics
- Interactive property comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import uuid
import json
import os
import sys

# Add the src directory to the path so we can import from the project
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.domain.entities.property import Property
from src.domain.entities.user import User, UserPreferences, UserInteraction
from sample_data import SampleDataGenerator
from components import PropertyCard, SearchFilters, RecommendationCard, MetricsDisplay
from utils import format_price, calculate_distance, generate_map_data

# Page configuration
st.set_page_config(
    page_title="Rental ML System Demo",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .property-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .property-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    .recommendation-score {
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        color: white;
        padding: 0.5rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
    }
    
    .filter-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-healthy { background-color: #28a745; }
    .status-warning { background-color: #ffc107; }
    .status-error { background-color: #dc3545; }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = SampleDataGenerator()
    st.session_state.properties = st.session_state.sample_data.generate_properties(100)
    st.session_state.users = st.session_state.sample_data.generate_users(50)
    st.session_state.interactions = st.session_state.sample_data.generate_interactions(
        st.session_state.users, st.session_state.properties, 500
    )
    st.session_state.current_user = st.session_state.users[0]
    st.session_state.selected_properties = []

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏠 Rental ML System Demo</h1>
        <p>Intelligent Property Search & Recommendation Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=Rental+ML", width=300)
        
        page = st.selectbox(
            "Navigate to:",
            [
                "🏠 Property Search",
                "🎯 Recommendations", 
                "👤 User Preferences",
                "📊 Analytics Dashboard",
                "⚡ ML Performance",
                "🔍 System Monitoring",
                "🆚 Property Comparison",
                "📈 Market Insights"
            ]
        )
        
        st.markdown("---")
        
        # User selection
        user_emails = [user.email for user in st.session_state.users[:10]]
        selected_email = st.selectbox("Current User:", user_emails)
        st.session_state.current_user = next(
            user for user in st.session_state.users if user.email == selected_email
        )
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### Quick Stats")
        st.metric("Total Properties", len(st.session_state.properties))
        st.metric("Active Users", len(st.session_state.users))
        st.metric("Total Interactions", len(st.session_state.interactions))
    
    # Route to selected page
    if page == "🏠 Property Search":
        show_property_search()
    elif page == "🎯 Recommendations":
        show_recommendations()
    elif page == "👤 User Preferences":
        show_user_preferences()
    elif page == "📊 Analytics Dashboard":
        show_analytics_dashboard()
    elif page == "⚡ ML Performance":
        show_ml_performance()
    elif page == "🔍 System Monitoring":
        show_system_monitoring()
    elif page == "🆚 Property Comparison":
        show_property_comparison()
    elif page == "📈 Market Insights":
        show_market_insights()

def show_property_search():
    """Property search interface with advanced filters"""
    st.header("🔍 Property Search")
    
    # Search and filter section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_input(
            "Search properties...", 
            placeholder="Enter location, amenities, or keywords"
        )
    
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    # Advanced filters
    with st.expander("🎛️ Advanced Filters", expanded=True):
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        
        with filter_col1:
            price_range = st.slider("Price Range ($)", 0, 10000, (1000, 5000), step=100)
            bedrooms = st.multiselect("Bedrooms", [1, 2, 3, 4, 5], default=[2, 3])
        
        with filter_col2:
            bathrooms = st.selectbox("Min Bathrooms", [1.0, 1.5, 2.0, 2.5, 3.0], index=1)
            property_types = st.multiselect(
                "Property Type", 
                ["apartment", "house", "condo", "studio", "townhouse"],
                default=["apartment", "house"]
            )
        
        with filter_col3:
            locations = list(set([prop.location for prop in st.session_state.properties]))
            selected_locations = st.multiselect("Locations", locations)
            square_feet_min = st.number_input("Min Sq Ft", min_value=0, value=500)
        
        with filter_col4:
            amenities = ["parking", "gym", "pool", "laundry", "pet-friendly", "balcony"]
            selected_amenities = st.multiselect("Required Amenities", amenities)
            sort_by = st.selectbox("Sort by", ["price", "bedrooms", "square_feet", "location"])
    
    # Filter properties
    filtered_properties = filter_properties(
        st.session_state.properties,
        price_range=price_range,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        property_types=property_types,
        locations=selected_locations,
        square_feet_min=square_feet_min,
        amenities=selected_amenities,
        search_query=search_query
    )
    
    # Sort properties
    if sort_by == "price":
        filtered_properties.sort(key=lambda x: x.price)
    elif sort_by == "bedrooms":
        filtered_properties.sort(key=lambda x: x.bedrooms, reverse=True)
    elif sort_by == "square_feet":
        filtered_properties.sort(key=lambda x: x.square_feet or 0, reverse=True)
    
    # Results header
    st.markdown(f"**Found {len(filtered_properties)} properties**")
    
    # Display properties
    if filtered_properties:
        for i in range(0, len(filtered_properties[:20]), 2):  # Show max 20 properties
            col1, col2 = st.columns(2)
            
            with col1:
                if i < len(filtered_properties):
                    display_property_card(filtered_properties[i])
            
            with col2:
                if i + 1 < len(filtered_properties):
                    display_property_card(filtered_properties[i + 1])
    else:
        st.info("No properties found matching your criteria. Try adjusting the filters.")

def show_recommendations():
    """ML-powered recommendation interface"""
    st.header("🎯 Smart Recommendations")
    
    user = st.session_state.current_user
    
    # Recommendation settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_recommendations = st.slider("Number of recommendations", 1, 20, 10)
    
    with col2:
        recommendation_type = st.selectbox(
            "Recommendation Type",
            ["Hybrid", "Collaborative Filtering", "Content-Based"]
        )
    
    with col3:
        diversity_level = st.slider("Diversity Level", 0.0, 1.0, 0.3, step=0.1)
    
    # Generate recommendations button
    if st.button("🔄 Generate Recommendations", use_container_width=True):
        with st.spinner("Generating personalized recommendations..."):
            time.sleep(2)  # Simulate ML processing
            recommendations = generate_sample_recommendations(
                user, st.session_state.properties, num_recommendations
            )
            st.session_state.recommendations = recommendations
    
    # Display recommendations
    if hasattr(st.session_state, 'recommendations'):
        st.subheader("🌟 Recommended for You")
        
        for i, rec in enumerate(st.session_state.recommendations):
            property_obj = next(
                (p for p in st.session_state.properties if p.id == rec['property_id']), None
            )
            
            if property_obj:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        display_property_card(property_obj, show_recommendation_score=True, 
                                            recommendation_score=rec['score'])
                    
                    with col2:
                        st.markdown("### Recommendation Details")
                        st.metric("Match Score", f"{rec['score']:.1%}")
                        st.markdown(f"**Reason:** {rec['explanation']}")
                        
                        # Action buttons
                        if st.button(f"💖 Like", key=f"like_{i}"):
                            add_user_interaction(user.id, property_obj.id, "like")
                            st.success("Added to liked properties!")
                        
                        if st.button(f"📞 Contact", key=f"contact_{i}"):
                            add_user_interaction(user.id, property_obj.id, "inquiry")
                            st.success("Contact request sent!")
                        
                        if st.button(f"💾 Save", key=f"save_{i}"):
                            add_user_interaction(user.id, property_obj.id, "save")
                            st.success("Property saved!")
                
                st.markdown("---")
    
    # Recommendation explanation
    with st.expander("🧠 How Recommendations Work"):
        st.markdown("""
        Our hybrid recommendation system combines multiple approaches:
        
        **1. Collaborative Filtering:**
        - Analyzes preferences of similar users
        - Finds patterns in user behavior
        - Recommends properties liked by similar users
        
        **2. Content-Based Filtering:**
        - Analyzes property features you've liked
        - Matches properties with similar characteristics
        - Considers location, price, amenities, etc.
        
        **3. Hybrid Approach:**
        - Combines both methods for better accuracy
        - Handles cold start problems for new users
        - Provides diverse and relevant recommendations
        """)

def show_user_preferences():
    """User preference configuration interface"""
    st.header("👤 User Preferences")
    
    user = st.session_state.current_user
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Current Profile")
        st.write(f"**Email:** {user.email}")
        st.write(f"**Member Since:** {user.created_at.strftime('%B %d, %Y')}")
        st.write(f"**Total Interactions:** {len(user.interactions)}")
        
        # Interaction history
        if user.interactions:
            interaction_df = pd.DataFrame([
                {
                    'Type': interaction.interaction_type,
                    'Date': interaction.timestamp.strftime('%Y-%m-%d'),
                    'Property ID': str(interaction.property_id)[:8] + '...'
                }
                for interaction in user.interactions[-10:]  # Last 10 interactions
            ])
            st.subheader("Recent Activity")
            st.dataframe(interaction_df, use_container_width=True)
    
    with col2:
        st.subheader("⚙️ Preference Settings")
        
        # Price preferences
        current_min = user.preferences.min_price or 0
        current_max = user.preferences.max_price or 10000
        
        price_range = st.slider(
            "Budget Range ($)", 
            0, 15000, 
            (current_min, current_max), 
            step=100
        )
        
        # Bedroom preferences
        bedroom_range = st.slider(
            "Bedrooms", 
            1, 6, 
            (user.preferences.min_bedrooms or 1, user.preferences.max_bedrooms or 3)
        )
        
        # Bathroom preferences
        bathroom_range = st.slider(
            "Bathrooms", 
            1.0, 4.0, 
            (user.preferences.min_bathrooms or 1.0, user.preferences.max_bathrooms or 2.0),
            step=0.5
        )
        
        # Location preferences
        all_locations = list(set([prop.location for prop in st.session_state.properties]))
        preferred_locations = st.multiselect(
            "Preferred Locations",
            all_locations,
            default=user.preferences.preferred_locations or []
        )
        
        # Amenity preferences
        all_amenities = ["parking", "gym", "pool", "laundry", "pet-friendly", "balcony", "ac", "heating"]
        required_amenities = st.multiselect(
            "Required Amenities",
            all_amenities,
            default=user.preferences.required_amenities or []
        )
        
        # Property type preferences
        property_types = st.multiselect(
            "Property Types",
            ["apartment", "house", "condo", "studio", "townhouse"],
            default=user.preferences.property_types or ["apartment"]
        )
        
        # Save preferences
        if st.button("💾 Save Preferences", use_container_width=True):
            new_preferences = UserPreferences(
                min_price=price_range[0],
                max_price=price_range[1],
                min_bedrooms=bedroom_range[0],
                max_bedrooms=bedroom_range[1],
                min_bathrooms=bathroom_range[0],
                max_bathrooms=bathroom_range[1],
                preferred_locations=preferred_locations,
                required_amenities=required_amenities,
                property_types=property_types
            )
            user.update_preferences(new_preferences)
            st.success("Preferences saved successfully!")
            time.sleep(1)
            st.rerun()

def show_analytics_dashboard():
    """Data visualization and analytics dashboard"""
    st.header("📊 Analytics Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    properties = st.session_state.properties
    users = st.session_state.users
    interactions = st.session_state.interactions
    
    with col1:
        st.metric("Total Properties", len(properties))
    
    with col2:
        active_properties = len([p for p in properties if p.is_active])
        st.metric("Active Properties", active_properties)
    
    with col3:
        avg_price = np.mean([p.price for p in properties])
        st.metric("Average Price", f"${avg_price:,.0f}")
    
    with col4:
        total_interactions = len(interactions)
        st.metric("Total Interactions", total_interactions)
    
    # Charts
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Analysis", "🏠 Property Distribution", "👥 User Activity", "📈 Trends"])
    
    with tab1:
        # Price distribution
        prices = [p.price for p in properties]
        fig_price = px.histogram(
            x=prices, 
            nbins=30, 
            title="Property Price Distribution",
            labels={'x': 'Price ($)', 'y': 'Count'}
        )
        fig_price.update_layout(height=400)
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Price by location
        price_location_data = []
        for prop in properties:
            price_location_data.append({'location': prop.location, 'price': prop.price})
        
        df_price_location = pd.DataFrame(price_location_data)
        fig_box = px.box(
            df_price_location, 
            x='location', 
            y='price', 
            title="Price Distribution by Location"
        )
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)
    
    with tab2:
        # Property type distribution
        property_types = [p.property_type for p in properties]
        type_counts = pd.Series(property_types).value_counts()
        
        fig_pie = px.pie(
            values=type_counts.values, 
            names=type_counts.index, 
            title="Property Type Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Bedrooms distribution
        bedrooms = [p.bedrooms for p in properties]
        bedroom_counts = pd.Series(bedrooms).value_counts().sort_index()
        
        fig_bar = px.bar(
            x=bedroom_counts.index, 
            y=bedroom_counts.values,
            title="Properties by Number of Bedrooms",
            labels={'x': 'Bedrooms', 'y': 'Count'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab3:
        # User interaction heatmap
        interaction_data = []
        for interaction in interactions:
            interaction_data.append({
                'hour': interaction.timestamp.hour,
                'day': interaction.timestamp.strftime('%A'),
                'type': interaction.interaction_type
            })
        
        df_interactions = pd.DataFrame(interaction_data)
        
        if not df_interactions.empty:
            # Interaction type distribution
            interaction_counts = df_interactions['type'].value_counts()
            fig_interaction = px.bar(
                x=interaction_counts.values,
                y=interaction_counts.index,
                orientation='h',
                title="User Interaction Types"
            )
            st.plotly_chart(fig_interaction, use_container_width=True)
            
            # Activity by hour
            hourly_activity = df_interactions.groupby('hour').size()
            fig_hourly = px.line(
                x=hourly_activity.index,
                y=hourly_activity.values,
                title="User Activity by Hour of Day",
                labels={'x': 'Hour', 'y': 'Interactions'}
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
    
    with tab4:
        # Simulate time series data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        
        # Property views trend
        views = np.random.poisson(100, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 20
        inquiries = views * 0.1 + np.random.normal(0, 2, len(dates))
        bookings = inquiries * 0.3 + np.random.normal(0, 1, len(dates))
        
        trends_df = pd.DataFrame({
            'date': dates,
            'views': views,
            'inquiries': inquiries,
            'bookings': bookings
        })
        
        fig_trends = px.line(
            trends_df, 
            x='date', 
            y=['views', 'inquiries', 'bookings'],
            title="Platform Activity Trends (2024)"
        )
        st.plotly_chart(fig_trends, use_container_width=True)

def show_ml_performance():
    """ML model performance metrics and monitoring"""
    st.header("⚡ ML Model Performance")
    
    # Model status overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤝 Collaborative Filtering</h3>
            <div style="display: flex; align-items: center;">
                <span class="status-indicator status-healthy"></span>
                <span>Healthy</span>
            </div>
            <p><strong>Accuracy:</strong> 87.3%</p>
            <p><strong>Last Training:</strong> 2 hours ago</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Content-Based</h3>
            <div style="display: flex; align-items: center;">
                <span class="status-indicator status-healthy"></span>
                <span>Healthy</span>
            </div>
            <p><strong>Accuracy:</strong> 82.1%</p>
            <p><strong>Last Training:</strong> 3 hours ago</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔗 Hybrid System</h3>
            <div style="display: flex; align-items: center;">
                <span class="status-indicator status-healthy"></span>
                <span>Healthy</span>
            </div>
            <p><strong>Accuracy:</strong> 91.7%</p>
            <p><strong>Last Update:</strong> 1 hour ago</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance metrics tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Accuracy Metrics", "⏱️ Performance", "🎯 Recommendation Quality", "🔄 Model Training"])
    
    with tab1:
        # Simulate accuracy metrics over time
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        np.random.seed(42)
        
        cf_accuracy = 0.85 + np.random.normal(0, 0.02, 30)
        cb_accuracy = 0.80 + np.random.normal(0, 0.015, 30)
        hybrid_accuracy = 0.90 + np.random.normal(0, 0.01, 30)
        
        accuracy_df = pd.DataFrame({
            'date': dates,
            'Collaborative Filtering': cf_accuracy,
            'Content-Based': cb_accuracy,
            'Hybrid': hybrid_accuracy
        })
        
        fig_accuracy = px.line(
            accuracy_df, 
            x='date', 
            y=['Collaborative Filtering', 'Content-Based', 'Hybrid'],
            title="Model Accuracy Over Time",
            labels={'value': 'Accuracy', 'variable': 'Model Type'}
        )
        fig_accuracy.update_layout(height=400)
        st.plotly_chart(fig_accuracy, use_container_width=True)
        
        # Confusion matrix simulation
        st.subheader("Recommendation Confusion Matrix")
        col1, col2 = st.columns(2)
        
        with col1:
            confusion_data = np.array([[85, 15], [12, 88]])
            fig_confusion = px.imshow(
                confusion_data,
                text_auto=True,
                title="Hybrid Model Confusion Matrix",
                labels={'x': 'Predicted', 'y': 'Actual'},
                x=['Not Relevant', 'Relevant'],
                y=['Not Relevant', 'Relevant']
            )
            st.plotly_chart(fig_confusion, use_container_width=True)
        
        with col2:
            st.markdown("### Performance Metrics")
            st.metric("Precision", "85.4%", "↑ 2.1%")
            st.metric("Recall", "88.0%", "↑ 1.7%")
            st.metric("F1-Score", "86.7%", "↑ 1.9%")
            st.metric("AUC-ROC", "0.912", "↑ 0.008")
    
    with tab2:
        # Response time metrics
        st.subheader("Response Time Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Simulate response times
            hours = list(range(24))
            response_times = [50 + np.random.normal(0, 10) + 20 * np.sin(h * np.pi / 12) for h in hours]
            
            fig_response = px.line(
                x=hours,
                y=response_times,
                title="Average Response Time by Hour",
                labels={'x': 'Hour of Day', 'y': 'Response Time (ms)'}
            )
            st.plotly_chart(fig_response, use_container_width=True)
        
        with col2:
            st.markdown("### Current Performance")
            st.metric("Avg Response Time", "47ms", "↓ 3ms")
            st.metric("95th Percentile", "89ms", "↓ 5ms")
            st.metric("99th Percentile", "156ms", "↓ 12ms")
            st.metric("Throughput", "2,847 req/min", "↑ 127 req/min")
        
        # Resource utilization
        st.subheader("Resource Utilization")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cpu_usage = 67
            fig_cpu = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cpu_usage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "CPU Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "red"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}))
            fig_cpu.update_layout(height=300)
            st.plotly_chart(fig_cpu, use_container_width=True)
        
        with col2:
            memory_usage = 45
            fig_memory = go.Figure(go.Indicator(
                mode="gauge+number",
                value=memory_usage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Memory Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkgreen"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "red"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}))
            fig_memory.update_layout(height=300)
            st.plotly_chart(fig_memory, use_container_width=True)
        
        with col3:
            gpu_usage = 23
            fig_gpu = go.Figure(go.Indicator(
                mode="gauge+number",
                value=gpu_usage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "GPU Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkred"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "red"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}))
            fig_gpu.update_layout(height=300)
            st.plotly_chart(fig_gpu, use_container_width=True)
    
    with tab3:
        st.subheader("Recommendation Quality Metrics")
        
        # Diversity and novelty metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Diversity Metrics")
            st.metric("Intra-list Diversity", "0.73", "↑ 0.05")
            st.metric("Coverage", "0.84", "↑ 0.02")
            st.metric("Novelty Score", "0.67", "↓ 0.01")
            
            # User satisfaction over time
            dates = pd.date_range(start='2024-11-01', periods=10, freq='D')
            satisfaction = np.random.uniform(4.2, 4.7, 10)
            
            fig_satisfaction = px.line(
                x=dates,
                y=satisfaction,
                title="User Satisfaction Score",
                labels={'x': 'Date', 'y': 'Rating (1-5)'}
            )
            fig_satisfaction.update_yaxis(range=[4.0, 5.0])
            st.plotly_chart(fig_satisfaction, use_container_width=True)
        
        with col2:
            st.markdown("### User Engagement")
            st.metric("Click-through Rate", "12.4%", "↑ 0.8%")
            st.metric("Conversion Rate", "3.7%", "↑ 0.3%")
            st.metric("User Retention", "78.9%", "↑ 2.1%")
            
            # Recommendation acceptance rate
            accept_rates = [0.65, 0.72, 0.68, 0.75, 0.71, 0.78, 0.73]
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            fig_accept = px.bar(
                x=days,
                y=accept_rates,
                title="Recommendation Acceptance Rate by Day",
                labels={'x': 'Day of Week', 'y': 'Acceptance Rate'}
            )
            st.plotly_chart(fig_accept, use_container_width=True)
    
    with tab4:
        st.subheader("Model Training Status")
        
        # Training progress
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Last Training Session")
            st.write("**Model:** Hybrid Recommender v2.1")
            st.write("**Started:** 2024-01-15 14:30:00")
            st.write("**Duration:** 2h 15m")
            st.write("**Status:** ✅ Completed Successfully")
            
            # Training loss over epochs
            epochs = list(range(1, 101))
            training_loss = [1.2 * np.exp(-0.05 * e) + 0.1 + np.random.normal(0, 0.02) for e in epochs]
            validation_loss = [1.1 * np.exp(-0.04 * e) + 0.12 + np.random.normal(0, 0.02) for e in epochs]
            
            fig_loss = px.line(
                x=epochs,
                y=[training_loss, validation_loss],
                title="Training Progress",
                labels={'x': 'Epoch', 'y': 'Loss'}
            )
            fig_loss.data[0].name = 'Training Loss'
            fig_loss.data[1].name = 'Validation Loss'
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with col2:
            st.markdown("### Training Schedule")
            st.write("**Next Training:** 2024-01-16 02:00:00")
            st.write("**Frequency:** Daily (02:00 UTC)")
            st.write("**Auto-deploy:** Enabled")
            st.write("**Rollback threshold:** 2% accuracy drop")
            
            # Model versions
            st.markdown("### Model Version History")
            version_data = {
                'Version': ['v2.1', 'v2.0', 'v1.9', 'v1.8'],
                'Accuracy': ['91.7%', '90.1%', '89.3%', '88.7%'],
                'Deploy Date': ['2024-01-15', '2024-01-10', '2024-01-05', '2024-01-01'],
                'Status': ['🟢 Active', '🟡 Backup', '🔴 Retired', '🔴 Retired']
            }
            st.dataframe(pd.DataFrame(version_data), use_container_width=True)

def show_system_monitoring():
    """System health and monitoring dashboard"""
    st.header("🔍 System Monitoring")
    
    # System health overview
    st.subheader("🏥 System Health Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #d4edda; border-radius: 10px;">
            <h3 style="color: #155724;">🟢 API Gateway</h3>
            <p style="color: #155724; margin: 0;"><strong>Status:</strong> Healthy</p>
            <p style="color: #155724; margin: 0;"><strong>Uptime:</strong> 99.97%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #d4edda; border-radius: 10px;">
            <h3 style="color: #155724;">🟢 Database</h3>
            <p style="color: #155724; margin: 0;"><strong>Status:</strong> Healthy</p>
            <p style="color: #155724; margin: 0;"><strong>Connections:</strong> 45/100</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #fff3cd; border-radius: 10px;">
            <h3 style="color: #856404;">🟡 ML Models</h3>
            <p style="color: #856404; margin: 0;"><strong>Status:</strong> Warning</p>
            <p style="color: #856404; margin: 0;"><strong>Load:</strong> High</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #d4edda; border-radius: 10px;">
            <h3 style="color: #155724;">🟢 Scrapers</h3>
            <p style="color: #155724; margin: 0;"><strong>Status:</strong> Healthy</p>
            <p style="color: #155724; margin: 0;"><strong>Success Rate:</strong> 98.5%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed monitoring tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📡 API Monitoring", "🗄️ Database Health", "🕷️ Scraping Status", "🚨 Alerts & Logs"])
    
    with tab1:
        st.subheader("API Endpoint Monitoring")
        
        # API metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Response time by endpoint
            endpoints = ['/properties', '/search', '/recommendations', '/users']
            response_times = [45, 67, 123, 34]
            
            fig_api = px.bar(
                x=endpoints,
                y=response_times,
                title="Average Response Time by Endpoint",
                labels={'x': 'Endpoint', 'y': 'Response Time (ms)'}
            )
            st.plotly_chart(fig_api, use_container_width=True)
        
        with col2:
            # Request volume over time
            hours = list(range(24))
            requests = [100 + 50 * np.sin(h * np.pi / 12) + np.random.normal(0, 10) for h in hours]
            
            fig_requests = px.line(
                x=hours,
                y=requests,
                title="Request Volume by Hour",
                labels={'x': 'Hour', 'y': 'Requests/min'}
            )
            st.plotly_chart(fig_requests, use_container_width=True)
        
        # API status table
        st.subheader("Endpoint Status")
        api_status = {
            'Endpoint': ['/properties', '/search', '/recommendations', '/users', '/health'],
            'Status': ['🟢 Healthy', '🟢 Healthy', '🟡 Slow', '🟢 Healthy', '🟢 Healthy'],
            'Response Time (ms)': [45, 67, 123, 34, 12],
            'Success Rate (%)': [99.8, 99.5, 97.2, 99.9, 100.0],
            'Last Check': ['1 min ago', '1 min ago', '1 min ago', '1 min ago', '30 sec ago']
        }
        st.dataframe(pd.DataFrame(api_status), use_container_width=True)
    
    with tab2:
        st.subheader("Database Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Active Connections", "45", "↑ 3")
            st.metric("Query Performance", "23ms avg", "↓ 2ms")
            st.metric("Cache Hit Rate", "94.7%", "↑ 1.2%")
            st.metric("Storage Used", "2.3 TB", "↑ 45 GB")
        
        with col2:
            # Database query performance
            query_types = ['SELECT', 'INSERT', 'UPDATE', 'DELETE']
            query_times = [15, 45, 32, 28]
            
            fig_queries = px.bar(
                x=query_types,
                y=query_times,
                title="Average Query Time by Type",
                labels={'x': 'Query Type', 'y': 'Average Time (ms)'}
            )
            st.plotly_chart(fig_queries, use_container_width=True)
        
        # Connection pool status
        st.subheader("Connection Pool Status")
        pool_data = {
            'Pool': ['Main', 'Read Replica 1', 'Read Replica 2', 'Analytics'],
            'Active': [23, 12, 8, 2],
            'Idle': [7, 8, 12, 18],
            'Max': [50, 30, 30, 25],
            'Status': ['🟢 Healthy', '🟢 Healthy', '🟢 Healthy', '🟢 Healthy']
        }
        st.dataframe(pd.DataFrame(pool_data), use_container_width=True)
    
    with tab3:
        st.subheader("Web Scraping Status")
        
        # Scraper performance
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Active Scrapers", "5", "→ 0")
            st.metric("Success Rate", "98.5%", "↑ 0.3%")
            st.metric("Properties Scraped Today", "1,247", "↑ 83")
            st.metric("Failed Requests", "18", "↓ 5")
        
        with col2:
            # Scraping success rate by source
            sources = ['Apartments.com', 'Rent.com', 'Zillow', 'Craigslist']
            success_rates = [98.5, 97.2, 99.1, 94.8]
            
            fig_scrapers = px.bar(
                x=sources,
                y=success_rates,
                title="Scraping Success Rate by Source",
                labels={'x': 'Source', 'y': 'Success Rate (%)'}
            )
            fig_scrapers.update_yaxis(range=[90, 100])
            st.plotly_chart(fig_scrapers, use_container_width=True)
        
        # Scraper status table
        st.subheader("Scraper Status")
        scraper_status = {
            'Source': ['Apartments.com', 'Rent.com', 'Zillow', 'Craigslist', 'PadMapper'],
            'Status': ['🟢 Running', '🟢 Running', '🟡 Rate Limited', '🟢 Running', '🔴 Error'],
            'Last Run': ['2 min ago', '5 min ago', '1 hour ago', '3 min ago', '30 min ago'],
            'Properties Found': [234, 189, 0, 156, 0],
            'Errors': [0, 2, 1, 0, 5],
            'Next Run': ['3 min', '10 min', '2 hours', '7 min', 'Manual']
        }
        st.dataframe(pd.DataFrame(scraper_status), use_container_width=True)
    
    with tab4:
        st.subheader("System Alerts & Logs")
        
        # Recent alerts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚨 Recent Alerts")
            alerts = [
                {"time": "14:32", "level": "WARNING", "message": "High ML model response time detected"},
                {"time": "14:15", "level": "INFO", "message": "Scraper rate limit reached for Zillow"},
                {"time": "13:45", "level": "ERROR", "message": "PadMapper scraper connection failed"},
                {"time": "13:20", "level": "INFO", "message": "Database maintenance completed"},
                {"time": "12:55", "level": "WARNING", "message": "High memory usage on ML server"}
            ]
            
            for alert in alerts:
                level_color = {"ERROR": "🔴", "WARNING": "🟡", "INFO": "🔵"}
                st.markdown(f"{level_color[alert['level']]} **{alert['time']}** - {alert['message']}")
        
        with col2:
            st.markdown("### 📊 Alert Statistics")
            st.metric("Alerts Today", "12", "↑ 3")
            st.metric("Critical Alerts", "0", "→ 0")
            st.metric("Average Resolution Time", "4.2 min", "↓ 30s")
            
            # Alert distribution
            alert_types = ['Info', 'Warning', 'Error', 'Critical']
            alert_counts = [8, 3, 1, 0]
            
            fig_alerts = px.pie(
                values=alert_counts,
                names=alert_types,
                title="Alert Distribution (Last 24h)"
            )
            st.plotly_chart(fig_alerts, use_container_width=True)
        
        # System logs
        st.subheader("📋 Recent System Logs")
        logs = [
            {"timestamp": "2024-01-15 14:32:15", "level": "WARN", "service": "ML-Service", "message": "Response time exceeded threshold: 150ms"},
            {"timestamp": "2024-01-15 14:30:42", "level": "INFO", "service": "API-Gateway", "message": "Health check completed successfully"},
            {"timestamp": "2024-01-15 14:28:33", "level": "INFO", "service": "Scraper", "message": "Apartments.com scraping completed: 234 properties"},
            {"timestamp": "2024-01-15 14:25:11", "level": "ERROR", "service": "Scraper", "message": "PadMapper connection timeout after 30s"},
            {"timestamp": "2024-01-15 14:22:56", "level": "INFO", "service": "Database", "message": "Query optimization completed"}
        ]
        
        log_df = pd.DataFrame(logs)
        st.dataframe(log_df, use_container_width=True)

def show_property_comparison():
    """Property comparison tool"""
    st.header("🆚 Property Comparison")
    
    # Property selection
    st.subheader("Select Properties to Compare")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Property 1 selection
        property_options = [f"{p.title} - ${p.price:,}" for p in st.session_state.properties[:20]]
        selected_1 = st.selectbox("Property 1:", property_options, key="prop1")
        property_1 = st.session_state.properties[property_options.index(selected_1)]
    
    with col2:
        # Property 2 selection
        selected_2 = st.selectbox("Property 2:", property_options, key="prop2", index=1)
        property_2 = st.session_state.properties[property_options.index(selected_2)]
    
    # Comparison table
    st.subheader("📊 Side-by-Side Comparison")
    
    comparison_data = {
        'Feature': [
            'Price', 'Price per Sq Ft', 'Bedrooms', 'Bathrooms', 
            'Square Feet', 'Location', 'Property Type', 'Amenities Count'
        ],
        property_1.title: [
            f"${property_1.price:,}",
            f"${property_1.get_price_per_sqft():.2f}" if property_1.get_price_per_sqft() else "N/A",
            property_1.bedrooms,
            property_1.bathrooms,
            f"{property_1.square_feet:,}" if property_1.square_feet else "N/A",
            property_1.location,
            property_1.property_type,
            len(property_1.amenities)
        ],
        property_2.title: [
            f"${property_2.price:,}",
            f"${property_2.get_price_per_sqft():.2f}" if property_2.get_price_per_sqft() else "N/A",
            property_2.bedrooms,
            property_2.bathrooms,
            f"{property_2.square_feet:,}" if property_2.square_feet else "N/A",
            property_2.location,
            property_2.property_type,
            len(property_2.amenities)
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visual comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Price comparison
        fig_price = go.Figure(data=[
            go.Bar(name=property_1.title[:20], x=['Price'], y=[property_1.price]),
            go.Bar(name=property_2.title[:20], x=['Price'], y=[property_2.price])
        ])
        fig_price.update_layout(title="Price Comparison", yaxis_title="Price ($)")
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        # Features comparison radar chart
        features = ['Bedrooms', 'Bathrooms', 'Amenities', 'Size']
        prop1_values = [
            property_1.bedrooms / 5 * 100,  # Normalize to 0-100
            property_1.bathrooms / 4 * 100,
            len(property_1.amenities) / 10 * 100,
            (property_1.square_feet or 1000) / 3000 * 100
        ]
        prop2_values = [
            property_2.bedrooms / 5 * 100,
            property_2.bathrooms / 4 * 100,
            len(property_2.amenities) / 10 * 100,
            (property_2.square_feet or 1000) / 3000 * 100
        ]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=prop1_values,
            theta=features,
            fill='toself',
            name=property_1.title[:20]
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=prop2_values,
            theta=features,
            fill='toself',
            name=property_2.title[:20]
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Feature Comparison"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Amenities comparison
    st.subheader("🏠 Amenities Comparison")
    
    all_amenities = list(set(property_1.amenities + property_2.amenities))
    
    amenity_comparison = []
    for amenity in all_amenities:
        amenity_comparison.append({
            'Amenity': amenity,
            property_1.title: '✅' if amenity in property_1.amenities else '❌',
            property_2.title: '✅' if amenity in property_2.amenities else '❌'
        })
    
    amenity_df = pd.DataFrame(amenity_comparison)
    st.dataframe(amenity_df, use_container_width=True)

def show_market_insights():
    """Market analysis and insights dashboard"""
    st.header("📈 Market Insights")
    
    # Market overview
    col1, col2, col3, col4 = st.columns(4)
    
    properties = st.session_state.properties
    
    with col1:
        avg_price = np.mean([p.price for p in properties])
        st.metric("Market Average Price", f"${avg_price:,.0f}", "↑ 3.2%")
    
    with col2:
        median_price = np.median([p.price for p in properties])
        st.metric("Median Price", f"${median_price:,.0f}", "↑ 2.8%")
    
    with col3:
        price_per_sqft = np.mean([p.get_price_per_sqft() for p in properties if p.get_price_per_sqft()])
        st.metric("Avg Price/Sq Ft", f"${price_per_sqft:.2f}", "↑ 1.5%")
    
    with col4:
        inventory = len([p for p in properties if p.is_active])
        st.metric("Active Inventory", inventory, "↓ 23")
    
    # Market analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏘️ Neighborhood Analysis", "💰 Price Trends", "📊 Market Segments", "🔮 Predictions"])
    
    with tab1:
        st.subheader("Neighborhood Market Analysis")
        
        # Location-based analysis
        location_data = []
        for prop in properties:
            location_data.append({
                'location': prop.location,
                'price': prop.price,
                'bedrooms': prop.bedrooms,
                'property_type': prop.property_type
            })
        
        location_df = pd.DataFrame(location_data)
        location_stats = location_df.groupby('location').agg({
            'price': ['mean', 'median', 'count'],
            'bedrooms': 'mean'
        }).round(2)
        
        # Flatten column names
        location_stats.columns = ['Avg Price', 'Median Price', 'Properties', 'Avg Bedrooms']
        location_stats = location_stats.reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_location_price = px.bar(
                location_stats,
                x='location',
                y='Avg Price',
                title="Average Price by Neighborhood",
                labels={'location': 'Neighborhood', 'Avg Price': 'Average Price ($)'}
            )
            st.plotly_chart(fig_location_price, use_container_width=True)
        
        with col2:
            fig_location_count = px.bar(
                location_stats,
                x='location',
                y='Properties',
                title="Property Count by Neighborhood",
                labels={'location': 'Neighborhood', 'Properties': 'Number of Properties'}
            )
            st.plotly_chart(fig_location_count, use_container_width=True)
        
        st.subheader("Neighborhood Statistics")
        st.dataframe(location_stats, use_container_width=True)
    
    with tab2:
        st.subheader("Price Trend Analysis")
        
        # Simulate historical price data
        dates = pd.date_range(start='2023-01-01', end='2024-01-15', freq='W')
        np.random.seed(42)
        
        base_price = 2500
        trend = np.linspace(0, 200, len(dates))  # Upward trend
        seasonal = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)  # Seasonal variation
        noise = np.random.normal(0, 50, len(dates))
        
        prices = base_price + trend + seasonal + noise
        
        price_trend_df = pd.DataFrame({
            'date': dates,
            'avg_price': prices,
            'lower_bound': prices - 100,
            'upper_bound': prices + 100
        })
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=price_trend_df['date'],
            y=price_trend_df['avg_price'],
            mode='lines',
            name='Average Price',
            line=dict(color='blue')
        ))
        fig_trend.add_trace(go.Scatter(
            x=price_trend_df['date'],
            y=price_trend_df['upper_bound'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig_trend.add_trace(go.Scatter(
            x=price_trend_df['date'],
            y=price_trend_df['lower_bound'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            name='Price Range',
            fillcolor='rgba(0,0,255,0.2)'
        ))
        
        fig_trend.update_layout(
            title="Market Price Trends (2023-2024)",
            xaxis_title="Date",
            yaxis_title="Average Price ($)"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Price change analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("1 Month Change", "+3.2%", "↑")
            st.metric("3 Month Change", "+8.7%", "↑")
        
        with col2:
            st.metric("6 Month Change", "+12.4%", "↑")
            st.metric("1 Year Change", "+15.8%", "↑")
        
        with col3:
            st.metric("Market Velocity", "Fast", "🔥")
            st.metric("Days on Market", "18 days", "↓ 3")
    
    with tab3:
        st.subheader("Market Segmentation Analysis")
        
        # Property type analysis
        type_data = []
        for prop in properties:
            type_data.append({
                'property_type': prop.property_type,
                'price': prop.price,
                'bedrooms': prop.bedrooms
            })
        
        type_df = pd.DataFrame(type_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Property type distribution
            type_counts = type_df['property_type'].value_counts()
            fig_type_dist = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Property Type Distribution"
            )
            st.plotly_chart(fig_type_dist, use_container_width=True)
        
        with col2:
            # Price by property type
            fig_type_price = px.box(
                type_df,
                x='property_type',
                y='price',
                title="Price Distribution by Property Type"
            )
            st.plotly_chart(fig_type_price, use_container_width=True)
        
        # Bedroom segment analysis
        bedroom_counts = type_df['bedrooms'].value_counts().sort_index()
        bedroom_avg_prices = type_df.groupby('bedrooms')['price'].mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_bedroom_count = px.bar(
                x=bedroom_counts.index,
                y=bedroom_counts.values,
                title="Properties by Bedroom Count",
                labels={'x': 'Bedrooms', 'y': 'Number of Properties'}
            )
            st.plotly_chart(fig_bedroom_count, use_container_width=True)
        
        with col2:
            fig_bedroom_price = px.bar(
                x=bedroom_avg_prices.index,
                y=bedroom_avg_prices.values,
                title="Average Price by Bedroom Count",
                labels={'x': 'Bedrooms', 'y': 'Average Price ($)'}
            )
            st.plotly_chart(fig_bedroom_price, use_container_width=True)
    
    with tab4:
        st.subheader("Market Predictions & Forecasts")
        
        # Price prediction
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Price Forecast (Next 6 Months)")
            
            # Generate forecast data
            future_dates = pd.date_range(start='2024-01-16', periods=26, freq='W')
            np.random.seed(42)
            
            forecast_prices = []
            last_price = 2700
            for i in range(26):
                change = np.random.normal(1.005, 0.01)  # Small upward trend with volatility
                last_price *= change
                forecast_prices.append(last_price)
            
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'predicted_price': forecast_prices,
                'confidence_lower': [p * 0.95 for p in forecast_prices],
                'confidence_upper': [p * 1.05 for p in forecast_prices]
            })
            
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['predicted_price'],
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='red')
            ))
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['confidence_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['confidence_lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                name='Confidence Interval',
                fillcolor='rgba(255,0,0,0.2)'
            ))
            
            fig_forecast.update_layout(
                title="6-Month Price Forecast",
                xaxis_title="Date",
                yaxis_title="Predicted Price ($)"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Market Predictions")
            
            st.metric("3-Month Outlook", "Bullish 📈", "Strong demand")
            st.metric("6-Month Outlook", "Moderate 📊", "Stable growth")
            st.metric("Predicted Price Change", "+7.3%", "Next 6 months")
            
            st.markdown("### 🔍 Key Factors")
            st.markdown("""
            - **Seasonal Demand**: Spring season typically shows increased activity
            - **Economic Indicators**: Low unemployment supporting rental demand
            - **Supply Constraints**: Limited new construction in urban areas
            - **Interest Rates**: Current rates favoring rental market
            - **Population Growth**: Steady influx of new residents
            """)
            
            st.markdown("### 📊 Risk Assessment")
            risk_factors = {
                'Factor': ['Economic Downturn', 'Interest Rate Changes', 'Oversupply Risk', 'Regulatory Changes'],
                'Probability': ['Low', 'Medium', 'Low', 'Medium'],
                'Impact': ['High', 'Medium', 'High', 'Medium']
            }
            st.dataframe(pd.DataFrame(risk_factors), use_container_width=True)

# Helper functions
def filter_properties(properties, **filters):
    """Filter properties based on search criteria"""
    filtered = properties.copy()
    
    # Price range filter
    if 'price_range' in filters:
        min_price, max_price = filters['price_range']
        filtered = [p for p in filtered if min_price <= p.price <= max_price]
    
    # Bedrooms filter
    if 'bedrooms' in filters and filters['bedrooms']:
        filtered = [p for p in filtered if p.bedrooms in filters['bedrooms']]
    
    # Bathrooms filter
    if 'bathrooms' in filters:
        filtered = [p for p in filtered if p.bathrooms >= filters['bathrooms']]
    
    # Property type filter
    if 'property_types' in filters and filters['property_types']:
        filtered = [p for p in filtered if p.property_type in filters['property_types']]
    
    # Location filter
    if 'locations' in filters and filters['locations']:
        filtered = [p for p in filtered if p.location in filters['locations']]
    
    # Square feet filter
    if 'square_feet_min' in filters:
        filtered = [p for p in filtered if p.square_feet and p.square_feet >= filters['square_feet_min']]
    
    # Amenities filter
    if 'amenities' in filters and filters['amenities']:
        for amenity in filters['amenities']:
            filtered = [p for p in filtered if amenity in p.amenities]
    
    # Search query filter
    if 'search_query' in filters and filters['search_query']:
        query = filters['search_query'].lower()
        filtered = [p for p in filtered if 
                   query in p.title.lower() or 
                   query in p.description.lower() or 
                   query in p.location.lower() or
                   any(query in amenity.lower() for amenity in p.amenities)]
    
    return filtered

def display_property_card(property_obj, show_recommendation_score=False, recommendation_score=None):
    """Display a property card"""
    price_per_sqft = property_obj.get_price_per_sqft()
    
    with st.container():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Property image placeholder
            st.image("https://via.placeholder.com/300x200/667eea/ffffff?text=Property+Image", width=300)
            
            if show_recommendation_score and recommendation_score:
                st.markdown(f"""
                <div class="recommendation-score">
                    Match: {recommendation_score:.1%}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"### {property_obj.title}")
            st.markdown(f"**📍 Location:** {property_obj.location}")
            st.markdown(f"**💰 Price:** ${property_obj.price:,}/month")
            
            col_details1, col_details2 = st.columns(2)
            
            with col_details1:
                st.markdown(f"**🛏️ Bedrooms:** {property_obj.bedrooms}")
                st.markdown(f"**🚿 Bathrooms:** {property_obj.bathrooms}")
            
            with col_details2:
                st.markdown(f"**📐 Sq Ft:** {property_obj.square_feet:,}" if property_obj.square_feet else "**📐 Sq Ft:** N/A")
                st.markdown(f"**💵 Price/Sq Ft:** ${price_per_sqft:.2f}" if price_per_sqft else "**💵 Price/Sq Ft:** N/A")
            
            # Amenities
            if property_obj.amenities:
                amenities_text = ", ".join(property_obj.amenities[:5])
                if len(property_obj.amenities) > 5:
                    amenities_text += f" (+{len(property_obj.amenities) - 5} more)"
                st.markdown(f"**🏠 Amenities:** {amenities_text}")
            
            # Description (truncated)
            description = property_obj.description[:150] + "..." if len(property_obj.description) > 150 else property_obj.description
            st.markdown(f"**📝 Description:** {description}")

def generate_sample_recommendations(user, properties, num_recommendations):
    """Generate sample recommendations for demo"""
    # Simple recommendation logic for demo purposes
    recommendations = []
    
    user_preferences = user.preferences
    scored_properties = []
    
    for prop in properties:
        score = 0.5  # Base score
        
        # Price matching
        if user_preferences.min_price and user_preferences.max_price:
            if user_preferences.min_price <= prop.price <= user_preferences.max_price:
                score += 0.3
        
        # Location preference
        if user_preferences.preferred_locations and prop.location in user_preferences.preferred_locations:
            score += 0.2
        
        # Bedroom preference
        if user_preferences.min_bedrooms and user_preferences.max_bedrooms:
            if user_preferences.min_bedrooms <= prop.bedrooms <= user_preferences.max_bedrooms:
                score += 0.2
        
        # Amenity matching
        if user_preferences.required_amenities:
            matching_amenities = set(prop.amenities) & set(user_preferences.required_amenities)
            score += len(matching_amenities) * 0.1
        
        # Add some randomness
        score += np.random.normal(0, 0.1)
        score = max(0, min(1, score))  # Clamp to [0, 1]
        
        scored_properties.append((prop, score))
    
    # Sort by score and take top recommendations
    scored_properties.sort(key=lambda x: x[1], reverse=True)
    
    for i, (prop, score) in enumerate(scored_properties[:num_recommendations]):
        explanation = generate_recommendation_explanation(user, prop, score)
        recommendations.append({
            'property_id': prop.id,
            'score': score,
            'explanation': explanation
        })
    
    return recommendations

def generate_recommendation_explanation(user, property_obj, score):
    """Generate explanation for why a property was recommended"""
    reasons = []
    
    user_prefs = user.preferences
    
    if user_prefs.preferred_locations and property_obj.location in user_prefs.preferred_locations:
        reasons.append(f"Located in your preferred area ({property_obj.location})")
    
    if user_prefs.min_price and user_prefs.max_price:
        if user_prefs.min_price <= property_obj.price <= user_prefs.max_price:
            reasons.append("Price matches your budget")
    
    if user_prefs.required_amenities:
        matching_amenities = set(property_obj.amenities) & set(user_prefs.required_amenities)
        if matching_amenities:
            reasons.append(f"Has {len(matching_amenities)} of your required amenities")
    
    if not reasons:
        reasons.append("Similar to properties you've viewed")
    
    return "; ".join(reasons)

def add_user_interaction(user_id, property_id, interaction_type):
    """Add a user interaction to the session state"""
    interaction = UserInteraction.create(property_id, interaction_type)
    
    # Find the user and add the interaction
    for user in st.session_state.users:
        if user.id == user_id:
            user.add_interaction(interaction)
            break
    
    # Also add to global interactions list
    st.session_state.interactions.append(interaction)

if __name__ == "__main__":
    main()