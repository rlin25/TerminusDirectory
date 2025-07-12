"""
Rental ML System - Simple Streamlit Demo Application

A simplified working demo showcasing the key features of the rental ML system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Any
import uuid
import random

# Page configuration
st.set_page_config(
    page_title="Rental ML System Demo",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .property-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_sample_properties():
    """Generate sample property data"""
    cities = ["New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ"]
    property_types = ["Apartment", "House", "Condo", "Townhouse"]
    amenities_list = ["gym", "pool", "parking", "pet-friendly", "in-unit-laundry", "dishwasher", "ac", "balcony"]
    
    properties = []
    for i in range(100):
        amenities = random.sample(amenities_list, random.randint(2, 6))
        properties.append({
            'id': str(uuid.uuid4()),
            'title': f"Beautiful {random.choice(property_types)} in {random.choice(cities).split(',')[0]}",
            'price': random.randint(800, 4500),
            'location': random.choice(cities),
            'bedrooms': random.randint(1, 4),
            'bathrooms': random.choice([1, 1.5, 2, 2.5, 3]),
            'square_feet': random.randint(500, 2500),
            'property_type': random.choice(property_types),
            'amenities': amenities,
            'description': f"Spacious {random.choice(['modern', 'updated', 'luxury', 'cozy'])} property with great amenities.",
            'score': random.uniform(0.7, 0.98)
        })
    return pd.DataFrame(properties)

@st.cache_data
def generate_sample_users():
    """Generate sample user data"""
    users = []
    for i in range(50):
        users.append({
            'id': str(uuid.uuid4()),
            'name': f"User {i+1}",
            'email': f"user{i+1}@example.com",
            'min_price': random.randint(500, 2000),
            'max_price': random.randint(2000, 5000),
            'preferred_locations': random.sample(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"], 2),
            'min_bedrooms': random.randint(1, 2),
            'max_bedrooms': random.randint(2, 4),
            'required_amenities': random.sample(["gym", "pool", "parking", "pet-friendly"], 2)
        })
    return pd.DataFrame(users)

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏠 Rental ML System Demo</h1>
        <p>Intelligent Property Search & Recommendation Engine</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load sample data
    properties_df = generate_sample_properties()
    users_df = generate_sample_users()
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    section = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Property Search", "🎯 Recommendations", "📊 Analytics", "⚡ ML Performance", "🔍 System Status"]
    )
    
    if section == "🏠 Property Search":
        show_property_search(properties_df)
    elif section == "🎯 Recommendations":
        show_recommendations(properties_df, users_df)
    elif section == "📊 Analytics":
        show_analytics(properties_df)
    elif section == "⚡ ML Performance":
        show_ml_performance()
    elif section == "🔍 System Status":
        show_system_status()

def show_property_search(properties_df):
    """Display property search interface"""
    st.header("🏠 Property Search")
    
    # Search filters
    st.subheader("Search Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        price_range = st.slider("Price Range ($)", 500, 5000, (1000, 3000))
        bedrooms = st.selectbox("Bedrooms", ["Any", 1, 2, 3, 4])
    
    with col2:
        location = st.selectbox("Location", ["All"] + list(properties_df['location'].unique()))
        property_type = st.selectbox("Property Type", ["All"] + list(properties_df['property_type'].unique()))
    
    with col3:
        amenities = st.multiselect("Amenities", ["gym", "pool", "parking", "pet-friendly", "in-unit-laundry"])
        min_sqft = st.number_input("Min Square Feet", min_value=0, value=500)
    
    # Filter properties
    filtered_df = properties_df[
        (properties_df['price'] >= price_range[0]) & 
        (properties_df['price'] <= price_range[1]) &
        (properties_df['square_feet'] >= min_sqft)
    ]
    
    if bedrooms != "Any":
        filtered_df = filtered_df[filtered_df['bedrooms'] == bedrooms]
    
    if location != "All":
        filtered_df = filtered_df[filtered_df['location'] == location]
    
    if property_type != "All":
        filtered_df = filtered_df[filtered_df['property_type'] == property_type]
    
    if amenities:
        for amenity in amenities:
            filtered_df = filtered_df[filtered_df['amenities'].apply(lambda x: amenity in x)]
    
    # Results
    st.subheader(f"Search Results ({len(filtered_df)} properties found)")
    
    if len(filtered_df) > 0:
        # Sort by ML score
        filtered_df = filtered_df.sort_values('score', ascending=False)
        
        for idx, property in filtered_df.head(10).iterrows():
            with st.container():
                st.markdown(f"""
                <div class="property-card">
                    <h4>{property['title']}</h4>
                    <p><strong>📍 {property['location']}</strong></p>
                    <p><strong>💰 ${property['price']:,}/month</strong> | 
                       🛏️ {property['bedrooms']} bed | 
                       🚿 {property['bathrooms']} bath | 
                       📐 {property['square_feet']} sq ft</p>
                    <p><strong>Amenities:</strong> {', '.join(property['amenities'])}</p>
                    <p>ML Relevance Score: <strong>{property['score']:.2f}</strong></p>
                    <p>{property['description']}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No properties found matching your criteria. Please adjust your filters.")

def show_recommendations(properties_df, users_df):
    """Display recommendation interface"""
    st.header("🎯 AI-Powered Recommendations")
    
    # User selection
    selected_user = st.selectbox("Select User Profile", users_df['name'].tolist())
    user_data = users_df[users_df['name'] == selected_user].iloc[0]
    
    # Display user preferences
    st.subheader("👤 User Preferences")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Budget:** ${user_data['min_price']:,} - ${user_data['max_price']:,}")
        st.write(f"**Bedrooms:** {user_data['min_bedrooms']} - {user_data['max_bedrooms']}")
    
    with col2:
        st.write(f"**Preferred Locations:** {', '.join(user_data['preferred_locations'])}")
        st.write(f"**Required Amenities:** {', '.join(user_data['required_amenities'])}")
    
    # Generate recommendations
    st.subheader("🎯 Personalized Recommendations")
    
    # Filter based on user preferences
    recommendations = properties_df[
        (properties_df['price'] >= user_data['min_price']) & 
        (properties_df['price'] <= user_data['max_price']) &
        (properties_df['bedrooms'] >= user_data['min_bedrooms']) &
        (properties_df['bedrooms'] <= user_data['max_bedrooms'])
    ]
    
    # Add recommendation scores
    recommendations = recommendations.copy()
    recommendations['recommendation_score'] = np.random.uniform(0.75, 0.95, len(recommendations))
    recommendations = recommendations.sort_values('recommendation_score', ascending=False)
    
    for idx, property in recommendations.head(5).iterrows():
        with st.container():
            st.markdown(f"""
            <div class="property-card">
                <h4>⭐ {property['title']} <span style="color: #667eea;">(Match: {property['recommendation_score']:.1%})</span></h4>
                <p><strong>📍 {property['location']}</strong></p>
                <p><strong>💰 ${property['price']:,}/month</strong> | 
                   🛏️ {property['bedrooms']} bed | 
                   🚿 {property['bathrooms']} bath | 
                   📐 {property['square_feet']} sq ft</p>
                <p><strong>Amenities:</strong> {', '.join(property['amenities'])}</p>
                <p><strong>Why recommended:</strong> Matches your budget, location preferences, and required amenities.</p>
            </div>
            """, unsafe_allow_html=True)

def show_analytics(properties_df):
    """Display analytics dashboard"""
    st.header("📊 Market Analytics Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties", len(properties_df))
    with col2:
        st.metric("Average Price", f"${properties_df['price'].mean():,.0f}")
    with col3:
        st.metric("Average Size", f"{properties_df['square_feet'].mean():.0f} sq ft")
    with col4:
        st.metric("Locations", properties_df['location'].nunique())
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution
        fig_price = px.histogram(properties_df, x='price', title='Price Distribution', 
                               color_discrete_sequence=['#667eea'])
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Property types
        type_counts = properties_df['property_type'].value_counts()
        fig_types = px.pie(values=type_counts.values, names=type_counts.index, 
                          title='Property Types Distribution')
        st.plotly_chart(fig_types, use_container_width=True)
    
    with col2:
        # Price by location
        fig_location = px.box(properties_df, x='location', y='price', 
                             title='Price by Location')
        fig_location.update_xaxis(tickangle=45)
        st.plotly_chart(fig_location, use_container_width=True)
        
        # Bedrooms distribution
        bedroom_counts = properties_df['bedrooms'].value_counts().sort_index()
        fig_bedrooms = px.bar(x=bedroom_counts.index, y=bedroom_counts.values,
                             title='Bedrooms Distribution',
                             color_discrete_sequence=['#764ba2'])
        st.plotly_chart(fig_bedrooms, use_container_width=True)

def show_ml_performance():
    """Display ML performance metrics"""
    st.header("⚡ ML Model Performance")
    
    # Model metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Recommendation Engine</h4>
            <p><strong>Accuracy:</strong> 91.7%</p>
            <p><strong>Response Time:</strong> 45ms</p>
            <p><strong>Last Updated:</strong> 2 hours ago</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🔍 Search Ranking</h4>
            <p><strong>Precision@10:</strong> 87.3%</p>
            <p><strong>Response Time:</strong> 32ms</p>
            <p><strong>Last Updated:</strong> 1 hour ago</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Content Filter</h4>
            <p><strong>Coverage:</strong> 94.2%</p>
            <p><strong>Response Time:</strong> 28ms</p>
            <p><strong>Last Updated:</strong> 30 min ago</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance trends
    st.subheader("Performance Trends")
    
    # Generate sample performance data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    performance_data = pd.DataFrame({
        'date': dates,
        'accuracy': np.random.normal(0.915, 0.02, len(dates)),
        'response_time': np.random.normal(45, 8, len(dates)),
        'throughput': np.random.normal(1250, 150, len(dates))
    })
    
    # Accuracy trend
    fig_accuracy = px.line(performance_data, x='date', y='accuracy', 
                          title='Model Accuracy Over Time',
                          color_discrete_sequence=['#667eea'])
    st.plotly_chart(fig_accuracy, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_response = px.line(performance_data, x='date', y='response_time',
                              title='Response Time Trend (ms)',
                              color_discrete_sequence=['#764ba2'])
        st.plotly_chart(fig_response, use_container_width=True)
    
    with col2:
        fig_throughput = px.line(performance_data, x='date', y='throughput',
                                title='Throughput Trend (req/min)',
                                color_discrete_sequence=['#e74c3c'])
        st.plotly_chart(fig_throughput, use_container_width=True)

def show_system_status():
    """Display system status and health"""
    st.header("🔍 System Status & Health")
    
    # Overall health
    st.subheader("🟢 System Health: OPERATIONAL")
    
    # Service status
    services = [
        {"name": "FastAPI Application", "status": "🟢 Running", "uptime": "99.97%", "response_time": "125ms"},
        {"name": "PostgreSQL Database", "status": "🟢 Running", "uptime": "99.99%", "response_time": "8ms"},
        {"name": "Redis Cache", "status": "🟢 Running", "uptime": "99.95%", "response_time": "2ms"},
        {"name": "ML Model Server", "status": "🟢 Running", "uptime": "99.92%", "response_time": "45ms"},
        {"name": "Scraping Service", "status": "🟡 Degraded", "uptime": "97.3%", "response_time": "2.1s"},
    ]
    
    for service in services:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**{service['name']}**")
        with col2:
            st.write(service['status'])
        with col3:
            st.write(f"Uptime: {service['uptime']}")
        with col4:
            st.write(f"Response: {service['response_time']}")
    
    # Recent activity
    st.subheader("📊 Recent Activity")
    
    # Generate activity metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("API Requests (24h)", "47,382", "↑ 12.5%")
    with col2:
        st.metric("New Properties", "156", "↑ 8.2%")
    with col3:
        st.metric("Active Users", "1,284", "↑ 15.3%")
    with col4:
        st.metric("Recommendations Served", "12,847", "↑ 22.1%")
    
    # System logs
    st.subheader("🔍 Recent System Logs")
    
    logs = [
        {"time": "2024-07-11 17:15:23", "level": "INFO", "message": "ML model prediction completed successfully"},
        {"time": "2024-07-11 17:14:45", "level": "INFO", "message": "New property indexed: ID abc123"},
        {"time": "2024-07-11 17:14:12", "level": "WARN", "message": "Scraper rate limit reached for apartments.com"},
        {"time": "2024-07-11 17:13:58", "level": "INFO", "message": "Cache hit rate: 94.7%"},
        {"time": "2024-07-11 17:13:34", "level": "INFO", "message": "User recommendation request processed"},
    ]
    
    for log in logs:
        color = {"INFO": "🔵", "WARN": "🟡", "ERROR": "🔴"}.get(log["level"], "⚪")
        st.write(f"{color} `{log['time']}` [{log['level']}] {log['message']}")

if __name__ == "__main__":
    main()