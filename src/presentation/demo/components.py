"""
Reusable Streamlit Components for Rental ML System Demo

This module contains custom UI components and widgets for the demo application:
- Property display cards
- Search filter interfaces
- Recommendation cards
- Metrics displays
- Interactive widgets
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.domain.entities.property import Property
from src.domain.entities.user import User


class PropertyCard:
    """Component for displaying property information in a card format"""
    
    @staticmethod
    def render(property_obj: Property, 
               show_recommendation_score: bool = False,
               recommendation_score: Optional[float] = None,
               show_actions: bool = True,
               compact: bool = False) -> None:
        """
        Render a property card with all relevant information
        
        Args:
            property_obj: Property entity to display
            show_recommendation_score: Whether to show ML recommendation score
            recommendation_score: Score to display (0-1)
            show_actions: Whether to show action buttons
            compact: Whether to use compact layout
        """
        
        with st.container():
            if compact:
                PropertyCard._render_compact(property_obj, show_recommendation_score, recommendation_score)
            else:
                PropertyCard._render_full(property_obj, show_recommendation_score, recommendation_score, show_actions)
    
    @staticmethod
    def _render_full(property_obj: Property, 
                    show_recommendation_score: bool,
                    recommendation_score: Optional[float],
                    show_actions: bool) -> None:
        """Render full property card layout"""
        
        # Create main layout
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            # Property image
            if property_obj.images:
                st.image(property_obj.images[0], width=300, caption="Property Image")
            else:
                st.image("https://via.placeholder.com/300x200/667eea/ffffff?text=No+Image", 
                        width=300, caption="No Image Available")
            
            # Recommendation score if applicable
            if show_recommendation_score and recommendation_score is not None:
                score_color = PropertyCard._get_score_color(recommendation_score)
                st.markdown(f"""
                <div style="background: {score_color}; color: white; padding: 10px; 
                           border-radius: 10px; text-align: center; margin-top: 10px;">
                    <strong>Match Score: {recommendation_score:.1%}</strong>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Property details
            st.markdown(f"### {property_obj.title}")
            
            # Location and basic info
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.markdown(f"📍 **Location:** {property_obj.location}")
                st.markdown(f"🏠 **Type:** {property_obj.property_type.title()}")
                st.markdown(f"🛏️ **Bedrooms:** {property_obj.bedrooms}")
                st.markdown(f"🚿 **Bathrooms:** {property_obj.bathrooms}")
            
            with col_info2:
                st.markdown(f"💰 **Price:** ${property_obj.price:,}/month")
                if property_obj.square_feet:
                    st.markdown(f"📐 **Size:** {property_obj.square_feet:,} sq ft")
                    price_per_sqft = property_obj.get_price_per_sqft()
                    if price_per_sqft:
                        st.markdown(f"💵 **Price/sq ft:** ${price_per_sqft:.2f}")
                
                # Status indicator
                status_color = "#28a745" if property_obj.is_active else "#dc3545"
                status_text = "Available" if property_obj.is_active else "Not Available"
                st.markdown(f"🔄 **Status:** <span style='color: {status_color}'>{status_text}</span>", 
                           unsafe_allow_html=True)
            
            # Amenities
            if property_obj.amenities:
                amenities_display = PropertyCard._format_amenities(property_obj.amenities)
                st.markdown(f"🏠 **Amenities:** {amenities_display}")
            
            # Description (truncated)
            max_desc_length = 200
            description = property_obj.description
            if len(description) > max_desc_length:
                description = description[:max_desc_length] + "..."
            st.markdown(f"📝 **Description:** {description}")
            
            # Additional info
            scraped_date = property_obj.scraped_at.strftime("%B %d, %Y")
            st.markdown(f"📅 **Listed:** {scraped_date}")
        
        with col3:
            if show_actions:
                PropertyCard._render_action_buttons(property_obj)
    
    @staticmethod
    def _render_compact(property_obj: Property,
                       show_recommendation_score: bool,
                       recommendation_score: Optional[float]) -> None:
        """Render compact property card layout"""
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Small property image
            if property_obj.images:
                st.image(property_obj.images[0], width=150)
            else:
                st.image("https://via.placeholder.com/150x100/667eea/ffffff?text=No+Image", width=150)
        
        with col2:
            st.markdown(f"**{property_obj.title}**")
            
            col_details1, col_details2 = st.columns(2)
            with col_details1:
                st.write(f"📍 {property_obj.location}")
                st.write(f"💰 ${property_obj.price:,}/month")
            
            with col_details2:
                st.write(f"🛏️ {property_obj.bedrooms} bed, 🚿 {property_obj.bathrooms} bath")
                if property_obj.square_feet:
                    st.write(f"📐 {property_obj.square_feet:,} sq ft")
            
            if show_recommendation_score and recommendation_score is not None:
                st.write(f"🎯 Match: {recommendation_score:.1%}")
    
    @staticmethod
    def _render_action_buttons(property_obj: Property) -> None:
        """Render action buttons for property card"""
        
        st.markdown("### Actions")
        
        if st.button(f"👁️ View Details", key=f"view_{property_obj.id}"):
            st.session_state[f"selected_property"] = property_obj.id
            st.success("Property details viewed!")
        
        if st.button(f"💖 Like", key=f"like_{property_obj.id}"):
            # Add to user's liked properties
            st.success("Added to liked properties!")
        
        if st.button(f"📞 Contact", key=f"contact_{property_obj.id}"):
            # Show contact info
            st.info(f"📧 {property_obj.contact_info.get('email', 'N/A')}")
            st.info(f"📱 {property_obj.contact_info.get('phone', 'N/A')}")
        
        if st.button(f"💾 Save", key=f"save_{property_obj.id}"):
            st.success("Property saved for later!")
        
        if st.button(f"📊 Compare", key=f"compare_{property_obj.id}"):
            # Add to comparison list
            if 'comparison_properties' not in st.session_state:
                st.session_state.comparison_properties = []
            
            if property_obj.id not in st.session_state.comparison_properties:
                st.session_state.comparison_properties.append(property_obj.id)
                st.success("Added to comparison!")
            else:
                st.warning("Already in comparison list!")
    
    @staticmethod
    def _format_amenities(amenities: List[str], max_display: int = 4) -> str:
        """Format amenities list for display"""
        if not amenities:
            return "None listed"
        
        # Convert amenities to readable format
        formatted_amenities = []
        for amenity in amenities:
            formatted = amenity.replace('-', ' ').replace('_', ' ').title()
            formatted_amenities.append(formatted)
        
        if len(formatted_amenities) <= max_display:
            return ", ".join(formatted_amenities)
        else:
            displayed = ", ".join(formatted_amenities[:max_display])
            remaining = len(formatted_amenities) - max_display
            return f"{displayed} (+{remaining} more)"
    
    @staticmethod
    def _get_score_color(score: float) -> str:
        """Get color based on recommendation score"""
        if score >= 0.8:
            return "linear-gradient(45deg, #28a745, #20c997)"
        elif score >= 0.6:
            return "linear-gradient(45deg, #ffc107, #fd7e14)"
        elif score >= 0.4:
            return "linear-gradient(45deg, #fd7e14, #dc3545)"
        else:
            return "linear-gradient(45deg, #dc3545, #6f42c1)"


class SearchFilters:
    """Component for property search filters"""
    
    @staticmethod
    def render(filter_key: str = "main") -> Dict[str, Any]:
        """
        Render search filter interface
        
        Args:
            filter_key: Unique key for this filter instance
            
        Returns:
            Dictionary containing filter values
        """
        
        filters = {}
        
        with st.expander("🎛️ Search Filters", expanded=True):
            
            # Price filters
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**💰 Price Range**")
                price_min = st.number_input(
                    "Minimum Price ($)", 
                    min_value=0, 
                    max_value=20000, 
                    value=1000, 
                    step=100,
                    key=f"{filter_key}_price_min"
                )
                price_max = st.number_input(
                    "Maximum Price ($)", 
                    min_value=price_min, 
                    max_value=20000, 
                    value=5000, 
                    step=100,
                    key=f"{filter_key}_price_max"
                )
                filters['price_range'] = (price_min, price_max)
            
            with col2:
                st.markdown("**🏠 Property Type**")
                property_types = st.multiselect(
                    "Select Types",
                    ["apartment", "house", "condo", "studio", "townhouse"],
                    default=["apartment", "house"],
                    key=f"{filter_key}_property_types"
                )
                filters['property_types'] = property_types
            
            # Bedroom and bathroom filters
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("**🛏️ Bedrooms**")
                bedrooms = st.multiselect(
                    "Number of Bedrooms",
                    [0, 1, 2, 3, 4, 5],
                    default=[1, 2, 3],
                    key=f"{filter_key}_bedrooms"
                )
                filters['bedrooms'] = bedrooms
            
            with col4:
                st.markdown("**🚿 Bathrooms**")
                min_bathrooms = st.selectbox(
                    "Minimum Bathrooms",
                    [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                    index=0,
                    key=f"{filter_key}_min_bathrooms"
                )
                filters['min_bathrooms'] = min_bathrooms
            
            # Location and size filters
            col5, col6 = st.columns(2)
            
            with col5:
                st.markdown("**📍 Locations**")
                # This would be populated from actual data
                all_locations = [
                    "Downtown", "Midtown", "Uptown", "Financial District",
                    "Arts District", "Suburban Heights", "Riverside", "Hillside"
                ]
                selected_locations = st.multiselect(
                    "Preferred Locations",
                    all_locations,
                    key=f"{filter_key}_locations"
                )
                filters['locations'] = selected_locations
            
            with col6:
                st.markdown("**📐 Size**")
                min_sqft = st.number_input(
                    "Minimum Sq Ft",
                    min_value=0,
                    max_value=5000,
                    value=500,
                    step=50,
                    key=f"{filter_key}_min_sqft"
                )
                filters['min_sqft'] = min_sqft
            
            # Amenities filter
            st.markdown("**🏠 Amenities**")
            amenities_options = [
                "parking", "gym", "pool", "laundry", "pet-friendly", 
                "balcony", "air-conditioning", "heating", "dishwasher"
            ]
            selected_amenities = st.multiselect(
                "Required Amenities",
                amenities_options,
                key=f"{filter_key}_amenities"
            )
            filters['amenities'] = selected_amenities
            
            # Sort options
            col7, col8 = st.columns(2)
            
            with col7:
                sort_by = st.selectbox(
                    "Sort By",
                    ["price", "bedrooms", "bathrooms", "square_feet", "location", "newest"],
                    key=f"{filter_key}_sort_by"
                )
                filters['sort_by'] = sort_by
            
            with col8:
                sort_order = st.selectbox(
                    "Sort Order",
                    ["ascending", "descending"],
                    index=0,
                    key=f"{filter_key}_sort_order"
                )
                filters['sort_order'] = sort_order
        
        return filters


class RecommendationCard:
    """Component for displaying recommendation results"""
    
    @staticmethod
    def render(property_obj: Property, 
               recommendation_data: Dict[str, Any],
               user: Optional[User] = None) -> None:
        """
        Render recommendation card with explanation
        
        Args:
            property_obj: Property being recommended
            recommendation_data: ML recommendation data
            user: User receiving recommendation
        """
        
        with st.container():
            # Main recommendation layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Use PropertyCard for basic display
                PropertyCard.render(property_obj, 
                                  show_recommendation_score=True,
                                  recommendation_score=recommendation_data.get('score', 0),
                                  show_actions=False,
                                  compact=True)
            
            with col2:
                RecommendationCard._render_explanation(recommendation_data, user)
    
    @staticmethod
    def _render_explanation(recommendation_data: Dict[str, Any], user: Optional[User]) -> None:
        """Render recommendation explanation"""
        
        st.markdown("### 🧠 Why Recommended?")
        
        score = recommendation_data.get('score', 0)
        explanation = recommendation_data.get('explanation', 'Based on your preferences')
        
        # Score visualization
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Match %"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgray"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation text
        st.markdown(f"**Reason:** {explanation}")
        
        # Additional details if available
        if 'details' in recommendation_data:
            details = recommendation_data['details']
            
            if 'cf_score' in details:
                st.metric("Similar Users", f"{details['cf_score']:.1%}")
            
            if 'cb_score' in details:
                st.metric("Feature Match", f"{details['cb_score']:.1%}")
            
            if 'confidence' in details:
                st.metric("Confidence", f"{details['confidence']:.1%}")


class MetricsDisplay:
    """Component for displaying various metrics and KPIs"""
    
    @staticmethod
    def render_kpi_cards(metrics: Dict[str, Any]) -> None:
        """Render KPI cards layout"""
        
        cols = st.columns(len(metrics))
        
        for i, (metric_name, metric_data) in enumerate(metrics.items()):
            with cols[i]:
                value = metric_data.get('value', 0)
                delta = metric_data.get('delta', None)
                format_type = metric_data.get('format', 'number')
                
                # Format value based on type
                if format_type == 'currency':
                    formatted_value = f"${value:,.0f}"
                elif format_type == 'percentage':
                    formatted_value = f"{value:.1%}"
                elif format_type == 'number':
                    formatted_value = f"{value:,.0f}"
                else:
                    formatted_value = str(value)
                
                st.metric(
                    label=metric_name,
                    value=formatted_value,
                    delta=delta
                )
    
    @staticmethod
    def render_performance_chart(data: Dict[str, Any], chart_type: str = "line") -> None:
        """Render performance chart"""
        
        if chart_type == "line":
            fig = px.line(
                x=data.get('x', []),
                y=data.get('y', []),
                title=data.get('title', 'Performance Chart'),
                labels=data.get('labels', {})
            )
        elif chart_type == "bar":
            fig = px.bar(
                x=data.get('x', []),
                y=data.get('y', []),
                title=data.get('title', 'Performance Chart'),
                labels=data.get('labels', {})
            )
        elif chart_type == "gauge":
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=data.get('value', 0),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': data.get('title', 'Gauge')},
                delta={'reference': data.get('reference', 0)},
                gauge={
                    'axis': {'range': [None, data.get('max_value', 100)]},
                    'bar': {'color': data.get('color', 'darkblue')},
                    'steps': data.get('steps', []),
                    'threshold': data.get('threshold', {})
                }
            ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_status_indicators(services: Dict[str, str]) -> None:
        """Render service status indicators"""
        
        for service_name, status in services.items():
            status_color = {
                'healthy': '#28a745',
                'warning': '#ffc107', 
                'error': '#dc3545',
                'unknown': '#6c757d'
            }.get(status.lower(), '#6c757d')
            
            status_icon = {
                'healthy': '🟢',
                'warning': '🟡',
                'error': '🔴',
                'unknown': '⚫'
            }.get(status.lower(), '⚫')
            
            st.markdown(
                f"{status_icon} **{service_name}:** "
                f"<span style='color: {status_color}'>{status.title()}</span>",
                unsafe_allow_html=True
            )


class InteractiveMap:
    """Component for displaying interactive property maps"""
    
    @staticmethod
    def render(properties: List[Property], center_lat: float = 40.7128, center_lon: float = -74.0060) -> None:
        """Render interactive map with property locations"""
        
        # Prepare map data
        map_data = []
        for prop in properties:
            # Generate random coordinates for demo (in real app, use actual coordinates)
            lat = center_lat + np.random.normal(0, 0.01)
            lon = center_lon + np.random.normal(0, 0.01)
            
            map_data.append({
                'lat': lat,
                'lon': lon,
                'title': prop.title,
                'price': prop.price,
                'location': prop.location,
                'bedrooms': prop.bedrooms,
                'bathrooms': prop.bathrooms,
                'property_type': prop.property_type
            })
        
        if map_data:
            df = pd.DataFrame(map_data)
            
            # Create scatter mapbox
            fig = px.scatter_mapbox(
                df,
                lat="lat",
                lon="lon",
                hover_name="title",
                hover_data=["price", "bedrooms", "bathrooms"],
                color="price",
                size="bedrooms",
                color_continuous_scale="Viridis",
                size_max=15,
                zoom=10,
                mapbox_style="open-street-map"
            )
            
            fig.update_layout(
                title="Property Locations",
                height=600,
                margin={"r": 0, "t": 30, "l": 0, "b": 0}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No properties to display on map")


class DataTable:
    """Component for displaying data in table format with advanced features"""
    
    @staticmethod
    def render(data: pd.DataFrame, 
               title: str = "",
               sortable_columns: List[str] = None,
               filterable_columns: List[str] = None,
               show_download: bool = True) -> pd.DataFrame:
        """
        Render interactive data table
        
        Args:
            data: DataFrame to display
            title: Table title
            sortable_columns: Columns that can be sorted
            filterable_columns: Columns that can be filtered
            show_download: Whether to show download button
            
        Returns:
            Filtered/sorted DataFrame
        """
        
        if title:
            st.subheader(title)
        
        # Filtering interface
        filtered_data = data.copy()
        
        if filterable_columns:
            with st.expander("🔍 Table Filters"):
                filter_cols = st.columns(len(filterable_columns))
                
                for i, col_name in enumerate(filterable_columns):
                    with filter_cols[i]:
                        if data[col_name].dtype in ['object', 'string']:
                            # Text filter for categorical columns
                            unique_values = data[col_name].unique()
                            selected_values = st.multiselect(
                                f"Filter {col_name}",
                                unique_values,
                                key=f"filter_{col_name}"
                            )
                            if selected_values:
                                filtered_data = filtered_data[filtered_data[col_name].isin(selected_values)]
                        
                        elif data[col_name].dtype in ['int64', 'float64']:
                            # Range filter for numerical columns
                            min_val = float(data[col_name].min())
                            max_val = float(data[col_name].max())
                            
                            range_values = st.slider(
                                f"Filter {col_name}",
                                min_val,
                                max_val,
                                (min_val, max_val),
                                key=f"range_{col_name}"
                            )
                            
                            filtered_data = filtered_data[
                                (filtered_data[col_name] >= range_values[0]) &
                                (filtered_data[col_name] <= range_values[1])
                            ]
        
        # Sorting interface
        if sortable_columns:
            col1, col2 = st.columns(2)
            
            with col1:
                sort_column = st.selectbox(
                    "Sort by",
                    ["None"] + sortable_columns,
                    key="sort_column"
                )
            
            with col2:
                sort_order = st.selectbox(
                    "Sort order",
                    ["Ascending", "Descending"],
                    key="sort_order"
                )
            
            if sort_column != "None":
                ascending = sort_order == "Ascending"
                filtered_data = filtered_data.sort_values(sort_column, ascending=ascending)
        
        # Display table
        st.dataframe(filtered_data, use_container_width=True)
        
        # Download button
        if show_download:
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        return filtered_data


class ComparisonTable:
    """Component for side-by-side property comparison"""
    
    @staticmethod
    def render(properties: List[Property], max_properties: int = 4) -> None:
        """
        Render property comparison table
        
        Args:
            properties: List of properties to compare
            max_properties: Maximum number of properties to compare
        """
        
        if not properties:
            st.info("No properties selected for comparison")
            return
        
        # Limit number of properties
        properties = properties[:max_properties]
        
        # Prepare comparison data
        comparison_data = {
            'Feature': [
                'Title', 'Price', 'Price per Sq Ft', 'Location', 'Property Type',
                'Bedrooms', 'Bathrooms', 'Square Feet', 'Amenities Count',
                'Status', 'Listed Date'
            ]
        }
        
        for i, prop in enumerate(properties):
            column_name = f"Property {i + 1}"
            
            # Truncate title if too long
            title = prop.title[:25] + "..." if len(prop.title) > 25 else prop.title
            
            comparison_data[column_name] = [
                title,
                f"${prop.price:,}",
                f"${prop.get_price_per_sqft():.2f}" if prop.get_price_per_sqft() else "N/A",
                prop.location,
                prop.property_type.title(),
                prop.bedrooms,
                prop.bathrooms,
                f"{prop.square_feet:,}" if prop.square_feet else "N/A",
                len(prop.amenities),
                "Available" if prop.is_active else "Not Available",
                prop.scraped_at.strftime("%m/%d/%Y")
            ]
        
        # Create and display comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Style the table
        def highlight_best_value(row):
            """Highlight best values in each row"""
            styles = [''] * len(row)
            
            # Highlight lowest price
            if row.name == 1:  # Price row
                prices = []
                for col in row[1:]:
                    try:
                        price = float(col.replace('$', '').replace(',', ''))
                        prices.append(price)
                    except:
                        prices.append(float('inf'))
                
                min_price_idx = prices.index(min(prices)) + 1
                styles[min_price_idx] = 'background-color: #d4edda'
            
            # Highlight most bedrooms
            elif row.name == 5:  # Bedrooms row
                bedrooms = [int(x) if str(x).isdigit() else 0 for x in row[1:]]
                max_bedrooms_idx = bedrooms.index(max(bedrooms)) + 1
                styles[max_bedrooms_idx] = 'background-color: #d4edda'
            
            return styles
        
        styled_df = comparison_df.style.apply(highlight_best_value, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Show detailed amenities comparison
        with st.expander("🏠 Detailed Amenities Comparison"):
            all_amenities = set()
            for prop in properties:
                all_amenities.update(prop.amenities)
            
            amenity_comparison = {'Amenity': list(all_amenities)}
            
            for i, prop in enumerate(properties):
                column_name = f"Property {i + 1}"
                amenity_comparison[column_name] = [
                    '✅' if amenity in prop.amenities else '❌'
                    for amenity in all_amenities
                ]
            
            amenity_df = pd.DataFrame(amenity_comparison)
            st.dataframe(amenity_df, use_container_width=True)