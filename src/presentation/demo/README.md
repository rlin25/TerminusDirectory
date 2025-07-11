# Rental ML System - Streamlit Demo

A comprehensive interactive demonstration of the Rental ML System showcasing intelligent property search, personalized recommendations, and advanced analytics capabilities.

## 🎯 Overview

This Streamlit demo application provides a complete showcase of the Rental ML System's features, designed for stakeholders, potential users, and technical evaluators to experience the platform's capabilities firsthand.

### Key Features

- **🔍 Intelligent Property Search**: Advanced filtering with ML-powered ranking
- **🎯 Personalized Recommendations**: Hybrid recommendation engine combining collaborative filtering and content-based approaches
- **👤 User Preference Management**: Dynamic preference configuration with real-time updates
- **📊 Analytics Dashboard**: Comprehensive market insights and trend analysis
- **⚡ ML Performance Monitoring**: Real-time model performance metrics and system health
- **🆚 Property Comparison**: Side-by-side comparison tools with detailed analysis
- **📈 Market Insights**: Predictive analytics and market forecasting
- **🔍 System Monitoring**: Live system health and operational metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Required packages (see `requirements/base.txt`)

### Installation

1. **Navigate to the project root:**
   ```bash
   cd /root/terminus_directory/rental-ml-system
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements/base.txt
   ```

3. **Run the Streamlit demo:**
   ```bash
   streamlit run src/presentation/demo/app.py
   ```

4. **Open your browser to:**
   ```
   http://localhost:8501
   ```

### Alternative Launch Methods

**Using Python module:**
```bash
python -m streamlit run src/presentation/demo/app.py
```

**With custom configuration:**
```bash
streamlit run src/presentation/demo/app.py --server.port 8502 --server.address localhost
```

## 📱 Application Structure

### Main Navigation

The demo is organized into 8 main sections accessible via the sidebar:

1. **🏠 Property Search** - Explore and filter properties
2. **🎯 Recommendations** - View personalized property recommendations  
3. **👤 User Preferences** - Configure and manage user preferences
4. **📊 Analytics Dashboard** - Market analysis and data visualizations
5. **⚡ ML Performance** - Model performance metrics and monitoring
6. **🔍 System Monitoring** - System health and operational status
7. **🆚 Property Comparison** - Compare multiple properties side-by-side
8. **📈 Market Insights** - Market trends and predictive analytics

### Core Components

#### Property Search Interface
- **Advanced Filters**: Price range, location, bedrooms, bathrooms, amenities
- **Smart Sorting**: Multiple sorting options with relevance ranking
- **Real-time Results**: Instant filtering with live property updates
- **Interactive Cards**: Rich property displays with actions

#### Recommendation Engine
- **Hybrid Approach**: Combines collaborative filtering and content-based methods
- **Personalization**: Adapts to user preferences and behavior
- **Explainable AI**: Detailed explanations for each recommendation
- **Confidence Scoring**: ML confidence levels for recommendations

#### Analytics Dashboard
- **Market Trends**: Price trends and market movement analysis
- **Location Analysis**: Neighborhood-based insights and comparisons
- **Property Distributions**: Type, size, and amenity distributions
- **User Behavior**: Interaction patterns and engagement metrics

## 🛠️ Technical Architecture

### Data Layer
- **Sample Data Generator**: Creates realistic property, user, and interaction data
- **In-Memory Storage**: Fast data access for demo purposes
- **Configurable Parameters**: Adjustable data generation settings

### ML Components
- **Recommendation Models**: Simulated collaborative filtering and content-based models
- **Performance Metrics**: Real-time accuracy, response time, and system health
- **Explainability**: Feature importance and recommendation reasoning

### UI Framework
- **Streamlit**: Interactive web application framework
- **Plotly**: Advanced charts and visualizations  
- **Custom Components**: Reusable UI widgets and displays
- **Responsive Design**: Mobile-friendly interface

## 📊 Demo Data

The application includes comprehensive sample data:

- **100 Properties**: Diverse property types across multiple locations
- **50 Users**: Varied user profiles with different preferences
- **500+ Interactions**: Realistic user behavior patterns
- **Real-time Metrics**: Simulated performance and system data

### Data Generation Features

- **Realistic Distributions**: Property prices, sizes, and features based on market data
- **Geographic Clustering**: Location-based property groupings
- **User Behavior Patterns**: Authentic interaction sequences and preferences
- **Temporal Dynamics**: Time-based activity patterns and trends

## 🎨 Customization

### Configuration Options

The demo supports extensive customization through `config.py`:

```python
# UI Configuration
ui_config = UIConfig(
    page_title="Custom Demo Title",
    primary_color="#custom_color",
    chart_height=500
)

# Data Configuration  
data_config = DemoDataConfig(
    default_property_count=200,
    price_range=(1000, 10000),
    random_seed=123
)
```

### Environment Variables

Override settings using environment variables:

```bash
export DEMO_PROPERTY_COUNT=150
export DEMO_USER_COUNT=75
export RANDOM_SEED=456
streamlit run src/presentation/demo/app.py
```

### Feature Flags

Enable/disable features in `config.py`:

```python
class FeatureFlags:
    ENABLE_RECOMMENDATIONS = True
    ENABLE_ANALYTICS = True
    ENABLE_MONITORING = True
    ENABLE_COMPARISON = True
```

## 📈 Features Deep Dive

### Property Search & Filtering

**Advanced Filter System:**
- Price range sliders with real-time updates
- Multi-select location and property type filters
- Amenity requirement checkboxes
- Size and bathroom minimum requirements
- Sorting by price, size, bedrooms, location

**Search Capabilities:**
- Full-text search across titles, descriptions, and amenities
- Location-based search with neighborhood matching
- Keyword highlighting and result ranking

### Recommendation Engine

**Algorithm Types:**
- **Collaborative Filtering**: User-based similarity recommendations
- **Content-Based**: Property feature matching recommendations  
- **Hybrid Model**: Weighted combination of both approaches

**Personalization Features:**
- Dynamic preference learning from user interactions
- Configurable recommendation weights
- Diversity and novelty optimization
- Cold-start handling for new users

### Analytics & Insights

**Market Analysis:**
- Price trend analysis with historical data
- Location-based market segmentation
- Property type distribution analysis
- Market velocity and competitiveness metrics

**User Analytics:**
- Interaction pattern analysis
- User behavior segmentation
- Engagement metrics and retention analysis
- Conversion funnel tracking

### Performance Monitoring

**ML Model Metrics:**
- Real-time accuracy tracking across models
- Response time and throughput monitoring
- Confidence score distributions
- Model drift detection

**System Health:**
- API endpoint status monitoring
- Database performance metrics
- Resource utilization tracking
- Error rate and alert management

## 🔧 Development

### Project Structure

```
src/presentation/demo/
├── app.py                 # Main Streamlit application
├── components.py          # Reusable UI components
├── sample_data.py         # Sample data generation
├── utils.py              # Utility functions
├── config.py             # Configuration settings
└── README.md             # This file
```

### Adding New Features

1. **Create Component**: Add new UI component to `components.py`
2. **Add Route**: Include new page in main navigation
3. **Update Config**: Add configuration parameters if needed
4. **Test Data**: Ensure sample data supports new features

### Extending Sample Data

```python
# In sample_data.py
def generate_custom_data(self, count: int) -> List[CustomEntity]:
    """Generate custom demo data"""
    # Implementation here
    pass
```

## 🚀 Deployment Options

### Local Development
```bash
streamlit run src/presentation/demo/app.py --server.runOnSave true
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements/base.txt
EXPOSE 8501
CMD ["streamlit", "run", "src/presentation/demo/app.py"]
```

### Cloud Deployment
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Web app deployment
- **AWS/GCP**: Container or serverless deployment

## 📚 Usage Examples

### Basic Property Search
1. Navigate to "🏠 Property Search"
2. Set price range: $1,500 - $3,500
3. Select locations: Downtown, Midtown
4. Choose property types: Apartment, Condo
5. Add required amenities: Parking, Gym
6. View filtered results with relevance ranking

### Getting Recommendations
1. Go to "👤 User Preferences"
2. Configure budget, location, and size preferences
3. Save preferences
4. Navigate to "🎯 Recommendations"
5. Generate personalized recommendations
6. View explanations and confidence scores

### Market Analysis
1. Open "📊 Analytics Dashboard"
2. Explore price distribution charts
3. Analyze location-based trends
4. Review user activity patterns
5. Export data for further analysis

### Performance Monitoring
1. Access "⚡ ML Performance"
2. Monitor model accuracy over time
3. Check response time metrics
4. Review system resource usage
5. Analyze recommendation quality metrics

## 🎓 Tutorial Mode

The demo includes an optional tutorial mode for new users:

1. **Welcome Screen**: Introduction and feature overview
2. **Guided Tour**: Step-by-step walkthrough of key features
3. **Interactive Examples**: Hands-on demonstration of capabilities
4. **Best Practices**: Tips for effective platform usage

Enable tutorial mode in the sidebar or through the help menu.

## 📊 Metrics & Analytics

### Key Performance Indicators
- **User Engagement**: Page views, interaction rates, session duration
- **Search Performance**: Query response times, result relevance
- **Recommendation Quality**: Click-through rates, conversion rates
- **System Health**: Uptime, error rates, resource utilization

### Export Capabilities
- **CSV Export**: Property data, user interactions, analytics results
- **JSON Export**: Complete data structures for API integration
- **Chart Downloads**: PNG/PDF exports of visualizations
- **Report Generation**: Automated summary reports

## 🔒 Security & Privacy

### Demo Security Features
- **No Personal Data**: All demo data is synthetic
- **Local Processing**: No external data transmission
- **Session Isolation**: User sessions are independent
- **Safe Defaults**: Conservative security settings

### Production Considerations
- User authentication and authorization
- Data encryption in transit and at rest
- API rate limiting and monitoring
- GDPR/privacy compliance measures

## 🐛 Troubleshooting

### Common Issues

**Streamlit won't start:**
```bash
# Check Python version
python --version  # Should be 3.8+

# Reinstall dependencies
pip install --upgrade streamlit

# Clear cache
streamlit cache clear
```

**Demo data not loading:**
```bash
# Check file permissions
ls -la src/presentation/demo/

# Regenerate sample data
python -c "from src.presentation.demo.sample_data import SampleDataGenerator; SampleDataGenerator().generate_properties(100)"
```

**Performance issues:**
```bash
# Reduce sample data size
export DEMO_PROPERTY_COUNT=50
export DEMO_USER_COUNT=25

# Disable expensive features
# Edit config.py FeatureFlags
```

### Error Messages

| Error | Solution |
|-------|----------|
| Module not found | Install requirements: `pip install -r requirements/base.txt` |
| Port already in use | Change port: `streamlit run app.py --server.port 8502` |
| Memory issues | Reduce data size in `config.py` |
| Chart rendering errors | Update Plotly: `pip install --upgrade plotly` |

## 📝 Changelog

### Version 1.0.0 (Current)
- ✅ Complete property search interface
- ✅ Hybrid recommendation engine demo
- ✅ Comprehensive analytics dashboard  
- ✅ ML performance monitoring
- ✅ System health monitoring
- ✅ Property comparison tools
- ✅ Market insights and forecasting
- ✅ Responsive mobile design

### Planned Features
- 🔄 Real-time data updates
- 🔄 Advanced user authentication
- 🔄 API integration examples
- 🔄 Extended ML explanations
- 🔄 Multi-language support

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-demo-feature`
3. Make changes and test thoroughly
4. Submit pull request with detailed description

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints for function signatures
- Add docstrings for all public functions
- Include unit tests for new features

## 📞 Support

### Getting Help
- **Documentation**: This README and inline code comments
- **Issues**: GitHub issue tracker for bug reports
- **Discussions**: GitHub discussions for questions and ideas
- **Demo Support**: Technical support for demo setup and usage

### Feedback
We welcome feedback on the demo application:
- Feature requests and improvements
- Bug reports and issues
- Usability suggestions
- Performance optimization ideas

## 📄 License

This demo application is part of the Rental ML System project. See the main project README for licensing information.

## 🙏 Acknowledgments

- **Streamlit Team**: For the excellent web app framework
- **Plotly Team**: For powerful visualization capabilities
- **Open Source Community**: For the many libraries that make this possible

---

**Ready to explore the future of rental property platforms? Launch the demo and discover the power of AI-driven property matching!** 🚀