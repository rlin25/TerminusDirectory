# Requirements Files

This directory contains the dependency management files for the rental ML system.

## File Structure

- `base.txt` - Core dependencies required for the application to run
- `dev.txt` - Development dependencies including testing, linting, and development tools
- `prod.txt` - Production-specific dependencies for deployment and monitoring
- `README.md` - This file explaining the requirements structure

## Usage

### Local Development
```bash
# Install base + development dependencies
pip install -r requirements/dev.txt
```

### Production Deployment
```bash
# Install base + production dependencies
pip install -r requirements/prod.txt
```

### Base Only
```bash
# Install only core dependencies
pip install -r requirements/base.txt
```

### Using pyproject.toml (Recommended)
```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install with production dependencies
pip install -e ".[prod]"

# Install base package only
pip install -e .
```

## Dependency Categories

### Core Dependencies (base.txt)
- **ML Framework**: TensorFlow 2.13+ for deep learning models
- **NLP**: Transformers library for natural language processing
- **Data Processing**: NumPy, Pandas, Scikit-learn for data manipulation
- **Async Operations**: AsyncIO, AioHTTP for asynchronous processing
- **Data Validation**: Pydantic for request/response validation
- **Database**: SQLAlchemy for ORM operations
- **Caching**: Redis for high-performance caching
- **Web Scraping**: BeautifulSoup4 for data extraction
- **API Framework**: FastAPI with Uvicorn for REST API
- **Demo Interface**: Streamlit for interactive demonstrations
- **Testing**: Pytest for unit and integration testing

### Development Dependencies (dev.txt)
- **Testing Extensions**: pytest-asyncio, pytest-cov for comprehensive testing
- **Code Quality**: Black, Flake8, isort for code formatting and linting
- **Type Checking**: MyPy for static type analysis
- **Development Tools**: Jupyter notebooks for experimentation
- **Visualization**: Matplotlib, Seaborn, Plotly for data visualization
- **Documentation**: Sphinx for API documentation
- **Security**: Bandit for security vulnerability scanning

### Production Dependencies (prod.txt)
- **WSGI Server**: Gunicorn for production deployment
- **Monitoring**: Prometheus client for metrics collection
- **Error Tracking**: Sentry SDK for error monitoring and alerting
- **Background Tasks**: Celery for asynchronous task processing
- **Security**: Additional security headers and SSL support
- **Performance**: Memory optimization and connection pooling

## Version Management

All dependencies use semantic versioning with compatible version ranges:
- Major version is pinned to ensure API compatibility
- Minor versions allow for feature updates and bug fixes
- Patch versions are flexible for security updates

Example: `tensorflow>=2.13.0,<2.15.0`

## Python Version Support

- **Minimum**: Python 3.9
- **Maximum**: Python 3.11 (tested)
- **Recommended**: Python 3.10 or 3.11

## Installation Best Practices

1. **Use Virtual Environments**: Always install in a virtual environment
2. **Pin Dependencies**: Use exact versions in production deployments
3. **Regular Updates**: Keep dependencies updated for security patches
4. **Security Scanning**: Run `safety check` before deployment
5. **Dependency Audit**: Regularly audit dependencies for vulnerabilities

## Troubleshooting

### Common Issues

1. **TensorFlow GPU Support**: Install tensorflow-gpu if CUDA is available
2. **Memory Issues**: Use `--no-cache-dir` flag for pip install
3. **Compilation Errors**: Install build tools (gcc, python-dev)
4. **Database Drivers**: Install system-level database libraries

### Environment Variables

Set these environment variables for optimal performance:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/rental-ml-system/src"
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow verbosity
export PYTHONIOENCODING=utf-8
```

## CI/CD Integration

These requirements files are designed to work with CI/CD pipelines:

```yaml
# Example GitHub Actions usage
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements/dev.txt
```

## Security Considerations

- All dependencies are pinned to specific version ranges
- Regular security audits using `safety` and `bandit`
- Production dependencies include security-focused packages
- No development tools are included in production requirements