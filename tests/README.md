# Rental ML System Test Suite

This directory contains a comprehensive test suite for the rental ML system, covering unit tests, integration tests, performance tests, and ML-specific validation tests.

## 📁 Test Structure

```
tests/
├── README.md                          # This file
├── conftest.py                         # Global pytest configuration and fixtures
├── utils/                              # Test utilities and helpers
│   ├── test_helpers.py                # Test helper functions and assertions
│   └── data_factories.py             # Data factories for generating test data
├── unit/                              # Unit tests (fast, isolated)
│   ├── test_domain/                   # Domain entity tests
│   │   ├── test_property.py          # Property entity tests
│   │   ├── test_user.py              # User entity tests
│   │   └── test_search_query.py     # SearchQuery entity tests
│   ├── test_application/             # Application service tests
│   │   ├── test_scraping_use_case.py # Scraping use case tests
│   │   └── test_api_routers.py       # API router tests
│   └── test_infrastructure/          # Infrastructure layer tests
│       ├── test_content_recommender.py      # Content-based recommender tests
│       ├── test_hybrid_recommender.py       # Hybrid recommender tests (existing)
│       ├── test_hybrid_recommender_enhanced.py # Enhanced hybrid tests
│       ├── test_repositories.py             # Repository tests
│       └── test_ml_training_accuracy.py     # ML training and accuracy tests
├── integration/                       # Integration tests (slower, with dependencies)
│   ├── test_api/                     # API integration tests
│   │   └── test_api_integration.py   # Full API request/response cycle tests
│   ├── test_ml_pipeline/             # ML pipeline integration tests
│   └── test_database/                # Database integration tests
├── performance/                       # Performance and load tests
│   └── test_ml_performance.py        # ML model performance tests
└── fixtures/                         # Test data fixtures
    ├── sample_properties.json        # Sample property data
    ├── sample_users.json            # Sample user data
    └── ml_test_data/                 # ML-specific test datasets
```

## 🚀 Running Tests

### Quick Start

```bash
# Run all tests
python scripts/run_tests.py --all

# Run unit tests only
python scripts/run_tests.py --unit

# Run with coverage
python scripts/run_tests.py --unit --coverage

# Run specific test markers
python scripts/run_tests.py --marker ml
python scripts/run_tests.py --marker performance
```

### Test Categories

#### Unit Tests
Fast, isolated tests for individual components:

```bash
# All unit tests
pytest tests/unit/ -m unit

# Domain entities only
pytest tests/unit/test_domain/ -v

# ML models only
pytest tests/unit/test_infrastructure/ -m ml

# API routers only
pytest tests/unit/test_application/ -m api
```

#### Integration Tests
Tests that verify component interactions:

```bash
# All integration tests
pytest tests/integration/ -m integration

# API integration tests
pytest tests/integration/test_api/ -v

# ML pipeline integration
pytest tests/integration/test_ml_pipeline/ -v
```

#### Performance Tests
Tests that measure performance and scalability:

```bash
# All performance tests
pytest tests/performance/ -m performance

# ML performance only
pytest tests/performance/test_ml_performance.py -v

# Skip slow tests
pytest -m "performance and not slow"
```

#### ML-Specific Tests
Comprehensive ML model validation:

```bash
# All ML tests
pytest -m ml

# Model training validation
pytest tests/unit/test_infrastructure/test_ml_training_accuracy.py

# Model inference performance
pytest tests/performance/test_ml_performance.py::TestMLModelInferencePerformance
```

### Advanced Usage

#### Parallel Testing
```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n 4

# Using test runner
python scripts/run_tests.py --parallel 8
```

#### Coverage Reports
```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Generate XML coverage report
pytest --cov=src --cov-report=xml

# Fail if coverage below threshold
pytest --cov=src --cov-fail-under=85
```

#### Debugging Tests
```bash
# Stop on first failure
pytest -x

# Enter debugger on failures
pytest --pdb

# Verbose output
pytest -v

# Show local variables in failures
pytest -l
```

## 🏷️ Test Markers

The test suite uses pytest markers to categorize tests:

- `unit`: Fast, isolated unit tests
- `integration`: Integration tests with dependencies
- `performance`: Performance and load tests
- `ml`: Machine learning specific tests
- `api`: API endpoint tests
- `db`: Database related tests
- `slow`: Slow running tests (can be excluded)
- `smoke`: Basic functionality smoke tests

### Using Markers

```bash
# Run only fast tests
pytest -m "not slow"

# Run ML and performance tests
pytest -m "ml or performance"

# Run unit tests excluding slow ones
pytest -m "unit and not slow"

# Run integration tests for API only
pytest -m "integration and api"
```

## 📊 Test Coverage

The test suite aims for high code coverage across all components:

- **Domain Layer**: >95% coverage (simple entities, critical business logic)
- **Application Layer**: >90% coverage (use cases, API routes)
- **Infrastructure Layer**: >85% coverage (ML models, repositories)
- **Overall Target**: >85% coverage

### Coverage Reports

After running tests with coverage:

```bash
# View terminal report
pytest --cov=src --cov-report=term-missing

# Open HTML report
open htmlcov/index.html

# View XML report (for CI/CD)
cat coverage.xml
```

## 🧪 Test Data and Fixtures

### Factories
The test suite uses factory classes to generate consistent test data:

```python
from tests.utils.data_factories import PropertyFactory, UserFactory, MLDataFactory

# Generate test properties
property_factory = PropertyFactory(FactoryConfig(seed=42))
properties = property_factory.create_batch(10)

# Generate test users
user_factory = UserFactory(FactoryConfig(seed=42))
users = user_factory.create_batch(5)

# Generate ML training data
ml_factory = MLDataFactory(FactoryConfig(seed=42))
training_data = ml_factory.create_training_data(
    num_users=100, num_properties=200, density=0.1
)
```

### Fixtures
Global fixtures are available in `conftest.py`:

```python
def test_with_fixtures(sample_property, sample_user, mock_redis):
    # Use pre-configured test objects
    assert sample_property.is_active
    assert sample_user.email
    mock_redis.get.return_value = "cached_data"
```

## 🔧 Test Configuration

### pytest.ini
Main pytest configuration with markers, coverage settings, and test discovery rules.

### conftest.py
Global test configuration including:
- Test fixtures and utilities
- Database test setup
- ML model mocks
- Performance timing utilities
- Custom assertion helpers

### Environment Variables
Tests use environment-specific settings:

```bash
export TESTING=1                    # Enable test mode
export TF_CPP_MIN_LOG_LEVEL=3      # Reduce TensorFlow logging
export PYTHONPATH="${PYTHONPATH}:src"  # Add src to Python path
```

## 🚦 Continuous Integration

### GitHub Actions / CI Pipeline

```yaml
# Example CI configuration
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      
      - name: Install dependencies
        run: |
          pip install -r requirements/dev.txt
      
      - name: Run linting
        run: python scripts/run_tests.py --lint
      
      - name: Run unit tests
        run: python scripts/run_tests.py --unit --coverage
      
      - name: Run integration tests
        run: python scripts/run_tests.py --integration
      
      - name: Upload coverage
        uses: codecov/codecov-action@v1
        with:
          file: ./coverage.xml
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 📈 Performance Benchmarks

The performance tests establish benchmarks for:

### ML Model Performance
- **Content-based inference**: <100ms for 100 predictions
- **Hybrid recommendations**: <200ms for 50 recommendations
- **Training convergence**: <30 seconds for small datasets
- **Memory usage**: <500MB for typical models

### API Performance
- **Property listing**: <500ms for 100 properties
- **Search queries**: <1s for complex searches
- **Recommendation generation**: <2s for 20 recommendations
- **Concurrent users**: >10 requests/second

### Database Performance
- **Property queries**: <100ms for filtered results
- **User lookups**: <50ms for user retrieval
- **Bulk operations**: <1s for 1000 records

## 🐛 Troubleshooting

### Common Issues

1. **TensorFlow warnings**: Set `TF_CPP_MIN_LOG_LEVEL=3`
2. **Import errors**: Ensure `PYTHONPATH` includes `src/`
3. **Database errors**: Check test database configuration
4. **Memory issues**: Run tests with smaller datasets
5. **Slow tests**: Use `-m "not slow"` to exclude slow tests

### Debug Mode

```bash
# Run specific test with debugging
pytest tests/unit/test_domain/test_property.py::TestProperty::test_creation -v -s

# Enter debugger on failure
pytest --pdb tests/unit/test_domain/test_property.py

# Run with verbose output
pytest -v -s tests/unit/test_domain/
```

### Test Isolation Issues

```bash
# Clear TensorFlow sessions between tests
pytest --forked tests/unit/test_infrastructure/

# Run tests in separate processes
pytest -n 1 tests/performance/
```

## 📋 Best Practices

### Writing Tests

1. **Use descriptive test names**: `test_property_creation_with_valid_data`
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Use appropriate markers**: Mark slow tests, ML tests, etc.
4. **Mock external dependencies**: Database, APIs, file system
5. **Use factories for test data**: Consistent, realistic data generation
6. **Test edge cases**: Empty inputs, large datasets, error conditions

### Test Organization

1. **Group related tests**: Use classes to group related test methods
2. **Separate concerns**: Unit vs integration vs performance tests
3. **Keep tests focused**: One concept per test method
4. **Use fixtures wisely**: Reuse setup code, but maintain test isolation
5. **Document complex tests**: Add docstrings for complex test scenarios

### Performance Testing

1. **Set realistic thresholds**: Based on production requirements
2. **Use consistent environments**: Control for external factors
3. **Measure what matters**: Focus on user-facing metrics
4. **Test under load**: Simulate realistic usage patterns
5. **Monitor trends**: Track performance over time

## 🤝 Contributing

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Add appropriate markers for test categorization
3. Include both positive and negative test cases
4. Update this README if adding new test categories
5. Ensure new tests pass in CI environment

### Test Review Checklist

- [ ] Tests follow naming conventions
- [ ] Appropriate markers are used
- [ ] Edge cases are covered
- [ ] Performance implications considered
- [ ] Documentation updated if needed
- [ ] Tests pass in isolation and in suite