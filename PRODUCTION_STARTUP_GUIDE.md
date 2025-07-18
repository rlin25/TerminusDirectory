# Production Startup Guide

## Quick Start with Docker

1. **Start Required Services:**
   ```bash
   # Start PostgreSQL and Redis
   docker-compose up -d postgres redis
   
   # Wait for services to be ready (about 30 seconds)
   sleep 30
   ```

2. **Run Database Setup:**
   ```bash
   # Create database and run migrations
   python3 migrations/run_migrations.py
   ```

3. **Start Production Application:**
   ```bash
   # Option 1: Direct Python execution
   python3 main_production.py
   
   # Option 2: Using the startup script
   ./start_production.sh
   
   # Option 3: Using Uvicorn directly
   uvicorn main_production:app --host 0.0.0.0 --port 8000 --workers 4
   ```

## Verify Installation

1. **Health Check:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **API Documentation:**
   - Open http://localhost:8000/docs in your browser

## Troubleshooting

1. **Database Connection Issues:**
   - Check if PostgreSQL is running: `docker ps | grep postgres`
   - Test connection: `pg_isready -h localhost -p 5432 -U rental_ml_user`

2. **Redis Connection Issues:**
   - Check if Redis is running: `docker ps | grep redis`
   - Test connection: `redis-cli -h localhost -p 6379 ping`

3. **Import Errors:**
   - Install dependencies: `pip install -r requirements/base.txt`
   - For full production setup: `pip install -r requirements/prod.txt`

## Next Steps

1. **Add Real Data:**
   - Run property scrapers: `python3 -m src.data.scraping.production_scraping_orchestrator`
   - Or load sample data: `python3 database/seeds/production_data_seeder.py`

2. **Enable Authentication:**
   - Review security configuration in `.env.production`
   - Integrate authentication middleware

3. **Set Up Monitoring:**
   - Start Prometheus: `docker-compose up -d prometheus`
   - Start Grafana: `docker-compose up -d grafana`
