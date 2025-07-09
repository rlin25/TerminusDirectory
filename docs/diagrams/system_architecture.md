# System Architecture Diagrams

## Clean Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            PRESENTATION LAYER                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │   FastAPI       │  │   Streamlit     │  │   Jupyter       │                │
│  │   REST API      │  │   Demo UI       │  │   Notebooks     │                │
│  │   (Port 8000)   │  │   (Port 8501)   │  │   (Port 8888)   │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            APPLICATION LAYER                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │   Use Cases     │  │      DTOs       │  │   API           │                │
│  │                 │  │                 │  │   Endpoints     │                │
│  │ • SearchProps   │  │ • SearchRequest │  │                 │                │
│  │ • GetRecommend  │  │ • UserProfile   │  │ • /search       │                │
│  │ • TrackUser     │  │ • PropertyView  │  │ • /recommend    │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DOMAIN LAYER                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │    Entities     │  │   Services      │  │   Repository    │                │
│  │                 │  │                 │  │   Interfaces    │                │
│  │ • Property      │  │ • SearchService │  │                 │                │
│  │ • User          │  │ • RecommendSvc  │  │ • PropertyRepo  │                │
│  │ • SearchQuery   │  │ • UserService   │  │ • UserRepo      │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          INFRASTRUCTURE LAYER                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │   Data Access   │  │   ML Models     │  │   External      │                │
│  │                 │  │                 │  │   Services      │                │
│  │ • PostgreSQL    │  │ • NLP Ranker    │  │                 │                │
│  │ • Redis Cache   │  │ • Hybrid Recomm │  │ • Web Scrapers  │                │
│  │ • Vector DB     │  │ • Embeddings    │  │ • Monitoring    │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Domain Model Relationships

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DOMAIN ENTITIES                                   │
│                                                                                 │
│  ┌─────────────────┐                    ┌─────────────────┐                    │
│  │      User       │                    │    Property     │                    │
│  │                 │                    │                 │                    │
│  │ • id: UUID      │                    │ • id: UUID      │                    │
│  │ • email         │                    │ • title         │                    │
│  │ • preferences   │                    │ • description   │                    │
│  │ • interactions  │                    │ • price         │                    │
│  │ • created_at    │                    │ • location      │                    │
│  │ • is_active     │                    │ • bedrooms      │                    │
│  └─────────────────┘                    │ • bathrooms     │                    │
│           │                             │ • amenities     │                    │
│           │                             │ • is_active     │                    │
│           │                             └─────────────────┘                    │
│           │                                       ▲                            │
│           │                                       │                            │
│           ▼                                       │                            │
│  ┌─────────────────┐                              │                            │
│  │ UserInteraction │                              │                            │
│  │                 │                              │                            │
│  │ • property_id   │──────────────────────────────┘                            │
│  │ • type          │                                                           │
│  │ • timestamp     │                                                           │
│  │ • duration      │                                                           │
│  └─────────────────┘                                                           │
│                                                                                 │
│  ┌─────────────────┐                    ┌─────────────────┐                    │
│  │  SearchQuery    │                    │ SearchFilters   │                    │
│  │                 │                    │                 │                    │
│  │ • id: UUID      │                    │ • min_price     │                    │
│  │ • user_id       │                    │ • max_price     │                    │
│  │ • query_text    │                    │ • bedrooms      │                    │
│  │ • filters       │────────────────────│ • bathrooms     │                    │
│  │ • created_at    │                    │ • locations     │                    │
│  │ • limit/offset  │                    │ • amenities     │                    │
│  │ • sort_by       │                    │ • property_types│                    │
│  └─────────────────┘                    └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Service Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DOMAIN SERVICES                                   │
│                                                                                 │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐           │
│  │ SearchService   │     │RecommendService │     │  UserService    │           │
│  │                 │     │                 │     │                 │           │
│  │ Dependencies:   │     │ Dependencies:   │     │ Dependencies:   │           │
│  │ • PropertyRepo  │     │ • PropertyRepo  │     │ • UserRepo      │           │
│  │ • ModelRepo     │     │ • UserRepo      │     │ • ModelRepo     │           │
│  │                 │     │ • ModelRepo     │     │                 │           │
│  │ Methods:        │     │                 │     │ Methods:        │           │
│  │ • search()      │     │ Methods:        │     │ • create()      │           │
│  │ • validate()    │     │ • recommend()   │     │ • update()      │           │
│  │ • suggestions() │     │ • similar()     │     │ • authenticate()│           │
│  │ • popular()     │     │ • record()      │     │ • preferences() │           │
│  └─────────────────┘     │ • explain()     │     └─────────────────┘           │
│                          └─────────────────┘                                   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                        REPOSITORY INTERFACES                               │ │
│  │                                                                             │ │
│  │ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              │ │
│  │ │ PropertyRepo    │ │   UserRepo      │ │  ModelRepo      │              │ │
│  │ │                 │ │                 │ │                 │              │ │
│  │ │ • create()      │ │ • create()      │ │ • save_model()  │              │ │
│  │ │ • get_by_id()   │ │ • get_by_id()   │ │ • load_model()  │              │ │
│  │ │ • search()      │ │ • interactions()│ │ • embeddings()  │              │ │
│  │ │ • similar()     │ │ • similar_users │ │ • cache()       │              │ │
│  │ │ • by_location() │ │ • by_email()    │ │ • predictions() │              │ │
│  │ └─────────────────┘ └─────────────────┘ └─────────────────┘              │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               DATA FLOW                                        │
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Client    │    │   FastAPI   │    │ Application │    │   Domain    │     │
│  │ Request     │───▶│ Endpoint    │───▶│ Use Case    │───▶│ Service     │     │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                                   │             │
│                                                                   ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Client    │    │   FastAPI   │    │ Application │    │ Repository  │     │
│  │ Response    │◀───│ Response    │◀───│ DTO         │◀───│ Data        │     │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                                   │             │
│                                                                   ▼             │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                      INFRASTRUCTURE LAYER                                  │ │
│  │                                                                             │ │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │ │
│  │ │ PostgreSQL  │ │   Redis     │ │ Vector DB   │ │ ML Models   │          │ │
│  │ │ Database    │ │ Cache       │ │ Embeddings  │ │ Serving     │          │ │
│  │ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘          │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Search Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SEARCH FLOW                                       │
│                                                                                 │
│  ┌─────────────┐                                                               │
│  │   User      │                                                               │
│  │ Search      │                                                               │
│  │ Request     │                                                               │
│  └─────────────┘                                                               │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                       │
│  │   Parse     │───▶│  Validate   │───▶│   Cache     │                       │
│  │   Query     │    │   Query     │    │   Check     │                       │
│  └─────────────┘    └─────────────┘    └─────────────┘                       │
│                                                │                               │
│                                                ▼                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                       │
│  │   Return    │◀───│  ML Ranking │◀───│  Database   │                       │
│  │   Results   │    │  (Optional) │    │   Search    │                       │
│  └─────────────┘    └─────────────┘    └─────────────┘                       │
│                                                │                               │
│                                                ▼                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                      SEARCH ALGORITHM                                      │ │
│  │                                                                             │ │
│  │ 1. Text Search:                                                             │ │
│  │    • Tokenize query                                                         │ │
│  │    • Full-text search on title, description                                │ │
│  │    • Semantic search using embeddings                                      │ │
│  │                                                                             │ │
│  │ 2. Apply Filters:                                                           │ │
│  │    • Price range                                                            │ │
│  │    • Location                                                               │ │
│  │    • Bedrooms/bathrooms                                                     │ │
│  │    • Amenities                                                              │ │
│  │                                                                             │ │
│  │ 3. Ranking:                                                                 │ │
│  │    • Relevance score                                                        │ │
│  │    • ML-based ranking                                                       │ │
│  │    • User preference boosting                                               │ │
│  │                                                                             │ │
│  │ 4. Pagination:                                                              │ │
│  │    • Apply limit/offset                                                     │ │
│  │    • Return total count                                                     │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Recommendation Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RECOMMENDATION FLOW                                  │
│                                                                                 │
│  ┌─────────────┐                                                               │
│  │   User      │                                                               │
│  │ Requests    │                                                               │
│  │ Recommend   │                                                               │
│  └─────────────┘                                                               │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                       │
│  │   Get User  │───▶│   Cache     │───▶│   Generate  │                       │
│  │   Profile   │    │   Check     │    │   Candidates│                       │
│  └─────────────┘    └─────────────┘    └─────────────┘                       │
│                                                │                               │
│                                                ▼                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                       │
│  │   Return    │◀───│  Deduplicate│◀───│   Hybrid    │                       │
│  │   Results   │    │  & Limit    │    │   Scoring   │                       │
│  └─────────────┘    └─────────────┘    └─────────────┘                       │
│                                                │                               │
│                                                ▼                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                   RECOMMENDATION STRATEGIES                                │ │
│  │                                                                             │ │
│  │ 1. Content-Based (50%):                                                     │ │
│  │    • Match user preferences                                                 │ │
│  │    • Location preferences                                                   │ │
│  │    • Price range matching                                                   │ │
│  │    • Amenity matching                                                       │ │
│  │                                                                             │ │
│  │ 2. Collaborative Filtering (30%):                                           │ │
│  │    • Find similar users                                                     │ │
│  │    • Get their liked properties                                             │ │
│  │    • Rank by popularity                                                     │ │
│  │                                                                             │ │
│  │ 3. Popular/Trending (20%):                                                  │ │
│  │    • Recently added properties                                              │ │
│  │    • High engagement properties                                             │ │
│  │    • Location-based trending                                                │ │
│  │                                                                             │ │
│  │ 4. Hybrid Scoring:                                                          │ │
│  │    • Weighted combination                                                   │ │
│  │    • Diversity boosting                                                     │ │
│  │    • Freshness factor                                                       │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Dependency Injection Pattern

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          DEPENDENCY INJECTION                                  │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                        ABSTRACT INTERFACES                                 │ │
│  │                                                                             │ │
│  │ PropertyRepository ◄────────────────────────────────────────────────────────┼─│
│  │ UserRepository     ◄────────────────────────────────────────────────────────┼─│
│  │ ModelRepository    ◄────────────────────────────────────────────────────────┼─│
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                      ▲                                         │
│                                      │                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                         DOMAIN SERVICES                                    │ │
│  │                                                                             │ │
│  │ SearchService(PropertyRepository, ModelRepository)                          │ │
│  │ RecommendService(PropertyRepository, UserRepository, ModelRepository)       │ │
│  │ UserService(UserRepository, ModelRepository)                                │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                      ▲                                         │
│                                      │                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                      CONCRETE IMPLEMENTATIONS                              │ │
│  │                                                                             │ │
│  │ PostgresPropertyRepository implements PropertyRepository                    │ │
│  │ PostgresUserRepository implements UserRepository                            │ │
│  │ RedisModelRepository implements ModelRepository                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

This architecture provides:
- **Separation of Concerns**: Each layer has specific responsibilities
- **Dependency Inversion**: High-level modules don't depend on low-level modules
- **Testability**: Easy to mock dependencies for testing
- **Scalability**: Can scale components independently
- **Maintainability**: Clear structure for adding new features