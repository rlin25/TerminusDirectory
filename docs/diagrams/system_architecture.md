# System Architecture Diagrams

## Clean Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           PRESENTATION LAYER                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │   FastAPI       │  │   Streamlit     │  │      Jupyter                │ │
│  │   REST API      │  │   Demo UI       │  │      Notebooks              │ │
│  │                 │  │                 │  │                             │ │
│  │ - Search        │  │ - Interactive   │  │ - Model Training            │ │
│  │ - Recommendations│  │   Search       │  │ - Data Analysis             │ │
│  │ - User Mgmt     │  │ - Recommendations│  │ - Evaluation               │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                          APPLICATION LAYER                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │   Use Cases     │  │      DTOs       │  │         API                 │ │
│  │                 │  │                 │  │                             │ │
│  │ - SearchProps   │  │ - SearchDTO     │  │ - search_endpoints.py       │ │
│  │ - GetRecommend  │  │ - RecommendDTO  │  │ - recommendation_endpoints  │ │
│  │ - TrackInteract │  │ - UserDTO       │  │ - user_endpoints.py         │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                            DOMAIN LAYER                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │    Entities     │  │   Repositories  │  │        Services             │ │
│  │                 │  │   (Interfaces)  │  │                             │ │
│  │ - Property      │  │ - PropertyRepo  │  │ - SearchService             │ │
│  │ - User          │  │ - UserRepo      │  │ - RecommendationService     │ │
│  │ - SearchQuery   │  │ - ModelRepo     │  │ - UserService               │ │
│  │ - Interactions  │  │                 │  │                             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        INFRASTRUCTURE LAYER                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │  Data Access    │  │   ML Models     │  │      Monitoring             │ │
│  │                 │  │                 │  │                             │ │
│  │ - PostgreSQL    │  │ - Search Ranker │  │ - Logging                   │ │
│  │ - Redis Cache   │  │ - Collaborative │  │ - Metrics                   │ │
│  │ - Web Scrapers  │  │ - Content-Based │  │ - Health Checks             │ │
│  │ - File Storage  │  │ - Hybrid System │  │ - Error Tracking            │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │───▶│   FastAPI   │───▶│   Use Case  │───▶│   Domain    │
│  Request    │    │  Endpoint   │    │  Handler    │    │  Service    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                 │
                                                                 ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Response   │◀───│   JSON      │◀───│   Entity    │◀───│ Repository  │
│   (JSON)    │    │  Formatter  │    │   Models    │    │ (Postgres)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                 │
                                                                 ▼
                                                         ┌─────────────┐
                                                         │   Cache     │
                                                         │   (Redis)   │
                                                         └─────────────┘
```

## ML Pipeline Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Data      │───▶│   Data Pipeline  │───▶│  Feature Store  │
│ - Web Scraping  │    │ - Cleaning       │    │ - Embeddings    │
│ - User Events   │    │ - Validation     │    │ - Interactions  │
│ - Property Info │    │ - Transformation │    │ - Property Meta │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Predictions   │◀───│   ML Models      │◀───│   Training      │
│ - Rankings      │    │ - Search Ranker  │    │ - Model Updates │
│ - Recommendations│   │ - Collaborative  │    │ - Evaluation    │
│ - Similarities  │    │ - Content-Based  │    │ - Validation    │
└─────────────────┘    │ - Hybrid System  │    └─────────────────┘
                       └──────────────────┘
```

## Component Dependencies

```
┌────────────────────────────────────────────────────────────────┐
│                      Dependency Flow                           │
│                                                                │
│  Presentation ──────────▶ Application ──────────▶ Domain       │
│       │                        │                     │         │
│       │                        │                     ▼         │
│       │                        │              ┌─────────────┐  │
│       │                        │              │  Entities   │  │
│       │                        │              │ - Property  │  │
│       │                        │              │ - User      │  │
│       │                        │              │ - Search    │  │
│       │                        │              └─────────────┘  │
│       │                        │                     │         │
│       │                        ▼                     ▼         │
│       │                 ┌─────────────┐       ┌─────────────┐  │
│       │                 │ Use Cases   │       │  Services   │  │
│       │                 │ - Search    │       │ - Search    │  │
│       │                 │ - Recommend │       │ - Recommend │  │
│       │                 │ - Interact  │       │ - User      │  │
│       │                 └─────────────┘       └─────────────┘  │
│       │                        │                     │         │
│       ▼                        ▼                     ▼         │
│ ┌─────────────┐         ┌─────────────┐       ┌─────────────┐  │
│ │   API       │         │    DTOs     │       │ Repositories│  │
│ │ Endpoints   │         │ - Request   │       │ (Abstract)  │  │
│ │             │         │ - Response  │       │             │  │
│ └─────────────┘         └─────────────┘       └─────────────┘  │
│       │                                              │         │
│       ▼                                              ▼         │
│ ┌─────────────┐                               ┌─────────────┐  │
│ │Infrastructure│                              │Infrastructure  │
│ │ - Web Layer │                               │ - Data Layer│  │
│ │ - FastAPI   │                               │ - ML Models │  │
│ │ - CORS      │                               │ - Cache     │  │
│ └─────────────┘                               └─────────────┘  │
└────────────────────────────────────────────────────────────────┘
```
