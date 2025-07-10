# Domain Entities Relationship Diagrams

## Entity Relationship Overview

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          DOMAIN ENTITIES                                  │
│                                                                           │
│  ┌─────────────────────┐              ┌─────────────────────┐             │
│  │      Property       │              │        User         │             │
│  │                     │              │                     │             │
│  │ + id: UUID          │              │ + id: UUID          │             │
│  │ + title: str        │              │ + email: str        │             │
│  │ + description: str  │              │ + preferences       │             │
│  │ + price: float      │              │ + interactions[]    │             │
│  │ + location: str     │              │ + created_at        │             │
│  │ + bedrooms: int     │              │ + is_active: bool   │             │
│  │ + bathrooms: float  │              │                     │             │
│  │ + square_feet: int  │              │                     │             │
│  │ + amenities: List   │              │                     │             │
│  │ + contact_info      │              │                     │             │
│  │ + images: List      │              │                     │             │
│  │ + scraped_at        │              │                     │             │
│  │ + is_active: bool   │              │                     │             │
│  │ + property_type     │              │                     │             │
│  └─────────────────────┘              └─────────────────────┘             │
│            │                                    │                         │
│            │                                    │                         │
│            ▼                                    ▼                         │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    UserInteraction                                  │  │
│  │                                                                     │  │
│  │  + property_id: UUID  ◄──────────┐                                  │  │
│  │  + interaction_type: str         │                                  │  │
│  │  + timestamp: datetime           │                                  │  │
│  │  + duration_seconds: int         │                                  │  │
│  │                                  │                                  │  │
│  │  Types:                          │                                  │  │
│  │  - "view"                        │                                  │  │
│  │  - "like"                        │                                  │  │
│  │  - "dislike"                     │                                  │  │
│  │  - "inquiry"                     │                                  │  │
│  │  - "save"                        │                                  │  │
│  │  - "share"                       │                                  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                     │                                     │
│                                     │                                     │
│            ┌────────────────────────┘                                     │
│            │                                                              │
│            ▼                                                              │
│  ┌─────────────────────┐              ┌─────────────────────┐             │
│  │   SearchQuery       │              │  UserPreferences    │             │
│  │                     │              │                     │             │
│  │ + id: UUID          │              │ + min_price: float  │             │
│  │ + user_id: UUID     │◄─────────────┤ + max_price: float  │             │
│  │ + query_text: str   │              │ + min_bedrooms: int │             │
│  │ + filters           │              │ + max_bedrooms: int │             │
│  │ + created_at        │              │ + min_bathrooms     │             │
│  │ + limit: int        │              │ + max_bathrooms     │             │
│  │ + offset: int       │              │ + locations: List   │             │
│  │ + sort_by: str      │              │ + amenities: List   │             │
│  └─────────────────────┘              │ + property_types    │             │
│            │                          └─────────────────────┘             │
│            │                                                              │
│            ▼                                                              │
│  ┌─────────────────────┐                                                  │
│  │   SearchFilters     │                                                  │
│  │                     │                                                  │
│  │ + min_price: float  │                                                  │
│  │ + max_price: float  │                                                  │
│  │ + min_bedrooms: int │                                                  │
│  │ + max_bedrooms: int │                                                  │
│  │ + min_bathrooms     │                                                  │
│  │ + max_bathrooms     │                                                  │
│  │ + locations: List   │                                                  │
│  │ + amenities: List   │                                                  │
│  │ + property_types    │                                                  │
│  │ + min_square_feet   │                                                  │
│  │ + max_square_feet   │                                                  │
│  └─────────────────────┘                                                  │
└───────────────────────────────────────────────────────────────────────────┘
```

## Property Entity Detailed View

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           PROPERTY ENTITY                                 │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                        Core Attributes                              │  │
│  │                                                                     │  │
│  │  id: UUID                    ── Unique identifier                   │  │
│  │  title: str                  ── "Luxury 2BR Downtown Apartment"     │  │
│  │  description: str            ── Full property description           │  │
│  │  price: float                ── Monthly rent amount                 │  │
│  │  location: str               ── "Downtown Manhattan, NY"            │  │
│  │  scraped_at: datetime        ── When data was collected             │  │
│  │  is_active: bool             ── Available for rent                  │  │
│  │  property_type: str          ── "apartment", "house", "condo"       │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                  │                                        │
│                                  ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    Physical Attributes                              │  │
│  │                                                                     │  │
│  │  bedrooms: int               ── Number of bedrooms                  │  │
│  │  bathrooms: float            ── Number of bathrooms (1.5, 2.0)      │  │
│  │  square_feet: Optional[int]  ── Floor area                          │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                  │                                        │
│                                  ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      Complex Attributes                             │  │
│  │                                                                     │  │
│  │  amenities: List[str]                                               │  │
│  │  ┌───────────────────────────────────────────────────────────────┐  │  │
│  │  │ ["gym", "pool", "parking", "pet-friendly",                    │  │  │
│  │  │  "in-unit-laundry", "dishwasher", "ac", "balcony"]            │  │  │
│  │  └───────────────────────────────────────────────────────────────┘  │  │
│  │                                                                     │  │
│  │  contact_info: Dict[str, str]                                       │  │
│  │  ┌───────────────────────────────────────────────────────────────┐  │  │
│  │  │ {                                                             │  │  │
│  │  │   "phone": "+1-555-123-4567",                                 │  │  │
│  │  │   "email": "leasing@property.com",                            │  │  │
│  │  │   "website": "https://property.com",                          │  │  │
│  │  │   "agent": "John Smith"                                       │  │  │
│  │  │ }                                                             │  │  │
│  │  └───────────────────────────────────────────────────────────────┘  │  │
│  │                                                                     │  │
│  │  images: List[str]                                                  │  │
│  │  ┌───────────────────────────────────────────────────────────────┐  │  │
│  │  │ ["https://images.com/prop1/living.jpg",                       │  │  │
│  │  │  "https://images.com/prop1/bedroom.jpg",                      │  │  │
│  │  │  "https://images.com/prop1/kitchen.jpg"]                      │  │  │
│  │  └───────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                  │                                        │
│                                  ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                        Methods                                      │  │
│  │                                                                     │  │
│  │  + create()                  ── Factory method                      │  │
│  │  + get_full_text()           ── Combined searchable text            │  │
│  │  + get_price_per_sqft()      ── Calculate price per sq ft           │  │
│  │  + activate()                ── Mark as available                   │  │
│  │  + deactivate()              ── Mark as unavailable                 │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
```

## User Entity Detailed View

```
┌───────────────────────────────────────────────────────────────────────────┐
│                             USER ENTITY                                   │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                        Core Attributes                              │  │
│  │                                                                     │  │
│  │  id: UUID                    ── Unique identifier                   │  │
│  │  email: str                  ── "user@example.com"                  │  │
│  │  created_at: datetime        ── Account creation time               │  │
│  │  is_active: bool             ── Account status                      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                  │                                        │
│                                  ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    User Preferences                                 │  │
│  │                                                                     │  │
│  │  preferences: UserPreferences                                       │  │
│  │  ┌───────────────────────────────────────────────────────────────┐  │  │
│  │  │  Price Range:                                                 │  │  │
│  │  │  + min_price: Optional[float] = 1000.0                        │  │  │
│  │  │  + max_price: Optional[float] = 3000.0                        │  │  │
│  │  │                                                               │  │  │
│  │  │  Size Preferences:                                            │  │  │
│  │  │  + min_bedrooms: Optional[int] = 1                            │  │  │
│  │  │  + max_bedrooms: Optional[int] = 3                            │  │  │
│  │  │  + min_bathrooms: Optional[float] = 1.0                       │  │  │
│  │  │  + max_bathrooms: Optional[float] = 2.0                       │  │  │
│  │  │                                                               │  │  │
│  │  │  Location Preferences:                                        │  │  │
│  │  │  + preferred_locations: List[str]                             │  │  │
│  │  │    ["Downtown", "Midtown", "Brooklyn Heights"]                │  │  │
│  │  │                                                               │  │  │
│  │  │  Feature Preferences:                                         │  │  │
│  │  │  + required_amenities: List[str]                              │  │  │
│  │  │    ["parking", "gym", "pet-friendly"]                         │  │  │
│  │  │  + property_types: List[str]                                  │  │  │
│  │  │    ["apartment", "condo"]                                     │  │  │
│  │  └───────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                  │                                        │
│                                  ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                   User Interactions                                 │  │
│  │                                                                     │  │
│  │  interactions: List[UserInteraction]                                │  │
│  │                                                                     │  │
│  │  Interaction Types:                                                 │  │
│  │  ┌───────────────────────────────────────────────────────────────┐  │  │
│  │  │  "view"     ── Property page visited                          │  │  │
│  │  │  "like"     ── User liked the property                        │  │  │
│  │  │  "dislike"  ── User disliked the property                     │  │  │
│  │  │  "inquiry"  ── User contacted about property                  │  │  │
│  │  │  "save"     ── Property saved to favorites                    │  │  │
│  │  │  "share"    ── Property shared with others                    │  │  │
│  │  └───────────────────────────────────────────────────────────────┘  │  │
│  │                                                                     │  │
│  │  Each interaction includes:                                         │  │
│  │  + property_id: UUID                                                │  │
│  │  + interaction_type: str                                            │  │
│  │  + timestamp: datetime                                              │  │
│  │  + duration_seconds: Optional[int]                                  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                  │                                        │
│                                  ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                          Methods                                    │  │
│  │                                                                     │  │
│  │  + create()                      ── Factory method                  │  │
│  │  + add_interaction()             ── Add new interaction             │  │
│  │  + get_interaction_history()     ── Filter by type                  │  │
│  │  + get_liked_properties()        ── Get liked property IDs          │  │
│  │  + get_viewed_properties()       ── Get viewed property IDs         │  │
│  │  + update_preferences()          ── Update user preferences         │  │
│  │  + activate() / deactivate()     ── Account management              │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
```

## Search Query Entity Flow

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          SEARCH QUERY FLOW                                │
│                                                                           │
│  User Input: "2 bedroom apartment downtown under $3000"                   │
│                              │                                            │
│                              ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      SearchQuery                                    │  │
│  │                                                                     │  │
│  │  id: UUID                        ── Unique query identifier         │  │
│  │  user_id: Optional[UUID]         ── Associated user (if logged in)  │  │
│  │  query_text: str                 ── "2 bedroom downtown under..."   │  │
│  │  created_at: datetime            ── Query timestamp                 │  │
│  │  limit: int = 50                 ── Results per page                │  │
│  │  offset: int = 0                 ── Pagination offset               │  │
│  │  sort_by: str = "relevance"      ── Sorting preference              │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                              │                                            │
│                              ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      SearchFilters                                  │  │
│  │                                                                     │  │
│  │  Extracted from query text and explicit filters:                    │  │
│  │                                                                     │  │
│  │  min_bedrooms: int = 2           ── From "2 bedroom"                │  │
│  │  max_bedrooms: int = 2                                              │  │
│  │  locations: List[str] = ["downtown"] ── From "downtown"             │  │
│  │  max_price: float = 3000.0       ── From "under $3000"              │  │
│  │                                                                     │  │
│  │  Optional filters:                                                  │  │
│  │  min_price: Optional[float]                                         │  │
│  │  min_bathrooms: Optional[float]                                     │  │
│  │  max_bathrooms: Optional[float]                                     │  │
│  │  amenities: List[str]                                               │  │
│  │  property_types: List[str]                                          │  │
│  │  min_square_feet: Optional[int]                                     │  │
│  │  max_square_feet: Optional[int]                                     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                              │                                            │
│                              ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      Query Processing                               │  │
│  │                                                                     │  │
│  │  Methods:                                                           │  │
│  │  + get_normalized_query()        ── Clean and normalize text        │  │
│  │  + has_location_filter()         ── Check if location specified     │  │
│  │  + has_price_filter()            ── Check if price range given      │  │
│  │  + has_size_filter()             ── Check if bed/bath specified     │  │
│  │  + to_dict()                     ── Serialize for API/cache         │  │
│  │                                                                     │  │
│  │  Sort Options:                                                      │  │
│  │  - "relevance"   ── ML-based ranking                                │  │
│  │  - "price_asc"   ── Lowest price first                              │  │
│  │  - "price_desc"  ── Highest price first                             │  │
│  │  - "date_new"    ── Newest listings first                           │  │
│  │  - "date_old"    ── Oldest listings first                           │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
```
