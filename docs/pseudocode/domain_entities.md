# Domain Entities Pseudocode

## Property Entity

```pseudocode
CLASS Property:
    ATTRIBUTES:
        - id: UUID (unique identifier)
        - title: string (property title)
        - description: string (detailed description)
        - price: float (monthly rent price)
        - location: string (address/location)
        - bedrooms: integer (number of bedrooms)
        - bathrooms: float (number of bathrooms)
        - square_feet: optional integer (property size)
        - amenities: list of strings (property features)
        - contact_info: dictionary (contact details)
        - images: list of strings (image URLs)
        - scraped_at: datetime (when data was collected)
        - is_active: boolean (availability status)
        - property_type: string (apartment, house, etc.)
    
    METHODS:
        CREATE(title, description, price, location, bedrooms, bathrooms, ...):
            - Generate unique UUID
            - Set current timestamp for scraped_at
            - Initialize with provided data
            - Return new Property instance
        
        get_full_text():
            - Concatenate title, description, location, amenities
            - Return combined text for search indexing
        
        get_price_per_sqft():
            - IF square_feet exists AND > 0:
                - RETURN price / square_feet
            - ELSE:
                - RETURN null
        
        deactivate():
            - SET is_active = false
        
        activate():
            - SET is_active = true
```

## User Entity

```pseudocode
CLASS UserPreferences:
    ATTRIBUTES:
        - min_price: optional float
        - max_price: optional float
        - min_bedrooms: optional integer
        - max_bedrooms: optional integer
        - min_bathrooms: optional float
        - max_bathrooms: optional float
        - preferred_locations: list of strings
        - required_amenities: list of strings
        - property_types: list of strings

CLASS UserInteraction:
    ATTRIBUTES:
        - property_id: UUID
        - interaction_type: string (view, like, inquiry, save)
        - timestamp: datetime
        - duration_seconds: optional integer
    
    METHODS:
        CREATE(property_id, interaction_type, duration_seconds):
            - Set current timestamp
            - Return new UserInteraction instance

CLASS User:
    ATTRIBUTES:
        - id: UUID
        - email: string
        - preferences: UserPreferences
        - interactions: list of UserInteraction
        - created_at: datetime
        - is_active: boolean
    
    METHODS:
        CREATE(email, preferences):
            - Generate unique UUID
            - Set current timestamp for created_at
            - Initialize with provided data
            - Return new User instance
        
        add_interaction(interaction):
            - APPEND interaction to interactions list
        
        get_interaction_history(interaction_type):
            - IF interaction_type specified:
                - FILTER interactions by type
            - RETURN filtered or all interactions
        
        get_liked_properties():
            - FILTER interactions WHERE type = "like"
            - RETURN list of property IDs
        
        get_viewed_properties():
            - FILTER interactions WHERE type = "view"
            - RETURN list of property IDs
        
        update_preferences(new_preferences):
            - SET preferences = new_preferences
```

## SearchQuery Entity

```pseudocode
CLASS SearchFilters:
    ATTRIBUTES:
        - min_price: optional float
        - max_price: optional float
        - min_bedrooms: optional integer
        - max_bedrooms: optional integer
        - min_bathrooms: optional float
        - max_bathrooms: optional float
        - locations: list of strings
        - amenities: list of strings
        - property_types: list of strings
        - min_square_feet: optional integer
        - max_square_feet: optional integer

CLASS SearchQuery:
    ATTRIBUTES:
        - id: UUID
        - user_id: optional UUID
        - query_text: string
        - filters: SearchFilters
        - created_at: datetime
        - limit: integer (default 50)
        - offset: integer (default 0)
        - sort_by: string (relevance, price_asc, price_desc, etc.)
    
    METHODS:
        CREATE(query_text, user_id, filters, limit, offset, sort_by):
            - Generate unique UUID
            - Set current timestamp
            - Initialize with provided data
            - Return new SearchQuery instance
        
        get_normalized_query():
            - CONVERT query_text to lowercase
            - REMOVE leading/trailing whitespace
            - RETURN normalized string
        
        has_location_filter():
            - RETURN true IF filters.locations is not empty
        
        has_price_filter():
            - RETURN true IF min_price OR max_price is set
        
        has_size_filter():
            - RETURN true IF any bedroom/bathroom filters are set
        
        to_dict():
            - CONVERT all attributes to dictionary format
            - RETURN serializable dictionary
```

## Domain Entity Relationships

```
User (1) ──────── (many) UserInteraction
│                           │
│                           │
└── UserPreferences         └── references Property
                            
SearchQuery ──── SearchFilters

Property ──── (attributes and methods)
```

## Key Design Patterns

1. **Factory Pattern**: `create()` class methods for entity instantiation
2. **Value Objects**: UserPreferences, SearchFilters as immutable data containers
3. **Domain Services**: Business logic methods within entities
4. **Encapsulation**: Private validation and utility methods
5. **Immutability**: Entities created through factory methods with validation