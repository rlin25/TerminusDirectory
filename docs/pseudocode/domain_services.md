# Domain Services Pseudocode

## SearchService

```pseudocode
CLASS SearchService:
    DEPENDENCIES:
        - property_repository: PropertyRepository
        - model_repository: ModelRepository
    
    CONSTRUCTOR(property_repository, model_repository):
        - Initialize dependencies
    
    ASYNC METHOD search_properties(query: SearchQuery) -> Tuple[List[Property], Integer]:
        """
        PURPOSE: Execute property search with ML ranking
        INPUT: SearchQuery with text, filters, pagination
        OUTPUT: (ranked properties, total count)
        
        ALGORITHM:
            1. VALIDATE query:
               - Check if query_text OR meaningful filters exist
               - Throw error if neither present
            
            2. EXECUTE search:
               - Call property_repository.search(query)
               - Get initial results from database
            
            3. APPLY ML ranking (if text query):
               - Generate cache key from query
               - Check for cached ML rankings
               - IF cached rankings exist:
                   - Reorder results using cached scores
               - ELSE:
                   - Use basic database ranking
                   - (ML ranking would be applied in production)
            
            4. RETURN (ranked_properties, total_count)
        """
    
    ASYNC METHOD get_search_suggestions(partial_query: String, limit: Integer) -> List[String]:
        """
        PURPOSE: Provide autocomplete suggestions
        INPUT: Partial query string, result limit
        OUTPUT: List of suggested completions
        
        ALGORITHM:
            1. Normalize partial query
            2. Query suggestion cache/index
            3. Return top matching suggestions
            4. (Currently returns empty - to be implemented)
        """
    
    ASYNC METHOD get_popular_searches(limit: Integer) -> List[String]:
        """
        PURPOSE: Get trending search terms
        INPUT: Result limit
        OUTPUT: List of popular search terms
        
        ALGORITHM:
            1. Query search analytics/logs
            2. Aggregate by frequency
            3. Return top searches
            4. (Currently returns hardcoded list)
        """
    
    PRIVATE METHOD _has_meaningful_filters(filters: SearchFilters) -> Boolean:
        """
        PURPOSE: Check if search has meaningful criteria
        INPUT: SearchFilters object
        OUTPUT: Boolean indicating if filters are meaningful
        
        ALGORITHM:
            Return true IF any of:
            - Price range specified
            - Bedroom/bathroom count specified
            - Location filters specified
            - Amenity filters specified
            - Property type filters specified
            - Square footage range specified
        """
    
    PRIVATE METHOD _apply_cached_rankings(properties: List[Property], rankings: Dictionary) -> List[Property]:
        """
        PURPOSE: Reorder properties using cached ML scores
        INPUT: Property list, ranking scores dictionary
        OUTPUT: Reordered property list
        
        ALGORITHM:
            1. Create property_id -> score mapping
            2. Sort properties by score (descending)
            3. Return sorted list
        """
    
    ASYNC METHOD validate_search_query(query: SearchQuery) -> List[String]:
        """
        PURPOSE: Validate search query and return errors
        INPUT: SearchQuery to validate
        OUTPUT: List of validation error messages
        
        ALGORITHM:
            1. Check query text length (<= 500 chars)
            2. Validate price filters:
               - Non-negative values
               - min_price <= max_price
            3. Validate bedroom/bathroom filters:
               - Non-negative values
               - min <= max
            4. Validate pagination:
               - limit between 1 and 100
               - offset >= 0
            5. Return list of error messages
        """
```

## RecommendationService

```pseudocode
CLASS RecommendationService:
    DEPENDENCIES:
        - property_repository: PropertyRepository
        - user_repository: UserRepository
        - model_repository: ModelRepository
    
    CONSTRUCTOR(property_repository, user_repository, model_repository):
        - Initialize dependencies
    
    ASYNC METHOD get_recommendations_for_user(user_id: UUID, limit: Integer) -> List[Dictionary]:
        """
        PURPOSE: Generate personalized property recommendations
        INPUT: User ID, result limit
        OUTPUT: List of recommendation dictionaries
        
        ALGORITHM:
            1. VALIDATE user exists:
               - Get user from repository
               - Throw error if not found
            
            2. CHECK cache:
               - Generate cache key
               - Return cached recommendations if available
            
            3. GET user context:
               - Get user interactions
               - Build set of interacted property IDs
            
            4. GENERATE recommendations:
               - Content-based (50% of limit):
                   - Based on user preferences
                   - Location preferences
                   - Price range matching
               - Collaborative filtering (50% of limit):
                   - If user has interaction history
                   - Based on similar users' preferences
               - Popular fallback:
                   - If not enough recommendations
                   - Add trending properties
            
            5. DEDUPLICATE and limit:
               - Remove duplicate property IDs
               - Limit to requested count
            
            6. CACHE results:
               - Store in cache with TTL
               - Return final recommendations
        """
    
    ASYNC METHOD get_similar_properties(property_id: UUID, limit: Integer) -> List[Dictionary]:
        """
        PURPOSE: Find properties similar to given property
        INPUT: Base property ID, result limit
        OUTPUT: List of similar property recommendations
        
        ALGORITHM:
            1. CHECK cache for similar properties
            2. GET base property details
            3. FIND similar properties via repository
            4. FORMAT as recommendation results:
               - Include similarity score
               - Add explanation text
            5. CACHE and return results
        """
    
    ASYNC METHOD record_user_interaction(user_id: UUID, property_id: UUID, interaction_type: String, duration_seconds: Optional[Integer]):
        """
        PURPOSE: Record user interaction and update recommendations
        INPUT: User ID, property ID, interaction type, optional duration
        OUTPUT: None (side effects only)
        
        ALGORITHM:
            1. VALIDATE interaction type:
               - Must be in [view, like, dislike, inquiry, save, share]
            2. CREATE interaction object:
               - Set current timestamp
               - Include duration if provided
            3. SAVE interaction:
               - Add to user's interaction history
            4. INVALIDATE cache:
               - Clear user's recommendation cache
               - Force fresh recommendations on next request
        """
    
    PRIVATE ASYNC METHOD _get_content_based_recommendations(user: User, excluded_property_ids: Set, limit: Integer) -> List[Dictionary]:
        """
        PURPOSE: Generate content-based recommendations
        INPUT: User object, excluded IDs, result limit
        OUTPUT: List of content-based recommendations
        
        ALGORITHM:
            1. EXTRACT user preferences:
               - Preferred locations
               - Price range
               - Bedroom/bathroom requirements
               - Required amenities
            
            2. QUERY matching properties:
               - Properties in preferred locations
               - Properties in price range
               - Properties with required amenities
            
            3. EXCLUDE already interacted properties
            
            4. FORMAT recommendations:
               - Add relevance score
               - Add explanation text
               - Limit to requested count
        """
    
    PRIVATE ASYNC METHOD _get_collaborative_recommendations(user_id: UUID, excluded_property_ids: Set, limit: Integer) -> List[Dictionary]:
        """
        PURPOSE: Generate collaborative filtering recommendations
        INPUT: User ID, excluded IDs, result limit
        OUTPUT: List of collaborative recommendations
        
        ALGORITHM:
            1. FIND similar users:
               - Query users with similar preferences
               - Get their interaction history
            
            2. COLLECT liked properties:
               - Get properties liked by similar users
               - Exclude already interacted properties
            
            3. RANK by popularity:
               - Properties liked by more similar users rank higher
            
            4. FORMAT recommendations:
               - Add relevance score
               - Add explanation text
               - Limit to requested count
        """
    
    PRIVATE ASYNC METHOD _get_popular_recommendations(excluded_property_ids: Set, limit: Integer) -> List[Dictionary]:
        """
        PURPOSE: Generate popular property recommendations as fallback
        INPUT: Excluded IDs, result limit
        OUTPUT: List of popular recommendations
        
        ALGORITHM:
            1. GET popular properties:
               - Query active properties
               - (Would include popularity metrics in production)
            
            2. EXCLUDE already interacted properties
            
            3. FORMAT recommendations:
               - Add default score
               - Add explanation text
               - Limit to requested count
        """
    
    ASYNC METHOD get_recommendation_explanation(user_id: UUID, property_id: UUID) -> Dictionary:
        """
        PURPOSE: Explain why property was recommended to user
        INPUT: User ID, property ID
        OUTPUT: Dictionary with explanation details
        
        ALGORITHM:
            1. GET user and property data
            2. ANALYZE matches:
               - Location preference match
               - Price range match
               - Bedroom/bathroom match
               - Amenity matches
            3. BUILD explanation:
               - List all matching factors
               - Provide overall explanation
            4. RETURN explanation dictionary
        """
```

## Service Layer Benefits

1. **Business Logic Encapsulation**: All business rules in one place
2. **Coordination**: Orchestrates multiple repositories and entities
3. **Validation**: Ensures data integrity and business rule compliance
4. **Caching**: Implements caching strategies for performance
5. **Error Handling**: Provides consistent error handling patterns

## Key Design Patterns

1. **Service Layer Pattern**: Encapsulates business logic
2. **Dependency Injection**: Services depend on repository abstractions
3. **Strategy Pattern**: Multiple recommendation strategies (content, collaborative, popular)
4. **Cache-Aside Pattern**: Check cache first, populate on miss
5. **Template Method Pattern**: Common validation and error handling flows

## Performance Considerations

- Caching frequently accessed data
- Batch operations where possible
- Async/await for non-blocking I/O
- Pagination for large result sets
- Cache invalidation strategies