# Recommendation Flow Diagrams

## Hybrid Recommendation Generation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HYBRID RECOMMENDATION FLOW                              │
│                                                                             │
│  User Request: Get recommendations for User ID 123                         │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Input Validation                                │   │
│  │                                                                     │   │
│  │  ✓ User exists in system                                           │   │
│  │  ✓ User is active                                                  │   │
│  │  ✓ Valid recommendation count (1-100)                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Cache Check                                      │   │
│  │                                                                     │   │
│  │  Cache Key: "user_recommendations:123:10"                          │   │
│  │                                                                     │   │
│  │  Cache Hit? ──────────┐                                           │   │
│  │                        │                                           │   │
│  │                        ▼                                           │   │
│  │              ┌─────────────────┐                                   │   │
│  │              │ Return Cached   │                                   │   │
│  │              │ Results         │                                   │   │
│  │              └─────────────────┘                                   │   │
│  │                                                                     │   │
│  │  Cache Miss ───────────┐                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          │                                                 │
│                          ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                 User Analysis                                       │   │
│  │                                                                     │   │
│  │  Get User Data:                                                     │   │
│  │  ├─ User preferences                                                │   │
│  │  ├─ Interaction history                                             │   │
│  │  ├─ Viewed properties                                               │   │
│  │  └─ Liked/disliked properties                                       │   │
│  │                                                                     │   │
│  │  Cold Start Check:                                                  │   │
│  │  Interactions < 5? ──────┐                                         │   │
│  │                           │                                         │   │
│  │                           ▼                                         │   │
│  │                  ┌─────────────────┐                               │   │
│  │                  │ Content-Based   │                               │   │
│  │                  │ Only Mode       │                               │   │
│  │                  └─────────────────┘                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                 Parallel Model Execution                           │   │
│  │                                                                     │   │
│  │  ┌───────────────────────┐    ┌─────────────────────────────────┐  │   │
│  │  │ Collaborative Filter  │    │    Content-Based Recommender   │  │   │
│  │  │                       │    │                                 │  │   │
│  │  │ Input:                │    │ Input:                          │  │   │
│  │  │ - User ID: 123        │    │ - User preferences              │  │   │
│  │  │ - User-item matrix    │    │ - Property features             │  │   │
│  │  │ - Item candidates     │    │ - Item candidates               │  │   │
│  │  │                       │    │                                 │  │   │
│  │  │ Process:              │    │ Process:                        │  │   │
│  │  │ 1. Get user embedding │    │ 1. Extract user profile         │  │   │
│  │  │ 2. Get item embeddings│    │ 2. Compute item features        │  │   │
│  │  │ 3. Neural prediction  │    │ 3. Feature matching             │  │   │
│  │  │ 4. Similarity scoring │    │ 4. Content similarity           │  │   │
│  │  │                       │    │                                 │  │   │
│  │  │ Output:               │    │ Output:                         │  │   │
│  │  │ CF Scores [0.0-1.0]   │    │ CB Scores [0.0-1.0]             │  │   │
│  │  └───────────────────────┘    └─────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Score Combination                               │   │
│  │                                                                     │   │
│  │  For each candidate property:                                       │   │
│  │                                                                     │   │
│  │  CF_Score = collaborative_filter.predict(user_id, item_id)         │   │
│  │  CB_Score = content_based.predict(user_profile, item_features)     │   │
│  │                                                                     │   │
│  │  Hybrid_Score = (cf_weight * CF_Score) + (cb_weight * CB_Score)    │   │
│  │                = (0.6 * CF_Score) + (0.4 * CB_Score)              │   │
│  │                                                                     │   │
│  │  Confidence = calculate_confidence(CF_Score, CB_Score, agreement)   │   │
│  │                                                                     │   │
│  │  Agreement Bonus:                                                   │   │
│  │  If |CF_Score - CB_Score| < 0.2: confidence += 0.1               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   Result Processing                                 │   │
│  │                                                                     │   │
│  │  1. Sort by Hybrid_Score (descending)                              │   │
│  │  2. Apply diversity filtering                                       │   │
│  │  3. Remove already interacted properties                           │   │
│  │  4. Generate explanations                                           │   │
│  │  5. Format results                                                  │   │
│  │                                                                     │   │
│  │  Result Format:                                                     │   │
│  │  {                                                                  │   │
│  │    "item_id": 456,                                                  │   │
│  │    "hybrid_score": 0.87,                                           │   │
│  │    "cf_score": 0.82,                                               │   │
│  │    "cb_score": 0.94,                                               │   │
│  │    "confidence": 0.91,                                             │   │
│  │    "explanation": "High match based on...",                        │   │
│  │    "reasoning": {                                                   │   │
│  │      "cf_contribution": 0.49,                                      │   │
│  │      "cb_contribution": 0.38,                                      │   │
│  │      "agreement_bonus": 0.1                                        │   │
│  │    }                                                                │   │
│  │  }                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Cache & Return                                │   │
│  │                                                                     │   │
│  │  1. Cache results with TTL (1 hour)                                │   │
│  │  2. Log recommendation metrics                                      │   │
│  │  3. Return top-N recommendations                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Cold Start Problem Handling

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       COLD START SCENARIOS                                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    New User (No History)                           │   │
│  │                                                                     │   │
│  │  User Profile:                                                      │   │
│  │  ├─ interactions: []                                                │   │
│  │  ├─ preferences: Basic preferences only                             │   │
│  │  └─ registration: Recent                                            │   │
│  │                                                                     │   │
│  │  Strategy:                                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │ 1. Content-Based Only                                         │ │   │
│  │  │    ├─ Use demographic defaults                                 │ │   │
│  │  │    ├─ Match user preferences exactly                           │ │   │
│  │  │    └─ Weight: CB = 1.0, CF = 0.0                             │ │   │
│  │  │                                                               │ │   │
│  │  │ 2. Popular Properties Fallback                                │ │   │
│  │  │    ├─ Show trending properties                                 │ │   │
│  │  │    ├─ High-rated in user's area                               │ │   │
│  │  │    └─ Recently listed properties                              │ │   │
│  │  │                                                               │ │   │
│  │  │ 3. Preference-Based Filtering                                 │ │   │
│  │  │    ├─ Price range matching                                    │ │   │
│  │  │    ├─ Location preferences                                    │ │   │
│  │  │    └─ Required amenities                                      │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                         │
│                                  ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                  Few Interactions (1-4)                            │   │
│  │                                                                     │   │
│  │  User Profile:                                                      │   │
│  │  ├─ interactions: [view, view, like]                               │   │
│  │  ├─ sparse_data: High uncertainty                                  │   │
│  │  └─ preferences: Partially learned                                 │   │
│  │                                                                     │   │
│  │  Strategy:                                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │ 1. Weighted Hybrid Approach                                   │ │   │
│  │  │    ├─ CB Weight: 0.8 (High confidence)                       │ │   │
│  │  │    ├─ CF Weight: 0.2 (Low confidence)                        │ │   │
│  │  │    └─ Gradual transition as data grows                        │ │   │
│  │  │                                                               │ │   │
│  │  │ 2. Similar User Bootstrapping                                 │ │   │
│  │  │    ├─ Find users with similar preferences                     │ │   │
│  │  │    ├─ Use their liked properties                              │ │   │
│  │  │    └─ Lower confidence scores                                 │ │   │
│  │  │                                                               │ │   │
│  │  │ 3. Exploration Encouragement                                  │ │   │
│  │  │    ├─ Add diverse recommendations                             │ │   │
│  │  │    ├─ Different price ranges                                  │ │   │
│  │  │    └─ Various neighborhoods                                   │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                         │
│                                  ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                  New Property (No Interactions)                    │   │
│  │                                                                     │   │
│  │  Property Profile:                                                  │   │
│  │  ├─ interactions: []                                                │   │
│  │  ├─ content_features: Full feature set                             │   │
│  │  └─ listing: Recently added                                        │   │
│  │                                                                     │   │
│  │  Strategy:                                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │ 1. Content-Based Immediately Available                       │ │   │
│  │  │    ├─ Property features fully known                           │ │   │
│  │  │    ├─ Match to user preferences                               │ │   │
│  │  │    └─ No CF data needed                                       │ │   │
│  │  │                                                               │ │   │
│  │  │ 2. Similar Property Analysis                                  │ │   │
│  │  │    ├─ Find properties with similar features                   │ │   │
│  │  │    ├─ Inherit interaction patterns                            │ │   │
│  │  │    └─ Bootstrap initial popularity                            │ │   │
│  │  │                                                               │ │   │
│  │  │ 3. Market Position Analysis                                   │ │   │
│  │  │    ├─ Price competitiveness                                   │ │   │
│  │  │    ├─ Feature uniqueness                                      │ │   │
│  │  │    └─ Location desirability                                   │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## User Interaction Tracking Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     USER INTERACTION TRACKING                              │
│                                                                             │
│  User Action: Views property page                                          │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   Interaction Capture                              │   │
│  │                                                                     │   │
│  │  Frontend Event:                                                    │   │
│  │  ├─ Event Type: "property_view"                                     │   │
│  │  ├─ Property ID: 789                                                │   │
│  │  ├─ User ID: 123                                                    │   │
│  │  ├─ Timestamp: 2024-01-15T14:30:00Z                                │   │
│  │  ├─ Session ID: abc123                                              │   │
│  │  ├─ Duration: 45 seconds                                            │   │
│  │  └─ Referrer: "search_results"                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   Validation & Enrichment                          │   │
│  │                                                                     │   │
│  │  Validation:                                                        │   │
│  │  ├─ ✓ Valid interaction type                                        │   │
│  │  ├─ ✓ User exists and is active                                     │   │
│  │  ├─ ✓ Property exists and is active                                 │   │
│  │  └─ ✓ Reasonable duration (1s - 1 hour)                            │   │
│  │                                                                     │   │
│  │  Enrichment:                                                        │   │
│  │  ├─ Add interaction weight (view=1, like=5, inquiry=10)            │   │
│  │  ├─ Calculate engagement score                                      │   │
│  │  ├─ Add device/browser context                                      │   │
│  │  └─ Determine interaction quality                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Interaction Storage                           │   │
│  │                                                                     │   │
│  │  UserInteraction Object:                                            │   │
│  │  {                                                                  │   │
│  │    "property_id": 789,                                              │   │
│  │    "interaction_type": "view",                                      │   │
│  │    "timestamp": "2024-01-15T14:30:00Z",                            │   │
│  │    "duration_seconds": 45,                                          │   │
│  │    "weight": 1,                                                     │   │
│  │    "engagement_score": 0.7,                                         │   │
│  │    "context": {                                                     │   │
│  │      "session_id": "abc123",                                        │   │
│  │      "referrer": "search_results",                                  │   │
│  │      "device_type": "desktop"                                       │   │
│  │    }                                                                │   │
│  │  }                                                                  │   │
│  │                                                                     │   │
│  │  Storage Actions:                                                   │   │
│  │  ├─ Add to user.interactions[]                                      │   │
│  │  ├─ Update user activity timestamp                                  │   │
│  │  ├─ Store in database                                               │   │
│  │  └─ Update real-time aggregates                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Model Updates                                    │   │
│  │                                                                     │   │
│  │  Immediate Updates:                                                  │   │
│  │  ├─ Invalidate user recommendation cache                            │   │
│  │  ├─ Update user-item interaction matrix                             │   │
│  │  ├─ Trigger real-time personalization                               │   │
│  │  └─ Update property popularity scores                               │   │
│  │                                                                     │   │
│  │  Batch Updates (Hourly/Daily):                                      │   │
│  │  ├─ Retrain collaborative filtering model                           │   │
│  │  ├─ Update content-based user profiles                              │   │
│  │  ├─ Refresh similarity matrices                                     │   │
│  │  └─ Update recommendation explanations                              │   │
│  │                                                                     │   │
│  │  Analytics Updates:                                                  │   │
│  │  ├─ User engagement metrics                                         │   │
│  │  ├─ Property performance metrics                                    │   │
│  │  ├─ Recommendation effectiveness                                    │   │
│  │  └─ A/B testing data collection                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   Response & Feedback                              │   │
│  │                                                                     │   │
│  │  API Response:                                                      │   │
│  │  {                                                                  │   │
│  │    "status": "success",                                             │   │
│  │    "interaction_id": "int_456",                                     │   │
│  │    "message": "Interaction recorded",                               │   │
│  │    "recommendations_updated": true,                                 │   │
│  │    "next_actions": [                                                │   │
│  │      "show_similar_properties",                                     │   │
│  │      "update_search_preferences"                                    │   │
│  │    ]                                                                │   │
│  │  }                                                                  │   │
│  │                                                                     │   │
│  │  Frontend Actions:                                                  │   │
│  │  ├─ Update user interface                                           │   │
│  │  ├─ Show personalized suggestions                                   │   │
│  │  ├─ Trigger recommendation refresh                                  │   │
│  │  └─ Update user preferences UI                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Explanation Generation Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RECOMMENDATION EXPLANATION FLOW                         │
│                                                                             │
│  Request: Explain why Property 789 was recommended to User 123             │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Gather Context Data                             │   │
│  │                                                                     │   │
│  │  User Context:                                                      │   │
│  │  ├─ User preferences                                                │   │
│  │  ├─ Interaction history                                             │   │
│  │  ├─ Liked/disliked properties                                       │   │
│  │  └─ Search patterns                                                 │   │
│  │                                                                     │   │
│  │  Property Context:                                                  │   │
│  │  ├─ Property features                                               │   │
│  │  ├─ Similar properties                                              │   │
│  │  ├─ Popularity metrics                                              │   │
│  │  └─ Market position                                                 │   │
│  │                                                                     │   │
│  │  Model Context:                                                     │   │
│  │  ├─ CF score and reasoning                                          │   │
│  │  ├─ CB score and features                                           │   │
│  │  ├─ Hybrid combination weights                                      │   │
│  │  └─ Confidence calculations                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                 Feature Importance Analysis                         │   │
│  │                                                                     │   │
│  │  Content-Based Contributions:                                       │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │ Location Match:        85% (Downtown preference)              │ │   │
│  │  │ Price Range:           92% (Within $2000-3000 budget)         │ │   │
│  │  │ Bedroom Count:         100% (Exactly 2 bedrooms wanted)       │ │   │
│  │  │ Amenities:             78% (Has gym, parking from wishlist)   │ │   │
│  │  │ Property Type:         100% (Apartment preference match)      │ │   │
│  │  │ Square Footage:        67% (850 sqft vs 800-1000 preference) │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  │  Collaborative Filtering Contributions:                             │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │ Similar Users:         [User 456, User 789, User 234]         │ │   │
│  │  │ Shared Preferences:    Downtown, 2BR, Gym amenities          │ │   │
│  │  │ Interaction Patterns:  High engagement with similar props     │ │   │
│  │  │ Prediction Confidence: 87% based on user embedding similarity │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                  Multi-Level Explanation Generation                 │   │
│  │                                                                     │   │
│  │  Simple Explanation (User-Facing):                                  │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │ "This property matches your preferences for a 2-bedroom       │ │   │
│  │  │ apartment in Downtown with gym access, and is within your     │ │   │
│  │  │ budget of $2000-3000. Users with similar preferences have     │ │   │
│  │  │ also shown high interest in this property."                   │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  │  Detailed Explanation (Power Users):                                │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │ Content Matching (Weight: 40%):                               │ │   │
│  │  │ • Perfect bedroom match (2BR) - High impact                   │ │   │
│  │  │ • Location preference (Downtown) - 85% match                  │ │   │
│  │  │ • Price within range ($2,800 vs $2000-3000) - Good fit       │ │   │
│  │  │ • Desired amenities present (gym, parking) - 78% coverage    │ │   │
│  │  │                                                               │ │   │
│  │  │ Collaborative Filtering (Weight: 60%):                        │ │   │
│  │  │ • 3 similar users also liked this property                    │ │   │
│  │  │ • Strong pattern match based on interaction history           │ │   │
│  │  │ • 87% confidence from user similarity analysis                │ │   │
│  │  │                                                               │ │   │
│  │  │ Final Score: 0.4×0.89 + 0.6×0.87 = 0.88 (High confidence)   │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  │  Technical Explanation (Debug/Admin):                               │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │ Model Scores:                                                 │ │   │
│  │  │ • Content-Based Score: 0.89                                   │ │   │
│  │  │   - Location embedding similarity: 0.85                      │ │   │
│  │  │   - Price feature match: 0.92                                 │ │   │
│  │  │   - Amenity TF-IDF cosine similarity: 0.78                   │ │   │
│  │  │   - Bedroom/bathroom exact match: 1.0                        │ │   │
│  │  │                                                               │ │   │
│  │  │ • Collaborative Score: 0.87                                   │ │   │
│  │  │   - User embedding (50-dim): [0.2, -0.1, 0.8, ...]          │ │   │
│  │  │   - Item embedding (50-dim): [0.1, -0.2, 0.9, ...]          │ │   │
│  │  │   - Neural network prediction: 0.87                          │ │   │
│  │  │   - Top similar users: [456, 789, 234] with similarities     │ │   │
│  │  │     [0.91, 0.89, 0.85]                                       │ │   │
│  │  │                                                               │ │   │
│  │  │ • Hybrid Combination: 0.6×0.87 + 0.4×0.89 = 0.878           │ │   │
│  │  │ • Confidence Boost: 0.1 (models agree within 0.02)          │ │   │
│  │  │ • Final Score: 0.878 + 0.1 = 0.978                          │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```