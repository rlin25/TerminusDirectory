# Search Ranking Flow Diagrams

## Search Query Processing Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       SEARCH PROCESSING PIPELINE                           │
│                                                                            │
│  User Input: "2 bedroom apartment downtown with gym under $3000"           │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Query Preprocessing                              │   │
│  │                                                                     │   │
│  │  Input Validation:                                                  │   │
│  │  ├─ ✓ Query length (1-500 characters)                               │   │
│  │  ├─ ✓ No malicious content                                          │   │
│  │  ├─ ✓ Encoding validation (UTF-8)                                   │   │
│  │  └─ ✓ Rate limiting check                                           │   │
│  │                                                                     │   │
│  │  Text Normalization:                                                │   │
│  │  ├─ Lowercase conversion                                            │   │
│  │  ├─ Remove extra whitespace                                         │   │
│  │  ├─ Handle special characters                                       │   │
│  │  └─ Basic typo correction                                           │   │
│  │                                                                     │   │
│  │  Result: "2 bedroom apartment downtown with gym under $3000"        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   Intent & Filter Extraction                        │   │
│  │                                                                     │   │
│  │  Named Entity Recognition:                                          │   │
│  │  ├─ Bedrooms: "2" → min_bedrooms=2, max_bedrooms=2                  │   │
│  │  ├─ Property Type: "apartment" → property_types=["apartment"]       │   │
│  │  ├─ Location: "downtown" → locations=["downtown"]                   │   │
│  │  ├─ Amenities: "gym" → amenities=["gym"]                            │   │
│  │  └─ Price: "under $3000" → max_price=3000.0                         │   │
│  │                                                                     │   │
│  │  Query Intent Classification:                                       │   │
│  │  ├─ Primary Intent: "property_search"                               │   │
│  │  ├─ Confidence: 0.98                                                │   │
│  │  └─ Sub-intents: ["filter_by_price", "filter_by_amenities"]         │   │
│  │                                                                     │   │
│  │  Constructed SearchQuery:                                           │   │
│  │  {                                                                  │   │
│  │    "query_text": "2 bedroom apartment downtown with gym under...",  │   │
│  │    "filters": {                                                     │   │
│  │      "min_bedrooms": 2,                                             │   │
│  │      "max_bedrooms": 2,                                             │   │
│  │      "property_types": ["apartment"],                               │   │
│  │      "locations": ["downtown"],                                     │   │
│  │      "amenities": ["gym"],                                          │   │
│  │      "max_price": 3000.0                                            │   │
│  │    }                                                                │   │
│  │  }                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   Candidate Retrieval                               │   │
│  │                                                                     │   │
│  │  Database Query Construction:                                       │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │ SELECT * FROM properties                                      │  │   │
│  │  │ WHERE                                                         │  │   │
│  │  │   is_active = true                                            │  │   │
│  │  │   AND bedrooms = 2                                            │  │   │
│  │  │   AND property_type = 'apartment'                             │  │   │
│  │  │   AND price <= 3000.0                                         │  │   │
│  │  │   AND (                                                       │  │   │
│  │  │     location ILIKE '%downtown%'                               │  │   │
│  │  │     OR city ILIKE '%downtown%'                                │  │   │
│  │  │   )                                                           │  │   │
│  │  │   AND amenities @> '["gym"]'                                  │  │   │
│  │  │ ORDER BY created_at DESC                                      │  │   │
│  │  │ LIMIT 1000;                                                   │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                     │   │
│  │  Index Optimization:                                                │   │
│  │  ├─ Use location GIN index                                          │   │
│  │  ├─ Use price range B-tree index                                    │   │
│  │  ├─ Use amenities GIN index                                         │   │
│  │  └─ Composite index on (bedrooms, property_type, is_active)         │   │
│  │                                                                     │   │
│  │  Result: 247 candidate properties                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      ML-Based Ranking                               │   │
│  │                                                                     │   │
│  │  NLP Search Ranker Processing:                                      │   │
│  │                                                                     │   │
│  │  Query Encoding:                                                    │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │ Input: "2 bedroom apartment downtown with gym under $3000"    │  │   │
│  │  │        │                                                      │  │   │
│  │  │        ▼                                                      │  │   │
│  │  │ Tokenizer: [CLS] 2 bedroom apartment downtown with gym        │  │   │
│  │  │           under $ 3000 [SEP]                                  │  │   │
│  │  │        │                                                      │  │   │
│  │  │        ▼                                                      │  │   │
│  │  │ Transformer Encoder (sentence-transformers/all-MiniLM-L6-v2)  │  │   │
│  │  │        │                                                      │  │   │
│  │  │        ▼                                                      │  │   │
│  │  │ Query Embedding: [0.1, -0.3, 0.8, ..., 0.2] (384 dims)        │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                     │   │
│  │  Property Encoding (Batch Processing):                              │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │ For each of 247 properties:                                   │  │   │
│  │  │                                                               │  │   │
│  │  │ Property 1: "Luxury 2BR Downtown Apt with Gym & Pool"         │  │   │
│  │  │ Property 2: "Modern 2 Bedroom in Financial District"          │  │   │
│  │  │ Property 3: "Spacious Apartment Near Central Park"            │  │   │
│  │  │ ...                                                           │  │   │
│  │  │        │                                                      │  │   │
│  │  │        ▼                                                      │  │   │
│  │  │ Combined Text: title + description + location + amenities     │  │   │
│  │  │        │                                                      │  │   │
│  │  │        ▼                                                      │  │   │
│  │  │ Batch Encoding → Property Embeddings [384 dims each]          │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                     │   │
│  │  Ranking Score Calculation:                                         │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │ Neural Ranking Model:                                         │  │   │
│  │  │                                                               │  │   │
│  │  │ Input: Query Embedding [384] + Property Embedding [384]       │  │   │
│  │  │        │                                                      │  │   │
│  │  │        ▼                                                      │  │   │
│  │  │ Concatenate → [768 dims]                                      │  │   │
│  │  │        │                                                      │  │   │
│  │  │        ▼                                                      │  │   │
│  │  │ Dense(512, ReLU) → Dropout(0.3)                               │  │   │
│  │  │        │                                                      │  │   │
│  │  │        ▼                                                      │  │   │
│  │  │ Dense(256, ReLU) → Dropout(0.3)                               │  │   │
│  │  │        │                                                      │  │   │
│  │  │        ▼                                                      │  │   │
│  │  │ Dense(128, ReLU) → Dropout(0.2)                               │  │   │
│  │  │        │                                                      │  │   │
│  │  │        ▼                                                      │  │   │
│  │  │ Dense(1, Sigmoid) → Relevance Score [0.0-1.0]                 │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Result Ranking & Filtering                       │   │
│  │                                                                     │   │
│  │  Ranking Results:                                                   │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │ Property 1: Score 0.95 - "Luxury 2BR Downtown with Gym"       │  │   │
│  │  │ Property 3: Score 0.89 - "Modern Downtown Apartment + Gym"    │  │   │
│  │  │ Property 7: Score 0.84 - "2BR Near Downtown, Fitness Center"  │  │   │
│  │  │ Property 2: Score 0.78 - "2 Bedroom Financial District"       │  │   │
│  │  │ Property 5: Score 0.72 - "Spacious 2BR Midtown"               │  │   │
│  │  │ ...                                                           │  │   │
│  │  │ Property 45: Score 0.23 - "1BR Studio in Downtown"            │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                     │   │
│  │  Post-Processing:                                                   │   │
│  │  ├─ Remove duplicates (same property, different listings)           │   │
│  │  ├─ Apply business rules (featured listings boost)                  │   │
│  │  ├─ Diversity filtering (don't show all from same building)         │   │
│  │  └─ Apply pagination (offset=0, limit=50)                           │   │
│  │                                                                     │   │
│  │  Final Ranking Features:                                            │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │ For each result, calculate:                                   │  │   │
│  │  │ • Relevance Score (from ML model)                             │  │   │
│  │  │ • Text Match Quality (keyword coverage)                       │  │   │
│  │  │ • Filter Compliance (how well it matches filters)             │  │   │
│  │  │ • Freshness Score (how recently listed)                       │  │   │
│  │  │ • Popularity Score (views, likes, inquiries)                  │  │   │
│  │  │ • Price Competitiveness (vs similar properties)               │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

## NLP Text Encoding Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NLP TEXT ENCODING PROCESS                            │
│                                                                             │
│  Input Text: "Luxury 2BR downtown apartment with gym, pool, parking"        │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Text Preprocessing                             │    │
│  │                                                                     │    │
│  │  Text Cleaning:                                                     │    │
│  │  ├─ Unicode normalization (NFKD)                                    │    │
│  │  ├─ Remove excessive punctuation                                    │    │
│  │  ├─ Normalize whitespace                                            │    │
│  │  └─ Handle special characters                                       │    │
│  │                                                                     │    │
│  │  Text Enhancement:                                                  │    │
│  │  ├─ Expand abbreviations (2BR → 2 bedroom)                          │    │
│  │  ├─ Normalize location names                                        │    │
│  │  ├─ Standardize amenity terms                                       │    │
│  │  └─ Add context (apartment → rental apartment)                      │    │
│  │                                                                     │    │
│  │  Result: "Luxury 2 bedroom downtown rental apartment with gym,      │    │
│  │           swimming pool, parking"                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       Tokenization                                  │    │
│  │                                                                     │    │
│  │  Transformer Tokenizer (all-MiniLM-L6-v2):                          │    │
│  │                                                                     │    │
│  │  Input: "Luxury 2 bedroom downtown rental apartment with gym,       │    │
│  │          swimming pool, parking"                                    │    │
│  │                              │                                      │    │
│  │                              ▼                                      │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │ Subword Tokenization (WordPiece):                             │  │    │
│  │  │                                                               │  │    │
│  │  │ Token IDs:                                                    │  │    │
│  │  │ [CLS]    → 101                                                │  │    │
│  │  │ luxury   → 6350                                               │  │    │
│  │  │ 2        → 1016                                               │  │    │
│  │  │ bedroom  → 5010                                               │  │    │
│  │  │ downtown → 7292                                               │  │    │
│  │  │ rental   → 7893                                               │  │    │
│  │  │ apartment→ 4598                                               │  │    │
│  │  │ with     → 2007                                               │  │    │
│  │  │ gym      → 9345                                               │  │    │
│  │  │ ,        → 1010                                               │  │    │
│  │  │ swimming → 5689                                               │  │    │
│  │  │ pool     → 4770                                               │  │    │
│  │  │ ,        → 1010                                               │  │    │
│  │  │ parking  → 5581                                               │  │    │
│  │  │ [SEP]    → 102                                                │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                                                                     │    │
│  │  Attention Masks & Position Encoding:                               │    │
│  │  ├─ Attention mask: [1,1,1,1,1,1,1,1,1,1,1,1,1,1] (all tokens)      │    │
│  │  ├─ Position IDs: [0,1,2,3,4,5,6,7,8,9,10,11,12,13]                 │    │
│  │  └─ Token type IDs: [0,0,0,0,0,0,0,0,0,0,0,0,0,0] (single sequence) │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Transformer Encoding                             │    │
│  │                                                                     │    │
│  │  BERT-based Encoder (12 layers):                                    │    │
│  │                                                                     │    │
│  │  Layer 0 (Embedding):                                               │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │ Token Embeddings [768 dims per token]                         │  │    │
│  │  │ + Position Embeddings [768 dims]                              │  │    │
│  │  │ + Segment Embeddings [768 dims]                               │  │    │
│  │  │ = Input Representations [14 tokens × 768 dims]                │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                              │                                      │    │
│  │                              ▼                                      │    │
│  │  Layers 1-12 (Transformer Blocks):                                  │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │ For each layer:                                               │  │    │
│  │  │                                                               │  │    │
│  │  │ Multi-Head Self-Attention:                                    │  │    │
│  │  │ ├─ Query, Key, Value projections                              │  │    │
│  │  │ ├─ 12 attention heads (64 dims each)                          │  │    │
│  │  │ ├─ Attention weights calculation                              │  │    │
│  │  │ └─ Context-aware representations                              │  │    │
│  │  │                                                               │  │    │
│  │  │ Feed-Forward Network:                                         │  │    │
│  │  │ ├─ Linear(768 → 3072) + GELU                                  │  │    │
│  │  │ ├─ Linear(3072 → 768)                                         │  │    │
│  │  │ └─ Residual connection + Layer norm                           │  │    │
│  │  │                                                               │  │    │
│  │  │ Output: Enhanced token representations                        │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                                                                     │    │
│  │  Final Layer Output: [14 tokens × 768 dims]                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Pooling Strategy                               │    │
│  │                                                                     │    │
│  │  CLS Token Pooling (Default):                                       │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │ Extract [CLS] token representation (position 0)               │  │    │
│  │  │ Shape: [768 dims]                                             │  │    │
│  │  │                                                               │  │    │
│  │  │ CLS Token captures:                                           │  │    │
│  │  │ ├─ Global sequence information                                │  │    │
│  │  │ ├─ Semantic relationships between tokens                      │  │    │
│  │  │ ├─ Context-aware meaning                                      │  │    │
│  │  │ └─ Task-specific learned representations                      │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                                                                     │    │
│  │  Alternative Pooling Strategies:                                    │    │
│  │  ├─ Mean Pooling: Average all token embeddings                      │    │
│  │  ├─ Max Pooling: Element-wise maximum                               │    │
│  │  └─ Attention-weighted: Learn attention weights for pooling         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Normalization                                  │    │
│  │                                                                     │    │
│  │  L2 Normalization:                                                  │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │ Input: CLS embedding [768 dims]                               │  │    │
│  │  │        [0.23, -0.45, 0.67, ..., 0.12]                         │  │    │
│  │  │                                                               │  │    │
│  │  │ L2 Norm = √(Σ(xi²)) = 15.67                                   │  │    │
│  │  │                                                               │  │    │
│  │  │ Normalized = embedding / L2_norm                              │  │    │
│  │  │            = [0.015, -0.029, 0.043, ..., 0.008]               │  │    │
│  │  │                                                               │  │    │
│  │  │ Final embedding: ||embedding||₂ = 1.0                         │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                                                                     │    │
│  │  Benefits of Normalization:                                         │    │
│  │  ├─ Consistent magnitude across different texts                     │    │
│  │  ├─ Better cosine similarity calculations                           │    │
│  │  ├─ Improved numerical stability                                    │    │
│  │  └─ Enhanced model convergence                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       Final Embedding                               │    │
│  │                                                                     │    │
│  │  Property Embedding: [384 dims] (model output size)                 │    │
│  │  [0.015, -0.029, 0.043, 0.067, -0.012, ..., 0.008]                  │    │
│  │                                                                     │    │
│  │  Embedding Characteristics:                                         │    │
│  │  ├─ Captures semantic meaning of property description               │    │
│  │  ├─ Encodes spatial relationships (downtown, location)              │    │
│  │  ├─ Represents amenity information (gym, pool, parking)             │    │
│  │  ├─ Maintains property type context (apartment, rental)             │    │
│  │  └─ Enables similarity comparison with query embeddings             │    │
│  │                                                                     │    │
│  │  Usage in Search:                                                   │    │
│  │  ├─ Compare with query embedding using cosine similarity            │    │
│  │  ├─ Input to neural ranking model                                   │    │
│  │  ├─ Cache for repeated searches                                     │    │
│  │  └─ Foundation for recommendation systems                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Similarity Computation and Ranking

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SIMILARITY COMPUTATION & RANKING                        │
│                                                                             │
│  Query Embedding: [0.12, -0.34, 0.78, ..., 0.23] (384 dims)                 │
│  Property Embeddings: N properties × 384 dims                               │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Batch Similarity Calculation                     │    │
│  │                                                                     │    │
│  │  Cosine Similarity Matrix Computation:                              │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │ Query: Q = [q₁, q₂, q₃, ..., q₃₈₄]                            │  │    │
│  │  │                                                               │  │    │
│  │  │ Properties:                                                   │  │    │
│  │  │ P₁ = [p₁₁, p₁₂, p₁₃, ..., p₁₃₈₄]                              │  │    │
│  │  │ P₂ = [p₂₁, p₂₂, p₂₃, ..., p₂₃₈₄]                              │  │    │
│  │  │ ...                                                           │  │    │
│  │  │ Pₙ = [pₙ₁, pₙ₂, pₙ₃, ..., pₙ₃₈₄]                              │  │    │
│  │  │                                                               │  │    │
│  │  │ Cosine Similarity:                                            │  │    │
│  │  │ sim(Q, Pᵢ) = (Q · Pᵢ) / (||Q||₂ × ||Pᵢ||₂)                    │  │    │
│  │  │                                                               │  │    │
│  │  │ Since embeddings are L2-normalized:                           │  │    │
│  │  │ sim(Q, Pᵢ) = Q · Pᵢ = Σ(qⱼ × pᵢⱼ)                             │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                                                                     │    │
│  │  Efficient Matrix Operations:                                       │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │ Using TensorFlow/NumPy:                                       │  │    │
│  │  │                                                               │  │    │
│  │  │ Q_matrix = tf.repeat(Q, [N], axis=0)     # [N × 384]          │  │    │
│  │  │ P_matrix = tf.stack([P₁, P₂, ..., Pₙ])   # [N × 384]          │  │    │
│  │  │                                                               │  │    │
│  │  │ similarities = tf.reduce_sum(Q_matrix * P_matrix, axis=1)     │  │    │
│  │  │ # Result: [sim₁, sim₂, ..., simₙ]                             │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                                                                     │    │
│  │  Example Results:                                                   │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │ Property 1: 0.89 (Downtown 2BR with gym)                      │  │    │
│  │  │ Property 2: 0.34 (Studio apartment)                           │  │    │
│  │  │ Property 3: 0.76 (2BR near downtown)                          │  │    │
│  │  │ Property 4: 0.92 (Luxury 2BR downtown + gym)                  │  │    │
│  │  │ Property 5: 0.23 (3BR house suburban)                         │  │    │
│  │  │ ...                                                           │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                Neural Ranking Model Enhancement                     │    │
│  │                                                                     │    │
│  │  Enhanced Ranking Beyond Cosine Similarity:                         │    │
│  │                                                                     │    │
│  │  Input Features per Property:                                       │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │ Query Embedding:     [384 dims]                               │  │    │
│  │  │ Property Embedding:  [384 dims]                               │  │    │
│  │  │ Concatenated:        [768 dims]                               │  │    │
│  │  │                                                               │  │    │
│  │  │ Additional Features (Optional):                               │  │    │
│  │  │ ├─ Price match score                                          │  │    │
│  │  │ ├─ Location relevance                                         │  │    │
│  │  │ ├─ Amenity overlap count                                      │  │    │
│  │  │ ├─ Property freshness                                         │  │    │
│  │  │ └─ Historical click-through rate                              │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                                                                     │    │
│  │  Neural Network Processing:                                         │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │ Layer 1: Dense(512, activation='relu')                        │  │    │
│  │  │          Input: [768] → Output: [512]                         │  │    │
│  │  │          + Dropout(0.3)                                       │  │    │
│  │  │                                                               │  │    │
│  │  │ Layer 2: Dense(256, activation='relu')                        │  │    │
│  │  │          Input: [512] → Output: [256]                         │  │    │
│  │  │          + Dropout(0.3)                                       │  │    │
│  │  │                                                               │  │    │
│  │  │ Layer 3: Dense(128, activation='relu')                        │  │    │
│  │  │          Input: [256] → Output: [128]                         │  │    │
│  │  │          + Dropout(0.2)                                       │  │    │
│  │  │                                                               │  │    │
│  │  │ Output:  Dense(1, activation='sigmoid')                       │  │    │
│  │  │          Input: [128] → Output: [1]                           │  │    │
│  │  │          Range: [0.0, 1.0]                                    │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                                                                     │    │
│  │  Enhanced Scores:                                                   │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │ Property 1: 0.94 (was 0.89 with cosine only)                  │  │    │
│  │  │ Property 2: 0.28 (was 0.34 with cosine only)                  │  │    │
│  │  │ Property 3: 0.81 (was 0.76 with cosine only)                  │  │    │
│  │  │ Property 4: 0.97 (was 0.92 with cosine only)                  │  │    │
│  │  │ Property 5: 0.19 (was 0.23 with cosine only)                  │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Final Ranking Pipeline                           │    │
│  │                                                                     │    │
│  │  Ranking Score Composition:                                         │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │ For each property i:                                          │  │    │
│  │  │                                                               │  │    │
│  │  │ Base Score = neural_ranking_score[i]                          │  │    │
│  │  │                                                               │  │    │
│  │  │ Business Logic Adjustments:                                   │  │    │
│  │  │ ├─ Featured listing boost: +0.1                               │  │    │
│  │  │ ├─ New listing boost: +0.05                                   │  │    │
│  │  │ ├─ High engagement boost: +0.03                               │  │    │
│  │  │ ├─ Price competitiveness: +0.02                               │  │    │
│  │  │ └─ Verified listing boost: +0.01                              │  │    │
│  │  │                                                               │  │    │
│  │  │ Penalty Factors:                                              │  │    │
│  │  │ ├─ Old listing penalty: -0.02                                 │  │    │
│  │  │ ├─ High price penalty: -0.05                                  │  │    │
│  │  │ └─ Low engagement penalty: -0.03                              │  │    │
│  │  │                                                               │  │    │
│  │  │ Final Score = Base + Boosts - Penalties                       │  │    │
│  │  │ Clamped to [0.0, 1.0] range                                   │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                                                                     │    │
│  │  Sorting and Pagination:                                            │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │ 1. Sort by Final Score (descending)                           │  │    │
│  │  │ 2. Apply tie-breaker (property_id for consistency)            │  │    │
│  │  │ 3. Apply offset and limit for pagination                      │  │    │
│  │  │ 4. Return top-N results with metadata                         │  │    │
│  │  │                                                               │  │    │
│  │  │ Final Ranked Results:                                         │  │    │
│  │  │ Rank 1: Property 4 (Score: 0.97)                              │  │    │
│  │  │ Rank 2: Property 1 (Score: 0.94)                              │  │    │
│  │  │ Rank 3: Property 3 (Score: 0.81)                              │  │    │
│  │  │ ...                                                           │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```
