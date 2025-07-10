# ML Models Pseudocode

## NLP Search Ranker Algorithms

### Text Encoding Algorithm

```
ALGORITHM: encode_text(texts: List[str]) -> Tensor
INPUT: List of property/query text strings
OUTPUT: Normalized embeddings tensor [N × 384]

BEGIN
    // Input validation
    IF texts is empty THEN
        RETURN empty tensor
    END IF
    
    // Tokenization
    tokenized_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='tf'
    )
    
    // Transformer encoding
    transformer_outputs = encoder(tokenized_inputs)
    
    // Extract CLS token embeddings
    embeddings = transformer_outputs.last_hidden_state[:, 0, :]
    
    // L2 normalization
    normalized_embeddings = l2_normalize(embeddings, axis=1)
    
    RETURN normalized_embeddings
EXCEPTION
    LOG error details
    RETURN zero tensor of appropriate shape
END

### Property Ranking Algorithm

```
ALGORITHM: rank_properties(query: str, properties: List[Dict]) -> List[RankingResult]
INPUT: Search query string and list of property dictionaries
OUTPUT: Ranked list of properties with relevance scores

BEGIN
    // Input validation
    IF query is empty OR properties is empty THEN
        RETURN empty list
    END IF
    
    // Encode query
    query_embedding = encode_text([query])
    
    // Prepare property texts
    property_texts = []
    FOR each property in properties DO
        text_parts = [
            property.get('title', ''),
            property.get('description', ''),
            property.get('location', ''),
            join(property.get('amenities', []))
        ]
        combined_text = join(filter_non_empty(text_parts), ' ')
        property_texts.append(combined_text)
    END FOR
    
    // Encode property descriptions
    property_embeddings = encode_text(property_texts)
    
    // Calculate relevance scores
    IF model_is_trained THEN
        // Use trained neural ranking model
        query_repeated = repeat(query_embedding, len(properties), axis=0)
        scores = ranking_model.predict([query_repeated, property_embeddings])
    ELSE
        // Fallback to cosine similarity
        scores = cosine_similarity(query_embedding, property_embeddings)
    END IF
    
    // Create ranking results
    ranking_results = []
    FOR i = 0 to len(properties) - 1 DO
        ranking_features = {
            'text_similarity': scores[i],
            'has_title': bool(properties[i].get('title')),
            'has_description': bool(properties[i].get('description')),
            'has_amenities': bool(properties[i].get('amenities')),
            'text_length': len(property_texts[i])
        }
        
        result = RankingResult(
            property_id=properties[i].get('id', str(i)),
            property_data=properties[i],
            relevance_score=float(scores[i]),
            ranking_features=ranking_features
        )
        ranking_results.append(result)
    END FOR
    
    // Sort by relevance score (descending)
    sort(ranking_results, key=lambda x: x.relevance_score, reverse=True)
    
    RETURN ranking_results
EXCEPTION
    LOG error details
    RETURN empty list
END
```

### Neural Ranking Model Training

```
ALGORITHM: train_ranking_model(training_data: List[Tuple], epochs: int, batch_size: int)
INPUT: Training triplets (query, property, relevance_label)
OUTPUT: Training metrics dictionary

BEGIN
    // Validate training data
    IF training_data is empty THEN
        RAISE ValueError("Training data cannot be empty")
    END IF
    
    // Prepare training data
    queries, properties, labels = unzip(training_data)
    
    // Encode queries and properties
    query_embeddings = encode_text(list(queries))
    
    property_texts = []
    FOR each property in properties DO
        text_parts = [
            property.get('title', ''),
            property.get('description', ''),
            property.get('location', '')
        ]
        combined_text = join(filter_non_empty(text_parts), ' ')
        property_texts.append(combined_text)
    END FOR
    
    property_embeddings = encode_text(property_texts)
    labels_array = convert_to_array(labels, dtype=float32)
    
    // Train model
    history = ranking_model.fit(
        inputs=[query_embeddings, property_embeddings],
        targets=labels_array,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    // Mark as trained
    is_trained = True
    
    // Return training metrics
    training_metrics = {
        'final_loss': history.history['loss'][-1],
        'final_accuracy': history.history['accuracy'][-1],
        'epochs_trained': len(history.history['loss']),
        'is_trained': is_trained
    }
    
    IF validation_data_present THEN
        training_metrics.update({
            'final_val_loss': history.history['val_loss'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1]
        })
    END IF
    
    LOG "Training completed successfully"
    RETURN training_metrics
EXCEPTION
    LOG error details
    RAISE training_exception
END
```

## Collaborative Filtering Algorithms

### Neural Collaborative Filtering Training

```
ALGORITHM: train_collaborative_filter(user_item_matrix: Matrix, epochs: int)
INPUT: User-item interaction matrix [M × N]
OUTPUT: Training results dictionary

BEGIN
    // Store user-item matrix
    interaction_matrix = user_item_matrix
    
    // Prepare training data from positive interactions
    users, items, ratings = [], [], []
    
    FOR user_id = 0 to num_users - 1 DO
        FOR item_id = 0 to num_items - 1 DO
            rating = user_item_matrix[user_id, item_id]
            IF rating > 0 THEN  // Observed interaction
                users.append(user_id)
                items.append(item_id)
                ratings.append(rating)
            END IF
        END FOR
    END FOR
    
    // Generate negative samples (50% of positive samples)
    negative_samples = len(ratings) // 2
    FOR i = 0 to negative_samples - 1 DO
        user_id = random_int(0, num_users - 1)
        item_id = random_int(0, num_items - 1)
        IF user_item_matrix[user_id, item_id] == 0 THEN  // Unobserved
            users.append(user_id)
            items.append(item_id)
            ratings.append(0.0)  // Negative sample
        END IF
    END FOR
    
    // Convert to arrays
    users_array = convert_to_array(users)
    items_array = convert_to_array(items)
    ratings_array = convert_to_array(ratings, dtype=float32)
    
    // Train neural collaborative filtering model
    history = model.fit(
        inputs=[users_array, items_array],
        targets=ratings_array,
        epochs=epochs,
        batch_size=256,
        validation_split=0.2,
        verbose=1,
        shuffle=True
    )
    
    // Mark as trained
    is_trained = True
    
    // Calculate and return metrics
    final_loss = history.history['loss'][-1]
    final_mae = history.history['mae'][-1]
    
    LOG "CF Training completed - Loss: " + final_loss + ", MAE: " + final_mae
    
    RETURN {
        'final_loss': final_loss,
        'final_mae': final_mae,
        'epochs_trained': len(history.history['loss']),
        'training_samples': len(users_array)
    }
EXCEPTION
    LOG error details
    RAISE training_exception
END
```

### Collaborative Filtering Prediction

```
ALGORITHM: predict_cf_ratings(user_id: int, item_ids: List[int]) -> Array
INPUT: User ID and list of item IDs to predict for
OUTPUT: Predicted ratings array [0.0-1.0]

BEGIN
    // Validate model is trained
    IF NOT is_trained THEN
        RAISE ValueError("Model must be trained before prediction")
    END IF
    
    // Validate user ID
    IF user_id < 0 OR user_id >= num_users THEN
        RAISE ValueError("User ID out of range")
    END IF
    
    // Filter valid item IDs
    valid_item_ids = []
    FOR each item_id in item_ids DO
        IF 0 <= item_id < num_items THEN
            valid_item_ids.append(item_id)
        END IF
    END FOR
    
    IF valid_item_ids is empty THEN
        RETURN empty_array()
    END IF
    
    // Prepare prediction inputs
    user_ids_array = create_array([user_id] * len(valid_item_ids))
    item_ids_array = convert_to_array(valid_item_ids)
    
    // Make predictions using neural network
    predictions = model.predict([user_ids_array, item_ids_array], verbose=0)
    
    RETURN flatten(predictions)
EXCEPTION
    LOG error details
    RETURN empty_array()
END
```

### Recommendation Generation

```
ALGORITHM: generate_cf_recommendations(user_id: int, num_recommendations: int) -> List[RecommendationResult]
INPUT: User ID and desired number of recommendations
OUTPUT: List of recommendation results with scores and explanations

BEGIN
    // Validate inputs
    IF NOT is_trained THEN
        RAISE ValueError("Model must be trained")
    END IF
    
    IF user_id < 0 OR user_id >= num_users THEN
        RAISE ValueError("User ID out of range")
    END IF
    
    // Get all candidate items
    all_item_ids = range(0, num_items)
    
    // Exclude items user has already interacted with
    IF interaction_matrix is not None THEN
        seen_items = find_indices_where(interaction_matrix[user_id] > 0)
        candidate_items = [item_id for item_id in all_item_ids 
                          if item_id not in seen_items]
    ELSE
        candidate_items = all_item_ids
    END IF
    
    IF candidate_items is empty THEN
        RETURN empty_list()
    END IF
    
    // Predict ratings for all candidate items
    predictions = predict_cf_ratings(user_id, candidate_items)
    
    IF predictions is empty THEN
        RETURN empty_list()
    END IF
    
    // Create recommendation results
    recommendations = []
    FOR i = 0 to len(candidate_items) - 1 DO
        item_id = candidate_items[i]
        prediction = predictions[i]
        
        // Calculate confidence score
        confidence = calculate_confidence(user_id, item_id, prediction)
        
        // Generate explanation
        explanation = generate_explanation(user_id, item_id, prediction)
        
        result = RecommendationResult(
            item_id=item_id,
            predicted_rating=float(prediction),
            confidence_score=confidence,
            explanation=explanation
        )
        recommendations.append(result)
    END FOR
    
    // Sort by predicted rating (descending)
    sort(recommendations, key=lambda x: x.predicted_rating, reverse=True)
    
    // Return top-N recommendations
    RETURN recommendations[:num_recommendations]
EXCEPTION
    LOG error details
    RETURN empty_list()
END
```

## Content-Based Recommender Algorithms

### Feature Processing Pipeline

```
ALGORITHM: process_property_features(properties: List[Dict]) -> PropertyFeatures
INPUT: List of property dictionaries with raw features
OUTPUT: Processed and encoded property features

BEGIN
    // Initialize feature processors
    location_encoder = LabelEncoder()
    price_scaler = StandardScaler()
    amenity_vectorizer = TfidfVectorizer(max_features=500)
    
    // Extract raw features
    locations = []
    price_features = []
    amenity_texts = []
    
    FOR each property in properties DO
        // Location processing
        location = property.get('location', 'unknown')
        locations.append(location)
        
        // Price and size features
        price = property.get('price', 0.0)
        bedrooms = property.get('bedrooms', 0)
        bathrooms = property.get('bathrooms', 0.0)
        sqft = property.get('square_feet', 0) or 0
        
        price_features.append([price, bedrooms, bathrooms, sqft])
        
        // Amenity processing
        amenities = property.get('amenities', [])
        IF isinstance(amenities, list) THEN
            amenity_text = join(amenities, ' ')
        ELSE
            amenity_text = str(amenities)
        END IF
        amenity_texts.append(amenity_text)
    END FOR
    
    // Encode features
    TRY
        // Fit and transform location features
        location_features = location_encoder.fit_transform(locations)
        location_features = reshape(location_features, (-1, 1))
        
        // Scale price features
        price_features_scaled = price_scaler.fit_transform(price_features)
        
        // Vectorize amenities
        amenity_features = amenity_vectorizer.fit_transform(amenity_texts)
        amenity_features = convert_to_dense(amenity_features)
        
        // Combine all features
        combined_features = concatenate([
            location_features,
            price_features_scaled,
            amenity_features
        ], axis=1)
        
        // Extract feature names
        feature_names = []
        feature_names.extend(['location'])
        feature_names.extend(['price', 'bedrooms', 'bathrooms', 'sqft'])
        feature_names.extend(amenity_vectorizer.get_feature_names_out())
        
        RETURN PropertyFeatures(
            location_features=location_features,
            price_features=price_features_scaled,
            bedroom_features=price_features_scaled[:, 1:2],
            bathroom_features=price_features_scaled[:, 2:3],
            amenity_features=amenity_features,
            combined_features=combined_features,
            feature_names=feature_names
        )
    EXCEPT Exception as e
        LOG "Feature processing failed: " + str(e)
        RAISE feature_processing_exception
    END TRY
END
```

### Content-Based Model Training

```
ALGORITHM: train_content_model(user_item_matrix: Matrix, property_data: List[Dict])
INPUT: User-item interactions and property feature data
OUTPUT: Training results dictionary

BEGIN
    // Process property features
    property_features = process_property_features(property_data)
    interaction_matrix = user_item_matrix
    
    // Prepare training data
    training_samples = []
    
    FOR user_id = 0 to num_users - 1 DO
        FOR item_id = 0 to num_items - 1 DO
            rating = user_item_matrix[user_id, item_id]
            IF rating > 0 THEN  // Positive interaction
                user_profile = extract_user_profile(user_id, user_item_matrix, property_features)
                item_features = property_features.combined_features[item_id]
                training_samples.append((user_profile, item_features, 1.0))
            END IF
        END FOR
    END FOR
    
    // Add negative samples
    negative_samples = len(training_samples) // 2
    FOR i = 0 to negative_samples - 1 DO
        user_id = random_int(0, num_users - 1)
        item_id = random_int(0, num_items - 1)
        IF user_item_matrix[user_id, item_id] == 0 THEN
            user_profile = extract_user_profile(user_id, user_item_matrix, property_features)
            item_features = property_features.combined_features[item_id]
            training_samples.append((user_profile, item_features, 0.0))
        END IF
    END FOR
    
    // Prepare training arrays
    user_profiles, item_features_array, labels = unzip(training_samples)
    
    user_profiles_array = convert_to_array(user_profiles)
    item_features_array = convert_to_array(item_features_array)
    labels_array = convert_to_array(labels, dtype=float32)
    
    // Train content-based model
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5)
    ]
    
    history = model.fit(
        inputs=[user_profiles_array, item_features_array],
        targets=labels_array,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    // Calculate property similarity matrix
    property_similarity_matrix = calculate_similarity_matrix(property_features)
    
    // Mark as trained
    is_trained = True
    
    // Return training metrics
    training_metrics = {
        'final_loss': history.history['loss'][-1],
        'final_accuracy': history.history['accuracy'][-1],
        'epochs_trained': len(history.history['loss']),
        'training_samples': len(training_samples),
        'feature_dimensions': property_features.combined_features.shape[1]
    }
    
    LOG "Content-based training completed successfully"
    RETURN training_metrics
EXCEPTION
    LOG error details
    RAISE training_exception
END
```

### User Profile Extraction

```
ALGORITHM: extract_user_profile(user_id: int, interactions: Matrix, features: PropertyFeatures) -> Array
INPUT: User ID, interaction matrix, and property features
OUTPUT: User preference profile vector

BEGIN
    // Get user's interaction history
    user_interactions = interactions[user_id]
    interacted_items = find_indices_where(user_interactions > 0)
    
    IF interacted_items is empty THEN
        // Return zero profile for new users
        RETURN zeros(features.combined_features.shape[1])
    END IF
    
    // Weight interactions by rating/engagement
    interaction_weights = user_interactions[interacted_items]
    
    // Get features of interacted properties
    interacted_features = features.combined_features[interacted_items]
    
    // Calculate weighted average of property features
    weighted_features = []
    FOR i = 0 to len(interacted_items) - 1 DO
        weight = interaction_weights[i]
        property_features = interacted_features[i]
        weighted_features.append(weight * property_features)
    END FOR
    
    // Average the weighted features
    user_profile = mean(weighted_features, axis=0)
    
    // Normalize the profile
    user_profile = l2_normalize(user_profile)
    
    RETURN user_profile
END
```

## Hybrid Recommender Algorithms

### Hybrid Score Calculation

```
ALGORITHM: calculate_hybrid_scores(user_id: int, item_ids: List[int], cf_weight: float, cb_weight: float) -> Array
INPUT: User ID, item IDs, and model weights
OUTPUT: Combined hybrid scores

BEGIN
    // Validate weights
    IF abs(cf_weight + cb_weight - 1.0) > 1e-6 THEN
        RAISE ValueError("Weights must sum to 1.0")
    END IF
    
    // Initialize score arrays
    cf_scores = zeros(len(item_ids))
    cb_scores = zeros(len(item_ids))
    
    // Get collaborative filtering scores
    IF cf_model is not None AND cf_model.is_trained THEN
        TRY
            cf_scores = cf_model.predict(user_id, item_ids)
        EXCEPT Exception as e
            LOG "CF prediction failed: " + str(e)
            cf_scores = zeros(len(item_ids))
        END TRY
    END IF
    
    // Get content-based scores
    IF cb_model is not None AND cb_model.is_trained THEN
        TRY
            user_profile = extract_user_profile(user_id)
            cb_scores = cb_model.predict_for_user(user_profile, item_ids)
        EXCEPT Exception as e
            LOG "CB prediction failed: " + str(e)
            cb_scores = zeros(len(item_ids))
        END TRY
    END IF
    
    // Calculate hybrid scores
    hybrid_scores = cf_weight * cf_scores + cb_weight * cb_scores
    
    // Calculate confidence based on model agreement
    FOR i = 0 to len(item_ids) - 1 DO
        cf_score = cf_scores[i]
        cb_score = cb_scores[i]
        base_confidence = hybrid_scores[i]
        
        // Boost confidence if models agree
        agreement_threshold = 0.2
        IF abs(cf_score - cb_score) < agreement_threshold THEN
            confidence_boost = 0.1 * (1.0 - abs(cf_score - cb_score) / agreement_threshold)
            hybrid_scores[i] = min(hybrid_scores[i] + confidence_boost, 1.0)
        END IF
    END FOR
    
    RETURN hybrid_scores
END
```

### Hybrid Recommendation Generation

```
ALGORITHM: generate_hybrid_recommendations(user_id: int, num_recommendations: int) -> List[HybridRecommendationResult]
INPUT: User ID and desired number of recommendations
OUTPUT: Ranked list of hybrid recommendations

BEGIN
    // Check if user exists
    user = get_user(user_id)
    IF user is None THEN
        RAISE ValueError("User not found")
    END IF
    
    // Determine cold start scenario
    interaction_count = count_user_interactions(user_id)
    cold_start_threshold = 5
    
    // Adjust weights based on data availability
    IF interaction_count < cold_start_threshold THEN
        // New user - favor content-based
        cf_weight = 0.2
        cb_weight = 0.8
    ELSE
        // Experienced user - use default weights
        cf_weight = default_cf_weight
        cb_weight = default_cb_weight
    END IF
    
    // Get candidate items (exclude already interacted)
    all_items = range(0, num_items)
    interacted_items = get_user_interactions(user_id)
    candidate_items = [item for item in all_items 
                      if item not in interacted_items]
    
    IF candidate_items is empty THEN
        RETURN empty_list()
    END IF
    
    // Calculate hybrid scores
    hybrid_scores = calculate_hybrid_scores(user_id, candidate_items, cf_weight, cb_weight)
    
    // Create recommendation results
    recommendations = []
    FOR i = 0 to len(candidate_items) - 1 DO
        item_id = candidate_items[i]
        hybrid_score = hybrid_scores[i]
        
        // Get individual model scores for explanation
        cf_score = get_cf_score(user_id, item_id)
        cb_score = get_cb_score(user_id, item_id)
        
        // Calculate confidence
        confidence = calculate_confidence(cf_score, cb_score, hybrid_score)
        
        // Generate explanation
        explanation = generate_hybrid_explanation(
            user_id, item_id, cf_score, cb_score, hybrid_score
        )
        
        result = HybridRecommendationResult(
            item_id=item_id,
            hybrid_score=hybrid_score,
            cf_score=cf_score,
            cb_score=cb_score,
            confidence=confidence,
            explanation=explanation,
            model_weights={'cf_weight': cf_weight, 'cb_weight': cb_weight}
        )
        recommendations.append(result)
    END FOR
    
    // Sort by hybrid score (descending)
    sort(recommendations, key=lambda x: x.hybrid_score, reverse=True)
    
    // Apply diversity filtering
    diverse_recommendations = apply_diversity_filter(recommendations, diversity_threshold=0.8)
    
    // Return top-N recommendations
    RETURN diverse_recommendations[:num_recommendations]
EXCEPTION
    LOG error details
    RETURN empty_list()
END
```

### Explanation Generation

```
ALGORITHM: generate_hybrid_explanation(user_id: int, item_id: int, cf_score: float, cb_score: float, hybrid_score: float) -> Dict
INPUT: User and item IDs, individual model scores, and hybrid score
OUTPUT: Multi-level explanation dictionary

BEGIN
    // Get user and item data
    user = get_user(user_id)
    item = get_item(item_id)
    
    // Initialize explanation components
    cf_explanation = ""
    cb_explanation = ""
    
    // Generate CF explanation
    IF cf_score > 0.7 THEN
        similar_users = get_similar_users(user_id, top_k=3)
        cf_explanation = f"Users with similar preferences ({len(similar_users)} users) also liked this property"
    ELIF cf_score > 0.5 THEN
        cf_explanation = "Recommended based on collaborative filtering patterns"
    ELSE
        cf_explanation = "Limited collaborative filtering data available"
    END IF
    
    // Generate CB explanation
    user_preferences = user.preferences
    item_features = get_item_features(item_id)
    
    matching_features = []
    IF item.location in user_preferences.preferred_locations THEN
        matching_features.append(f"Located in preferred area: {item.location}")
    END IF
    
    IF user_preferences.min_price <= item.price <= user_preferences.max_price THEN
        matching_features.append("Within your budget range")
    END IF
    
    IF user_preferences.min_bedrooms <= item.bedrooms <= user_preferences.max_bedrooms THEN
        matching_features.append(f"Has {item.bedrooms} bedrooms as preferred")
    END IF
    
    matching_amenities = intersection(item.amenities, user_preferences.required_amenities)
    IF matching_amenities THEN
        matching_features.append(f"Includes desired amenities: {join(matching_amenities, ', ')}")
    END IF
    
    cb_explanation = join(matching_features, "; ") if matching_features else "General content match"
    
    // Generate simple explanation
    simple_explanation = ""
    IF hybrid_score > 0.8 THEN
        simple_explanation = "Highly recommended based on your preferences and similar users"
    ELIF hybrid_score > 0.6 THEN
        simple_explanation = "Good match based on your profile and user patterns"
    ELSE
        simple_explanation = "Moderately recommended"
    END IF
    
    // Generate detailed explanation
    detailed_explanation = f"""
    Content Matching (Weight: {cb_weight*100:.0f}%): {cb_explanation}
    
    Collaborative Filtering (Weight: {cf_weight*100:.0f}%): {cf_explanation}
    
    Final Score: {cf_weight:.1f}×{cf_score:.2f} + {cb_weight:.1f}×{cb_score:.2f} = {hybrid_score:.2f}
    """
    
    // Generate technical explanation
    technical_explanation = {
        'model_scores': {
            'collaborative_filtering': cf_score,
            'content_based': cb_score,
            'hybrid_combined': hybrid_score
        },
        'model_weights': {
            'cf_weight': cf_weight,
            'cb_weight': cb_weight
        },
        'feature_contributions': get_feature_importance(user_id, item_id),
        'similar_users': get_similar_users(user_id, top_k=5),
        'matching_criteria': matching_features
    }
    
    RETURN {
        'simple': simple_explanation,
        'detailed': detailed_explanation,
        'technical': technical_explanation,
        'confidence': calculate_explanation_confidence(cf_score, cb_score, hybrid_score)
    }
END
```
CLASS NLPSearchRanker:
    ATTRIBUTES:
        - tokenizer: AutoTokenizer (pre-trained tokenizer)
        - encoder: TFAutoModel (transformer encoder)
        - ranking_model: TensorFlow Model (neural ranking model)
        - model_name: string (default: "sentence-transformers/all-MiniLM-L6-v2")
    
    CONSTRUCTOR(model_name):
        - Load pre-trained tokenizer and encoder
        - Build ranking model architecture
        - Compile model with optimizer and loss function
    
    PRIVATE METHOD _build_ranking_model() -> TensorFlow Model:
        """
        PURPOSE: Build neural ranking model for search relevance
        OUTPUT: Compiled TensorFlow model
        
        ARCHITECTURE:
            Input Layer:
                - query_embedding: 384-dimensional vector
                - property_embedding: 384-dimensional vector
            
            Feature Interaction:
                - Concatenate query and property embeddings
                - Combined vector: 768 dimensions
            
            Deep Network:
                - Dense(512, activation='relu')
                - Dropout(0.3)
                - Dense(256, activation='relu')
                - Dropout(0.3)
                - Dense(1, activation='sigmoid')
            
            Compilation:
                - Optimizer: Adam(learning_rate=0.001)
                - Loss: binary_crossentropy
                - Metrics: accuracy, precision, recall
        """
    
    METHOD encode_text(texts: List[String]) -> TensorFlow Tensor:
        """
        PURPOSE: Convert text to embeddings using transformer
        INPUT: List of text strings
        OUTPUT: Tensor of embeddings (batch_size, 384)
        
        ALGORITHM:
            1. Tokenize texts:
               - Apply padding and truncation
               - Max length: 512 tokens
               - Return TensorFlow tensors
            
            2. Generate embeddings:
               - Pass through transformer encoder
               - Extract CLS token representation
               - Return normalized embeddings
        """
    
    METHOD rank_properties(query: String, properties: List[Dictionary]) -> List[Tuple]:
        """
        PURPOSE: Rank properties by relevance to search query
        INPUT: Search query string, list of property dictionaries
        OUTPUT: List of (property, relevance_score) tuples, sorted by score
        
        ALGORITHM:
            1. ENCODE query:
               - Convert query to embedding vector
               - Shape: (1, 384)
            
            2. ENCODE properties:
               - Combine title, description, location for each property
               - Convert to embedding vectors
               - Shape: (num_properties, 384)
            
            3. PREDICT relevance:
               - Repeat query embedding for each property
               - Pass query-property pairs through ranking model
               - Get relevance scores (0-1)
            
            4. RANK and return:
               - Combine properties with scores
               - Sort by relevance score (descending)
               - Return ranked list
        """
    
    METHOD train(training_data: List[Tuple]) -> Dictionary:
        """
        PURPOSE: Train the ranking model on labeled data
        INPUT: List of (query, property, relevance_label) tuples
        OUTPUT: Training metrics dictionary
        
        ALGORITHM:
            1. PREPARE data:
               - Extract queries, properties, labels
               - Encode queries and properties
               - Create training/validation split
            
            2. TRAIN model:
               - Fit ranking model on encoded data
               - Use binary crossentropy loss
               - Monitor validation metrics
               - Apply early stopping
            
            3. EVALUATE performance:
               - Calculate NDCG, MAP, MRR
               - Return training metrics
        """
```

## Hybrid Recommendation System

```pseudocode
CLASS BaseRecommender:
    """Abstract base class for recommendation models"""
    
    ABSTRACT METHOD fit(user_item_matrix: Matrix, **kwargs):
        """Train the recommendation model"""
        pass
    
    ABSTRACT METHOD predict(user_id: Integer, item_ids: List[Integer]) -> Array:
        """Predict ratings for user-item pairs"""
        pass

CLASS CollaborativeFilteringModel extends BaseRecommender:
    ATTRIBUTES:
        - num_users: integer
        - num_items: integer
        - embedding_dim: integer (default: 50)
        - model: TensorFlow Model
    
    CONSTRUCTOR(num_users, num_items, embedding_dim):
        - Set model dimensions
        - Build neural collaborative filtering model
    
    PRIVATE METHOD _build_model() -> TensorFlow Model:
        """
        PURPOSE: Build neural collaborative filtering model
        OUTPUT: Compiled TensorFlow model
        
        ARCHITECTURE:
            Input Layers:
                - user_input: scalar user ID
                - item_input: scalar item ID
            
            Embedding Layers:
                - User embedding: (num_users, embedding_dim)
                - Item embedding: (num_items, embedding_dim)
                - L2 regularization: 1e-6
            
            Neural MF Layers:
                - Concatenate user and item embeddings
                - Dense(128, activation='relu')
                - Dropout(0.2)
                - Dense(64, activation='relu')
                - Dropout(0.2)
                - Dense(1, activation='sigmoid')
            
            Compilation:
                - Optimizer: Adam
                - Loss: binary_crossentropy
                - Metrics: MAE, MSE
        """
    
    METHOD fit(user_item_matrix: Matrix, epochs: Integer):
        """
        PURPOSE: Train collaborative filtering model
        INPUT: User-item interaction matrix, training epochs
        OUTPUT: None (model is trained in-place)
        
        ALGORITHM:
            1. PREPARE training data:
               - Extract (user_id, item_id, rating) triplets
               - Create positive and negative samples
               - Convert to numpy arrays
            
            2. TRAIN model:
               - Fit model on user-item pairs
               - Batch size: 256
               - Validation split: 20%
               - Monitor training progress
            
            3. SAVE model:
               - Serialize trained model
               - Save embeddings for inference
        """
    
    METHOD predict(user_id: Integer, item_ids: List[Integer]) -> Array:
        """
        PURPOSE: Predict user ratings for given items
        INPUT: User ID, list of item IDs
        OUTPUT: Array of predicted ratings
        
        ALGORITHM:
            1. PREPARE input:
               - Create user_id array (repeated)
               - Create item_ids array
            
            2. PREDICT ratings:
               - Pass through trained model
               - Get probability scores
               - Convert to rating predictions
            
            3. RETURN predictions as array
        """

CLASS ContentBasedModel extends BaseRecommender:
    ATTRIBUTES:
        - feature_dim: integer
        - model: TensorFlow Model
        - item_features: Matrix
    
    CONSTRUCTOR(feature_dim):
        - Set feature dimensions
        - Build content-based model
    
    PRIVATE METHOD _build_model() -> TensorFlow Model:
        """
        PURPOSE: Build content-based recommendation model
        OUTPUT: Compiled TensorFlow model
        
        ARCHITECTURE:
            Input Layer:
                - item_features: feature vector
            
            Feature Processing:
                - Dense(256, activation='relu')
                - Dropout(0.3)
                - Dense(128, activation='relu')
                - Dropout(0.3)
                - Dense(64, activation='relu')
            
            Output Layer:
                - Dense(1, activation='sigmoid')
            
            Compilation:
                - Optimizer: Adam
                - Loss: binary_crossentropy
                - Metrics: accuracy, precision, recall
        """
    
    METHOD fit(item_features: Matrix, user_item_matrix: Matrix):
        """
        PURPOSE: Train content-based model
        INPUT: Item feature matrix, user-item interactions
        OUTPUT: None (model is trained in-place)
        
        ALGORITHM:
            1. PREPARE training data:
               - Extract item features for positive interactions
               - Create negative samples
               - Balance positive/negative examples
            
            2. TRAIN model:
               - Fit on item features -> user preference
               - Learn user preference patterns
               - Validate on held-out data
            
            3. STORE item features for inference
        """
    
    METHOD predict(user_id: Integer, item_features: Matrix) -> Array:
        """
        PURPOSE: Predict user preferences for items
        INPUT: User ID, item feature matrix
        OUTPUT: Array of preference scores
        
        ALGORITHM:
            1. GET user profile:
               - Extract user's historical preferences
               - Create user preference vector
            
            2. COMPUTE similarity:
               - Calculate cosine similarity
               - Between user profile and item features
            
            3. RETURN similarity scores
        """

CLASS HybridRecommendationSystem:
    ATTRIBUTES:
        - cf_weight: float (default: 0.6)
        - cb_weight: float (default: 0.4)
        - cf_model: CollaborativeFilteringModel
        - cb_model: ContentBasedModel
    
    CONSTRUCTOR(cf_weight, cb_weight):
        - Set combination weights
        - Initialize model placeholders
    
    METHOD fit(user_item_matrix: Matrix, item_features: Matrix):
        """
        PURPOSE: Train both collaborative and content-based models
        INPUT: User-item matrix, item features
        OUTPUT: None (models are trained in-place)
        
        ALGORITHM:
            1. TRAIN collaborative filtering:
               - Create CF model with matrix dimensions
               - Train on user-item interactions
               - Store trained model
            
            2. TRAIN content-based model:
               - Create CB model with feature dimensions
               - Train on item features and interactions
               - Store trained model
            
            3. VALIDATE hybrid performance:
               - Test both models on validation set
               - Optimize combination weights
        """
    
    METHOD recommend(user_id: Integer, item_ids: List[Integer], 
                    item_features: Matrix, top_k: Integer) -> List[Dictionary]:
        """
        PURPOSE: Generate hybrid recommendations
        INPUT: User ID, candidate items, item features, result count
        OUTPUT: List of recommendation dictionaries
        
        ALGORITHM:
            1. GET collaborative filtering scores:
               - Predict user preferences using CF model
               - Normalize scores to [0, 1]
            
            2. GET content-based scores:
               - Predict user preferences using CB model
               - Normalize scores to [0, 1]
            
            3. COMBINE scores:
               - Weighted combination: cf_weight * cf_scores + cb_weight * cb_scores
               - Apply diversity boosting
               - Handle cold start scenarios
            
            4. RANK and return:
               - Sort by combined score
               - Select top-k items
               - Format as recommendation dictionaries:
                 {
                   'item_id': item_id,
                   'score': combined_score,
                   'cf_score': cf_score,
                   'cb_score': cb_score,
                   'explanation': reason_text
                 }
        """
    
    METHOD explain_recommendation(user_id: Integer, item_id: Integer) -> Dictionary:
        """
        PURPOSE: Explain why item was recommended to user
        INPUT: User ID, item ID
        OUTPUT: Explanation dictionary
        
        ALGORITHM:
            1. GET model contributions:
               - CF contribution and reason
               - CB contribution and reason
            
            2. ANALYZE user-item match:
               - Feature similarities
               - User preference alignment
               - Similar user behaviors
            
            3. FORMAT explanation:
               - Primary reasons (top factors)
               - Secondary reasons
               - Confidence score
               - Return structured explanation
        """

CLASS ModelTrainer:
    """Utility class for training and evaluating ML models"""
    
    ATTRIBUTES:
        - models: Dictionary (model registry)
        - metrics: Dictionary (evaluation metrics)
    
    METHOD train_search_ranker(training_data: List[Tuple]) -> NLPSearchRanker:
        """
        PURPOSE: Train search ranking model
        INPUT: Training data (query, property, relevance) tuples
        OUTPUT: Trained NLP search ranker
        
        ALGORITHM:
            1. PREPARE data:
               - Split into train/validation/test
               - Balance positive/negative examples
               - Create data loaders
            
            2. TRAIN model:
               - Initialize NLP ranker
               - Train with early stopping
               - Monitor validation metrics
            
            3. EVALUATE performance:
               - Calculate NDCG@10, MAP@10
               - Compute ranking metrics
               - Save model and metrics
            
            4. RETURN trained model
        """
    
    METHOD train_recommender(user_item_matrix: Matrix, item_features: Matrix) -> HybridRecommendationSystem:
        """
        PURPOSE: Train hybrid recommendation system
        INPUT: User-item interactions, item features
        OUTPUT: Trained hybrid recommender
        
        ALGORITHM:
            1. PREPARE data:
               - Split users into train/test
               - Create evaluation protocol
               - Handle cold start scenarios
            
            2. TRAIN models:
               - Train collaborative filtering
               - Train content-based model
               - Optimize combination weights
            
            3. EVALUATE performance:
               - Calculate Precision@K, Recall@K
               - Compute diversity metrics
               - Measure coverage and novelty
            
            4. RETURN trained system
        """
    
    METHOD evaluate_model(model: Any, test_data: Any) -> Dictionary:
        """
        PURPOSE: Evaluate model performance
        INPUT: Trained model, test dataset
        OUTPUT: Evaluation metrics dictionary
        
        ALGORITHM:
            1. GENERATE predictions:
               - Run model on test data
               - Collect predictions and ground truth
            
            2. CALCULATE metrics:
               - Accuracy, precision, recall
               - NDCG, MAP, MRR (for ranking)
               - Diversity, coverage (for recommendations)
            
            3. RETURN metrics dictionary
        """
```

## Model Serving Architecture

```pseudocode
CLASS ModelServer:
    """Production model serving system"""
    
    ATTRIBUTES:
        - search_ranker: NLPSearchRanker
        - recommender: HybridRecommendationSystem
        - model_cache: Dictionary (in-memory cache)
        - prediction_cache: Redis (external cache)
    
    CONSTRUCTOR():
        - Load trained models
        - Initialize caches
        - Set up health checks
    
    METHOD load_models():
        """
        PURPOSE: Load trained models from storage
        OUTPUT: None (models loaded in-place)
        
        ALGORITHM:
            1. LOAD search ranker:
               - Load tokenizer and encoder
               - Load ranking model weights
               - Warm up model (dummy inference)
            
            2. LOAD recommender:
               - Load CF model weights
               - Load CB model weights
               - Load item features
            
            3. VALIDATE models:
               - Run basic inference tests
               - Check model versions
               - Log loading status
        """
    
    METHOD rank_search_results(query: String, properties: List[Dictionary]) -> List[Dictionary]:
        """
        PURPOSE: Rank search results using ML model
        INPUT: Search query, candidate properties
        OUTPUT: Ranked properties with scores
        
        ALGORITHM:
            1. CHECK cache:
               - Generate cache key from query
               - Return cached rankings if available
            
            2. RANK properties:
               - Use NLP search ranker
               - Get relevance scores
               - Sort by score (descending)
            
            3. CACHE results:
               - Store rankings in cache
               - Set appropriate TTL
            
            4. RETURN ranked properties
        """
    
    METHOD get_recommendations(user_id: Integer, candidate_items: List[Integer]) -> List[Dictionary]:
        """
        PURPOSE: Generate recommendations for user
        INPUT: User ID, candidate item IDs
        OUTPUT: Ranked recommendations
        
        ALGORITHM:
            1. CHECK cache:
               - Generate cache key from user_id
               - Return cached recommendations if available
            
            2. GENERATE recommendations:
               - Use hybrid recommendation system
               - Get top-k recommendations
               - Include explanation metadata
            
            3. CACHE results:
               - Store recommendations in cache
               - Set appropriate TTL
            
            4. RETURN recommendations
        """
    
    METHOD health_check() -> Dictionary:
        """
        PURPOSE: Check model server health
        OUTPUT: Health status dictionary
        
        ALGORITHM:
            1. CHECK model status:
               - Verify models are loaded
               - Test basic inference
               - Check memory usage
            
            2. CHECK cache status:
               - Verify cache connections
               - Check cache hit rates
               - Monitor cache size
            
            3. RETURN health status:
               - Overall status (healthy/unhealthy)
               - Model-specific status
               - Performance metrics
        """

CLASS BatchInference:
    """Batch processing for offline model inference"""
    
    ATTRIBUTES:
        - model_server: ModelServer
        - batch_size: Integer
        - output_storage: Storage
    
    METHOD generate_embeddings(texts: List[String]) -> Dictionary:
        """
        PURPOSE: Generate embeddings for large text collections
        INPUT: List of text strings
        OUTPUT: Dictionary mapping text -> embedding
        
        ALGORITHM:
            1. BATCH processing:
               - Split texts into batches
               - Process each batch through encoder
               - Avoid memory overflow
            
            2. STORE embeddings:
               - Save to vector database
               - Update search index
               - Log processing statistics
            
            3. RETURN embedding dictionary
        """
    
    METHOD update_recommendations(user_ids: List[Integer]):
        """
        PURPOSE: Pre-compute recommendations for all users
        INPUT: List of user IDs
        OUTPUT: None (recommendations stored)
        
        ALGORITHM:
            1. BATCH processing:
               - Process users in batches
               - Generate recommendations for each user
               - Store in cache with long TTL
            
            2. MONITOR progress:
               - Track completion rate
               - Handle failures gracefully
               - Update processing statistics
            
            3. INVALIDATE old cache:
               - Clear expired recommendations
               - Update cache metadata
        """
```

## Key ML Architecture Patterns

1. **Model Registry Pattern**: Centralized model storage and versioning
2. **Prediction Cache Pattern**: Cache frequent predictions for performance
3. **Hybrid Ensemble Pattern**: Combine multiple models for better performance
4. **Batch/Online Serving Pattern**: Support both real-time and batch inference
5. **Model Monitoring Pattern**: Track model performance and drift
6. **A/B Testing Pattern**: Compare different model versions
7. **Feature Store Pattern**: Centralized feature management and serving