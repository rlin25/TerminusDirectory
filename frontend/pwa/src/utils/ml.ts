/**
 * Machine Learning utilities for client-side inference
 * Handles TensorFlow.js models, on-device recommendations, and ML predictions
 */

import * as tf from '@tensorflow/tfjs'
import { offlineManager } from './offline'

interface MLModel {
  name: string
  version: string
  model: tf.LayersModel | tf.GraphModel | null
  lastUpdated: number
  size: number
  accuracy?: number
}

interface PredictionResult {
  predictions: number[]
  confidence: number
  modelUsed: string
  inferenceTime: number
}

interface RecommendationInput {
  userId?: string
  userPreferences?: any
  viewedProperties?: string[]
  currentProperty?: any
  location?: any
  priceRange?: [number, number]
  propertyType?: string
}

class MLManager {
  private models: Map<string, MLModel> = new Map()
  private isInitialized = false
  private modelBaseUrl = process.env.REACT_APP_ML_MODELS_URL || '/models'

  async initialize() {
    try {
      // Set TensorFlow.js backend
      await tf.ready()
      console.log('TensorFlow.js backend:', tf.getBackend())

      // Load cached models
      await this.loadCachedModels()

      // Download and cache essential models
      await this.downloadEssentialModels()

      this.isInitialized = true
      console.log('ML Manager initialized')
    } catch (error) {
      console.error('Failed to initialize ML manager:', error)
    }
  }

  private async loadCachedModels() {
    try {
      const cachedModels = await offlineManager.getUserData('ml_models') || {}
      
      for (const [modelName, modelInfo] of Object.entries(cachedModels)) {
        try {
          // Load model from IndexedDB
          const model = await tf.loadLayersModel(`indexeddb://${modelName}`)
          
          this.models.set(modelName, {
            name: modelName,
            version: (modelInfo as any).version,
            model,
            lastUpdated: (modelInfo as any).lastUpdated,
            size: (modelInfo as any).size,
            accuracy: (modelInfo as any).accuracy,
          })
          
          console.log(`Loaded cached model: ${modelName}`)
        } catch (error) {
          console.warn(`Failed to load cached model ${modelName}:`, error)
        }
      }
    } catch (error) {
      console.error('Failed to load cached models:', error)
    }
  }

  private async downloadEssentialModels() {
    const essentialModels = [
      {
        name: 'property_recommender',
        url: `${this.modelBaseUrl}/property_recommender/model.json`,
        priority: 'high',
      },
      {
        name: 'price_predictor', 
        url: `${this.modelBaseUrl}/price_predictor/model.json`,
        priority: 'medium',
      },
      {
        name: 'image_classifier',
        url: `${this.modelBaseUrl}/image_classifier/model.json`,
        priority: 'low',
      },
      {
        name: 'search_ranker',
        url: `${this.modelBaseUrl}/search_ranker/model.json`,
        priority: 'medium',
      },
    ]

    // Download high priority models first
    const highPriorityModels = essentialModels.filter(m => m.priority === 'high')
    await Promise.all(highPriorityModels.map(m => this.downloadModel(m.name, m.url)))

    // Download other models in background
    const otherModels = essentialModels.filter(m => m.priority !== 'high')
    otherModels.forEach(m => {
      this.downloadModel(m.name, m.url).catch(error => {
        console.warn(`Failed to download ${m.name}:`, error)
      })
    })
  }

  async downloadModel(name: string, url: string): Promise<boolean> {
    try {
      // Check if we already have this model
      if (this.models.has(name)) {
        console.log(`Model ${name} already loaded`)
        return true
      }

      console.log(`Downloading model: ${name}`)
      
      // Download model
      const model = await tf.loadLayersModel(url)
      
      // Save to IndexedDB for offline use
      await model.save(`indexeddb://${name}`)
      
      // Store model info
      const modelInfo = {
        name,
        version: '1.0.0', // Would come from model metadata
        model,
        lastUpdated: Date.now(),
        size: this.estimateModelSize(model),
      }
      
      this.models.set(name, modelInfo)
      
      // Update cached models list
      await this.updateCachedModelsList()
      
      console.log(`Model ${name} downloaded and cached`)
      return true
    } catch (error) {
      console.error(`Failed to download model ${name}:`, error)
      return false
    }
  }

  // Property recommendation using on-device ML
  async getPropertyRecommendations(
    input: RecommendationInput,
    maxResults = 10
  ): Promise<any[]> {
    try {
      const model = this.models.get('property_recommender')
      if (!model?.model) {
        console.warn('Property recommender model not available')
        return this.getFallbackRecommendations(input, maxResults)
      }

      // Prepare input features
      const features = await this.prepareRecommendationFeatures(input)
      
      // Run inference
      const startTime = Date.now()
      const predictions = model.model.predict(features) as tf.Tensor
      const scores = await predictions.data()
      const inferenceTime = Date.now() - startTime
      
      // Get cached properties for ranking
      const cachedProperties = await offlineManager.getCachedProperties()
      
      // Rank properties by ML scores
      const rankedProperties = this.rankPropertiesByScores(
        cachedProperties,
        Array.from(scores),
        maxResults
      )
      
      // Track inference
      await this.trackMLInference('property_recommendation', inferenceTime, rankedProperties.length)
      
      // Cleanup tensors
      predictions.dispose()
      features.dispose()
      
      return rankedProperties
    } catch (error) {
      console.error('Property recommendation failed:', error)
      return this.getFallbackRecommendations(input, maxResults)
    }
  }

  // Price prediction for properties
  async predictPropertyPrice(propertyFeatures: any): Promise<number | null> {
    try {
      const model = this.models.get('price_predictor')
      if (!model?.model) {
        console.warn('Price predictor model not available')
        return null
      }

      // Prepare features
      const features = this.preparePriceFeatures(propertyFeatures)
      
      // Run inference
      const startTime = Date.now()
      const prediction = model.model.predict(features) as tf.Tensor
      const priceArray = await prediction.data()
      const inferenceTime = Date.now() - startTime
      
      const predictedPrice = priceArray[0]
      
      // Track inference
      await this.trackMLInference('price_prediction', inferenceTime, 1)
      
      // Cleanup
      prediction.dispose()
      features.dispose()
      
      return predictedPrice
    } catch (error) {
      console.error('Price prediction failed:', error)
      return null
    }
  }

  // Image classification for property photos
  async classifyPropertyImage(imageElement: HTMLImageElement): Promise<any> {
    try {
      const model = this.models.get('image_classifier')
      if (!model?.model) {
        console.warn('Image classifier model not available')
        return null
      }

      // Preprocess image
      const tensorImage = tf.browser.fromPixels(imageElement)
        .resizeNearestNeighbor([224, 224]) // Standard input size
        .cast('float32')
        .div(255.0)
        .expandDims(0)

      // Run inference
      const startTime = Date.now()
      const predictions = model.model.predict(tensorImage) as tf.Tensor
      const scores = await predictions.data()
      const inferenceTime = Date.now() - startTime

      // Get top classifications
      const classNames = [
        'kitchen', 'bathroom', 'bedroom', 'living_room', 'dining_room',
        'exterior', 'balcony', 'garage', 'garden', 'office'
      ]
      
      const results = Array.from(scores)
        .map((score, index) => ({
          class: classNames[index] || `class_${index}`,
          confidence: score,
        }))
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 3)

      // Track inference
      await this.trackMLInference('image_classification', inferenceTime, results.length)

      // Cleanup
      predictions.dispose()
      tensorImage.dispose()

      return results
    } catch (error) {
      console.error('Image classification failed:', error)
      return null
    }
  }

  // Search ranking using ML
  async rankSearchResults(query: string, properties: any[]): Promise<any[]> {
    try {
      const model = this.models.get('search_ranker')
      if (!model?.model || properties.length === 0) {
        return properties // Return as-is if no model
      }

      // Prepare features for each property
      const features = await this.prepareSearchFeatures(query, properties)
      
      // Run inference
      const startTime = Date.now()
      const predictions = model.model.predict(features) as tf.Tensor
      const scores = await predictions.data()
      const inferenceTime = Date.now() - startTime

      // Rank properties by scores
      const rankedProperties = properties
        .map((property, index) => ({
          ...property,
          mlScore: scores[index],
        }))
        .sort((a, b) => b.mlScore - a.mlScore)

      // Track inference
      await this.trackMLInference('search_ranking', inferenceTime, properties.length)

      // Cleanup
      predictions.dispose()
      features.dispose()

      return rankedProperties
    } catch (error) {
      console.error('Search ranking failed:', error)
      return properties
    }
  }

  // Feature preparation methods
  private async prepareRecommendationFeatures(input: RecommendationInput): Promise<tf.Tensor> {
    // This would prepare features based on user preferences, location, etc.
    // Simplified example:
    const features = [
      input.priceRange?.[0] || 0,
      input.priceRange?.[1] || 10000,
      input.location?.latitude || 0,
      input.location?.longitude || 0,
      input.userPreferences?.bedrooms || 2,
      input.userPreferences?.bathrooms || 1,
      input.viewedProperties?.length || 0,
    ]

    // Normalize features (would use proper scaling in production)
    const normalizedFeatures = features.map(f => f / 10000)
    
    return tf.tensor2d([normalizedFeatures])
  }

  private preparePriceFeatures(propertyFeatures: any): tf.Tensor {
    const features = [
      propertyFeatures.bedrooms || 0,
      propertyFeatures.bathrooms || 0,
      propertyFeatures.sqft || 0,
      propertyFeatures.latitude || 0,
      propertyFeatures.longitude || 0,
      propertyFeatures.yearBuilt || 2000,
      propertyFeatures.lotSize || 0,
    ]

    // Normalize features
    const normalizedFeatures = [
      features[0] / 10, // bedrooms
      features[1] / 10, // bathrooms  
      features[2] / 5000, // sqft
      (features[3] + 90) / 180, // latitude
      (features[4] + 180) / 360, // longitude
      (features[5] - 1900) / 120, // year built
      features[6] / 50000, // lot size
    ]

    return tf.tensor2d([normalizedFeatures])
  }

  private async prepareSearchFeatures(query: string, properties: any[]): Promise<tf.Tensor> {
    // Simple text similarity features (would use embeddings in production)
    const queryWords = query.toLowerCase().split(' ')
    
    const features = properties.map(property => {
      const title = property.title?.toLowerCase() || ''
      const description = property.description?.toLowerCase() || ''
      
      // Calculate simple word overlap
      const titleMatches = queryWords.filter(word => title.includes(word)).length
      const descMatches = queryWords.filter(word => description.includes(word)).length
      
      return [
        titleMatches / queryWords.length,
        descMatches / queryWords.length,
        property.price || 0,
        property.bedrooms || 0,
        property.bathrooms || 0,
      ]
    })

    return tf.tensor2d(features)
  }

  private rankPropertiesByScores(
    properties: any[],
    scores: number[],
    maxResults: number
  ): any[] {
    return properties
      .map((property, index) => ({
        ...property,
        mlScore: scores[index] || 0,
      }))
      .sort((a, b) => b.mlScore - a.mlScore)
      .slice(0, maxResults)
  }

  private getFallbackRecommendations(input: RecommendationInput, maxResults: number): any[] {
    // Simple fallback recommendations based on rules
    // This would be more sophisticated in production
    return []
  }

  // Model management
  async checkForModelUpdates(): Promise<void> {
    try {
      // Check for model updates from server
      const response = await fetch(`${this.modelBaseUrl}/manifest.json`)
      const manifest = await response.json()

      for (const modelInfo of manifest.models) {
        const currentModel = this.models.get(modelInfo.name)
        
        if (!currentModel || currentModel.version !== modelInfo.version) {
          console.log(`Updating model: ${modelInfo.name}`)
          await this.downloadModel(modelInfo.name, modelInfo.url)
        }
      }
    } catch (error) {
      console.error('Failed to check for model updates:', error)
    }
  }

  async getModelInfo(): Promise<any[]> {
    return Array.from(this.models.values()).map(model => ({
      name: model.name,
      version: model.version,
      size: model.size,
      lastUpdated: model.lastUpdated,
      loaded: !!model.model,
    }))
  }

  async removeModel(modelName: string): Promise<boolean> {
    try {
      // Remove from IndexedDB
      await tf.io.removeModel(`indexeddb://${modelName}`)
      
      // Remove from memory
      const model = this.models.get(modelName)
      if (model?.model) {
        model.model.dispose()
      }
      this.models.delete(modelName)
      
      // Update cached models list
      await this.updateCachedModelsList()
      
      console.log(`Removed model: ${modelName}`)
      return true
    } catch (error) {
      console.error(`Failed to remove model ${modelName}:`, error)
      return false
    }
  }

  // Utility methods
  private estimateModelSize(model: tf.LayersModel | tf.GraphModel): number {
    try {
      // Rough estimation based on parameter count
      let totalParams = 0
      
      if ('layers' in model) {
        model.layers.forEach(layer => {
          if (layer.countParams) {
            totalParams += layer.countParams()
          }
        })
      }
      
      // Estimate 4 bytes per parameter (float32)
      return totalParams * 4
    } catch (error) {
      return 0
    }
  }

  private async updateCachedModelsList(): Promise<void> {
    try {
      const modelsList = {}
      
      for (const [name, model] of this.models) {
        modelsList[name] = {
          version: model.version,
          lastUpdated: model.lastUpdated,
          size: model.size,
          accuracy: model.accuracy,
        }
      }
      
      await offlineManager.saveUserData('ml_models', modelsList)
    } catch (error) {
      console.error('Failed to update cached models list:', error)
    }
  }

  private async trackMLInference(
    modelType: string,
    inferenceTime: number,
    resultCount: number
  ): Promise<void> {
    try {
      await offlineManager.trackUserAction('ml_inference', {
        modelType,
        inferenceTime,
        resultCount,
        timestamp: Date.now(),
        device: navigator.userAgent,
      })
    } catch (error) {
      console.error('Failed to track ML inference:', error)
    }
  }

  // Performance optimization
  async warmUpModels(): Promise<void> {
    try {
      for (const [name, model] of this.models) {
        if (model.model) {
          // Run a dummy prediction to warm up the model
          const dummyInput = tf.zeros([1, 10])
          try {
            const prediction = model.model.predict(dummyInput) as tf.Tensor
            prediction.dispose()
          } catch (error) {
            console.warn(`Failed to warm up model ${name}:`, error)
          }
          dummyInput.dispose()
        }
      }
      console.log('Models warmed up')
    } catch (error) {
      console.error('Failed to warm up models:', error)
    }
  }

  // Memory management
  disposeModels(): void {
    for (const [name, model] of this.models) {
      if (model.model) {
        model.model.dispose()
        console.log(`Disposed model: ${name}`)
      }
    }
    this.models.clear()
  }

  getMemoryInfo(): any {
    return {
      numTensors: tf.memory().numTensors,
      numBytes: tf.memory().numBytes,
      unreliable: tf.memory().unreliable,
      models: this.models.size,
    }
  }

  // Feature extraction helpers
  async extractPropertyFeatures(property: any): Promise<any> {
    return {
      bedrooms: property.bedrooms || 0,
      bathrooms: property.bathrooms || 0,
      sqft: property.area_sqft || 0,
      price: property.price || 0,
      latitude: property.location?.latitude || 0,
      longitude: property.location?.longitude || 0,
      yearBuilt: property.year_built || 2000,
      propertyType: this.encodePropertyType(property.property_type),
      amenities: this.encodeAmenities(property.amenities || []),
    }
  }

  private encodePropertyType(type: string): number {
    const types = {
      'apartment': 0,
      'house': 1,
      'condo': 2,
      'townhouse': 3,
      'studio': 4,
    }
    return types[type?.toLowerCase()] || 0
  }

  private encodeAmenities(amenities: string[]): number[] {
    const standardAmenities = [
      'parking', 'laundry', 'dishwasher', 'air_conditioning',
      'heating', 'pet_friendly', 'gym', 'pool', 'balcony', 'wifi'
    ]
    
    return standardAmenities.map(amenity => 
      amenities.some(a => a.toLowerCase().includes(amenity)) ? 1 : 0
    )
  }
}

// Global ML manager instance
export const mlManager = new MLManager()

// Initialize ML capabilities
export async function initializeML() {
  try {
    await mlManager.initialize()
    console.log('ML manager initialized')
  } catch (error) {
    console.error('Failed to initialize ML manager:', error)
  }
}

// Export for use in components
export default mlManager