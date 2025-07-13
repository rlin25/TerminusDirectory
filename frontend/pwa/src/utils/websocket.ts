/**
 * WebSocket utilities for real-time updates
 * Handles property updates, recommendations, and real-time notifications
 */

import { io, Socket } from 'socket.io-client'
import { offlineManager } from './offline'

interface PropertyUpdate {
  propertyId: string
  type: 'price_change' | 'availability' | 'new_images' | 'status_change'
  data: any
  timestamp: number
}

interface RecommendationUpdate {
  userId: string
  recommendations: any[]
  type: 'new' | 'updated'
  timestamp: number
}

interface NotificationData {
  id: string
  type: 'property_alert' | 'price_change' | 'new_recommendation' | 'system'
  title: string
  message: string
  data?: any
  timestamp: number
}

class WebSocketManager {
  private socket: Socket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectInterval = 5000
  private isConnected = false
  private eventListeners: Map<string, Function[]> = new Map()

  async initialize() {
    try {
      // Only connect if online
      if (!navigator.onLine) {
        console.log('WebSocket: Offline, will connect when online')
        this.setupNetworkListeners()
        return
      }

      await this.connect()
      this.setupNetworkListeners()
    } catch (error) {
      console.error('Failed to initialize WebSocket:', error)
    }
  }

  private async connect() {
    if (this.socket?.connected) {
      return
    }

    try {
      // Get auth token for authenticated connection
      const token = await this.getAuthToken()
      
      this.socket = io(process.env.REACT_APP_WS_URL || 'ws://localhost:8001', {
        auth: {
          token,
        },
        transports: ['websocket', 'polling'],
        upgrade: true,
        rememberUpgrade: true,
        timeout: 20000,
        forceNew: false,
        reconnection: true,
        reconnectionAttempts: this.maxReconnectAttempts,
        reconnectionDelay: this.reconnectInterval,
      })

      this.setupEventHandlers()
      
      console.log('WebSocket: Attempting to connect...')
    } catch (error) {
      console.error('WebSocket connection failed:', error)
      this.scheduleReconnect()
    }
  }

  private setupEventHandlers() {
    if (!this.socket) return

    // Connection events
    this.socket.on('connect', () => {
      console.log('WebSocket: Connected')
      this.isConnected = true
      this.reconnectAttempts = 0
      
      // Join user-specific rooms
      this.joinUserRooms()
      
      // Emit app-wide event
      window.dispatchEvent(new CustomEvent('websocket-connected'))
    })

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket: Disconnected:', reason)
      this.isConnected = false
      
      // Emit app-wide event
      window.dispatchEvent(new CustomEvent('websocket-disconnected', { detail: reason }))
      
      // Auto-reconnect unless it was intentional
      if (reason === 'io server disconnect') {
        this.scheduleReconnect()
      }
    })

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket: Connection error:', error)
      this.scheduleReconnect()
    })

    // Property updates
    this.socket.on('property:update', (update: PropertyUpdate) => {
      this.handlePropertyUpdate(update)
    })

    this.socket.on('property:new', (property: any) => {
      this.handleNewProperty(property)
    })

    // Recommendation updates
    this.socket.on('recommendations:update', (update: RecommendationUpdate) => {
      this.handleRecommendationUpdate(update)
    })

    // Notifications
    this.socket.on('notification', (notification: NotificationData) => {
      this.handleNotification(notification)
    })

    // Search updates (real-time search suggestions)
    this.socket.on('search:suggestions', (suggestions: any[]) => {
      this.emitEvent('search-suggestions', suggestions)
    })

    // Market updates
    this.socket.on('market:update', (marketData: any) => {
      this.handleMarketUpdate(marketData)
    })

    // User activity
    this.socket.on('user:activity', (activity: any) => {
      this.handleUserActivity(activity)
    })

    // System messages
    this.socket.on('system:message', (message: any) => {
      this.handleSystemMessage(message)
    })
  }

  private async joinUserRooms() {
    if (!this.socket || !this.isConnected) return

    try {
      const userId = await this.getUserId()
      const location = await this.getUserLocation()
      
      if (userId) {
        this.socket.emit('join:user', userId)
      }
      
      if (location) {
        this.socket.emit('join:location', {
          city: location.city,
          state: location.state,
          country: location.country,
        })
      }

      // Join preferences-based rooms
      const preferences = await this.getUserPreferences()
      if (preferences) {
        this.socket.emit('join:preferences', preferences)
      }
    } catch (error) {
      console.error('Failed to join user rooms:', error)
    }
  }

  // Event handlers
  private async handlePropertyUpdate(update: PropertyUpdate) {
    try {
      console.log('Property update received:', update)
      
      // Update cached property data
      const cachedProperty = await offlineManager.getCachedProperty(update.propertyId)
      if (cachedProperty) {
        const updatedProperty = {
          ...cachedProperty,
          ...update.data,
          lastUpdated: update.timestamp,
        }
        await offlineManager.cacheProperty(update.propertyId, updatedProperty)
      }

      // Emit event for components to handle
      this.emitEvent('property-update', update)

      // Show notification for significant updates
      if (update.type === 'price_change') {
        this.showPriceChangeNotification(update)
      }
    } catch (error) {
      console.error('Failed to handle property update:', error)
    }
  }

  private handleNewProperty(property: any) {
    console.log('New property received:', property)
    
    // Cache the new property
    offlineManager.cacheProperty(property.id, property)
    
    // Emit event for components
    this.emitEvent('property-new', property)
    
    // Check if property matches user preferences
    this.checkPropertyMatchesPreferences(property)
  }

  private handleRecommendationUpdate(update: RecommendationUpdate) {
    console.log('Recommendation update received:', update)
    
    // Cache updated recommendations
    offlineManager.saveUserData('recommendations', update.recommendations)
    
    // Emit event for components
    this.emitEvent('recommendations-update', update)
    
    // Show notification for new recommendations
    if (update.type === 'new' && update.recommendations.length > 0) {
      this.showNotification({
        id: `rec_${Date.now()}`,
        type: 'new_recommendation',
        title: 'New Recommendations',
        message: `We found ${update.recommendations.length} new properties for you!`,
        timestamp: Date.now(),
      })
    }
  }

  private handleNotification(notification: NotificationData) {
    console.log('Notification received:', notification)
    
    // Store notification locally
    this.storeNotification(notification)
    
    // Show browser notification if permission granted
    this.showBrowserNotification(notification)
    
    // Emit event for components
    this.emitEvent('notification', notification)
  }

  private handleMarketUpdate(marketData: any) {
    console.log('Market update received:', marketData)
    
    // Cache market data
    offlineManager.saveUserData('market_data', marketData)
    
    // Emit event for components
    this.emitEvent('market-update', marketData)
  }

  private handleUserActivity(activity: any) {
    // Handle real-time user activity updates
    this.emitEvent('user-activity', activity)
  }

  private handleSystemMessage(message: any) {
    console.log('System message:', message)
    
    // Show system notification
    this.showNotification({
      id: `sys_${Date.now()}`,
      type: 'system',
      title: 'System Message',
      message: message.text,
      timestamp: Date.now(),
    })
  }

  // Public methods for sending data
  async trackPropertyView(propertyId: string) {
    if (!this.socket || !this.isConnected) return

    this.socket.emit('track:property_view', {
      propertyId,
      timestamp: Date.now(),
      url: window.location.href,
    })
  }

  async trackSearch(query: string, filters: any, resultsCount: number) {
    if (!this.socket || !this.isConnected) return

    this.socket.emit('track:search', {
      query,
      filters,
      resultsCount,
      timestamp: Date.now(),
    })
  }

  async requestRecommendations() {
    if (!this.socket || !this.isConnected) return

    const userId = await this.getUserId()
    if (userId) {
      this.socket.emit('request:recommendations', { userId })
    }
  }

  async updateUserLocation(location: any) {
    if (!this.socket || !this.isConnected) return

    this.socket.emit('user:location_update', location)
  }

  async joinPropertyRoom(propertyId: string) {
    if (!this.socket || !this.isConnected) return

    this.socket.emit('join:property', propertyId)
  }

  async leavePropertyRoom(propertyId: string) {
    if (!this.socket || !this.isConnected) return

    this.socket.emit('leave:property', propertyId)
  }

  // Event system for components
  addEventListener(event: string, callback: Function) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, [])
    }
    this.eventListeners.get(event)!.push(callback)
  }

  removeEventListener(event: string, callback: Function) {
    const listeners = this.eventListeners.get(event)
    if (listeners) {
      const index = listeners.indexOf(callback)
      if (index > -1) {
        listeners.splice(index, 1)
      }
    }
  }

  private emitEvent(event: string, data: any) {
    const listeners = this.eventListeners.get(event)
    if (listeners) {
      listeners.forEach(callback => {
        try {
          callback(data)
        } catch (error) {
          console.error('Event listener error:', error)
        }
      })
    }

    // Also emit as window event for global listening
    window.dispatchEvent(new CustomEvent(`ws-${event}`, { detail: data }))
  }

  // Notification helpers
  private async showPriceChangeNotification(update: PropertyUpdate) {
    const property = await offlineManager.getCachedProperty(update.propertyId)
    if (!property) return

    const priceChange = update.data.price - (property.price || 0)
    const changeType = priceChange > 0 ? 'increased' : 'decreased'
    
    this.showNotification({
      id: `price_${update.propertyId}_${Date.now()}`,
      type: 'price_change',
      title: 'Price Alert',
      message: `${property.title} price ${changeType} by $${Math.abs(priceChange)}`,
      data: { propertyId: update.propertyId, priceChange },
      timestamp: Date.now(),
    })
  }

  private showNotification(notification: NotificationData) {
    // Emit for app components
    this.emitEvent('notification', notification)
    
    // Show browser notification
    this.showBrowserNotification(notification)
  }

  private async showBrowserNotification(notification: NotificationData) {
    // Check notification permission
    if (Notification.permission !== 'granted') {
      return
    }

    try {
      const browserNotification = new Notification(notification.title, {
        body: notification.message,
        icon: '/pwa-192x192.png',
        badge: '/favicon.ico',
        tag: notification.id,
        data: notification.data,
        requireInteraction: notification.type === 'price_change',
      })

      browserNotification.onclick = () => {
        // Handle notification click
        if (notification.data?.propertyId) {
          window.open(`/property/${notification.data.propertyId}`, '_blank')
        }
        browserNotification.close()
      }

      // Auto-close after 5 seconds unless it requires interaction
      if (!browserNotification.requireInteraction) {
        setTimeout(() => {
          browserNotification.close()
        }, 5000)
      }
    } catch (error) {
      console.error('Failed to show browser notification:', error)
    }
  }

  private async storeNotification(notification: NotificationData) {
    try {
      const notifications = await offlineManager.getUserData('notifications') || []
      notifications.unshift(notification)
      
      // Keep only last 50 notifications
      if (notifications.length > 50) {
        notifications.splice(50)
      }
      
      await offlineManager.saveUserData('notifications', notifications)
    } catch (error) {
      console.error('Failed to store notification:', error)
    }
  }

  private async checkPropertyMatchesPreferences(property: any) {
    try {
      const preferences = await this.getUserPreferences()
      if (!preferences) return

      // Simple matching logic (would be more sophisticated in practice)
      const matches = (
        (!preferences.maxPrice || property.price <= preferences.maxPrice) &&
        (!preferences.minBedrooms || property.bedrooms >= preferences.minBedrooms) &&
        (!preferences.location || property.location?.city === preferences.location)
      )

      if (matches) {
        this.showNotification({
          id: `match_${property.id}_${Date.now()}`,
          type: 'property_alert',
          title: 'New Property Match!',
          message: `${property.title} matches your preferences`,
          data: { propertyId: property.id },
          timestamp: Date.now(),
        })
      }
    } catch (error) {
      console.error('Failed to check property preferences:', error)
    }
  }

  // Network event handling
  private setupNetworkListeners() {
    window.addEventListener('online', () => {
      if (!this.isConnected) {
        console.log('Network online - reconnecting WebSocket')
        this.connect()
      }
    })

    window.addEventListener('offline', () => {
      if (this.socket) {
        console.log('Network offline - WebSocket will reconnect when online')
      }
    })
  }

  private scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('WebSocket: Max reconnection attempts reached')
      return
    }

    this.reconnectAttempts++
    const delay = this.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1)
    
    console.log(`WebSocket: Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`)
    
    setTimeout(() => {
      if (!this.isConnected && navigator.onLine) {
        this.connect()
      }
    }, delay)
  }

  // Utility methods
  private async getAuthToken(): Promise<string> {
    try {
      return localStorage.getItem('auth_token') || ''
    } catch (error) {
      return ''
    }
  }

  private async getUserId(): Promise<string | null> {
    try {
      const userData = await offlineManager.getUserData('profile')
      return userData?.id || null
    } catch (error) {
      return null
    }
  }

  private async getUserLocation(): Promise<any> {
    try {
      return await offlineManager.getUserData('location')
    } catch (error) {
      return null
    }
  }

  private async getUserPreferences(): Promise<any> {
    try {
      return await offlineManager.getUserData('preferences')
    } catch (error) {
      return null
    }
  }

  // Connection status
  isWebSocketConnected(): boolean {
    return this.isConnected && !!this.socket?.connected
  }

  // Cleanup
  disconnect() {
    if (this.socket) {
      console.log('WebSocket: Disconnecting...')
      this.socket.disconnect()
      this.socket = null
      this.isConnected = false
    }
  }
}

// Global WebSocket manager instance
export const webSocketManager = new WebSocketManager()

// Initialize WebSocket connection
export async function initializeWebSocket() {
  try {
    await webSocketManager.initialize()
    console.log('WebSocket manager initialized')
  } catch (error) {
    console.error('Failed to initialize WebSocket:', error)
  }
}

// Export for use in components
export default webSocketManager