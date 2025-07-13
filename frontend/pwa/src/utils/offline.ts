/**
 * Offline utilities for PWA functionality
 * Handles data synchronization, offline storage, and background sync
 */

import localforage from 'localforage'
import { openDB, DBSchema, IDBPDatabase } from 'idb'

// IndexedDB schema
interface PWADatabase extends DBSchema {
  properties: {
    key: string
    value: {
      id: string
      data: any
      timestamp: number
      source: 'api' | 'cache' | 'offline'
    }
  }
  searches: {
    key: string
    value: {
      query: string
      filters: any
      results: any[]
      timestamp: number
    }
  }
  favorites: {
    key: string
    value: {
      propertyId: string
      data: any
      addedAt: number
      synced: boolean
    }
  }
  sync_queue: {
    key: string
    value: {
      id: string
      type: 'favorite' | 'search' | 'user_action'
      data: any
      timestamp: number
      retries: number
    }
  }
  user_data: {
    key: string
    value: {
      type: string
      data: any
      timestamp: number
      synced: boolean
    }
  }
}

class OfflineManager {
  private db: IDBPDatabase<PWADatabase> | null = null
  private syncQueue: Array<any> = []
  private syncInProgress = false

  async initialize() {
    try {
      // Configure localforage
      localforage.config({
        driver: [localforage.INDEXEDDB, localforage.WEBSQL, localforage.LOCALSTORAGE],
        name: 'RentalMLPWA',
        version: 1.0,
        storeName: 'app_data',
      })

      // Initialize IndexedDB
      this.db = await openDB<PWADatabase>('RentalMLPWA', 1, {
        upgrade(db) {
          // Properties store
          if (!db.objectStoreNames.contains('properties')) {
            db.createObjectStore('properties', { keyPath: 'id' })
          }

          // Searches store
          if (!db.objectStoreNames.contains('searches')) {
            const searchStore = db.createObjectStore('searches', { keyPath: 'query' })
            searchStore.createIndex('timestamp', 'timestamp')
          }

          // Favorites store
          if (!db.objectStoreNames.contains('favorites')) {
            const favStore = db.createObjectStore('favorites', { keyPath: 'propertyId' })
            favStore.createIndex('synced', 'synced')
            favStore.createIndex('addedAt', 'addedAt')
          }

          // Sync queue store
          if (!db.objectStoreNames.contains('sync_queue')) {
            const syncStore = db.createObjectStore('sync_queue', { keyPath: 'id' })
            syncStore.createIndex('timestamp', 'timestamp')
            syncStore.createIndex('type', 'type')
          }

          // User data store
          if (!db.objectStoreNames.contains('user_data')) {
            const userStore = db.createObjectStore('user_data', { keyPath: 'type' })
            userStore.createIndex('synced', 'synced')
          }
        },
      })

      // Start background sync
      this.startBackgroundSync()

      // Listen for network status changes
      window.addEventListener('online', () => {
        this.handleOnline()
      })

      window.addEventListener('offline', () => {
        this.handleOffline()
      })

      console.log('Offline manager initialized')
    } catch (error) {
      console.error('Failed to initialize offline manager:', error)
    }
  }

  // Property caching
  async cacheProperty(propertyId: string, propertyData: any) {
    if (!this.db) return

    try {
      await this.db.put('properties', {
        id: propertyId,
        data: propertyData,
        timestamp: Date.now(),
        source: 'api',
      })
    } catch (error) {
      console.error('Failed to cache property:', error)
    }
  }

  async getCachedProperty(propertyId: string) {
    if (!this.db) return null

    try {
      const property = await this.db.get('properties', propertyId)
      return property ? property.data : null
    } catch (error) {
      console.error('Failed to get cached property:', error)
      return null
    }
  }

  async getCachedProperties(limit = 50) {
    if (!this.db) return []

    try {
      const properties = await this.db.getAll('properties', undefined, limit)
      return properties.map(p => p.data)
    } catch (error) {
      console.error('Failed to get cached properties:', error)
      return []
    }
  }

  // Search caching
  async cacheSearchResults(query: string, filters: any, results: any[]) {
    if (!this.db) return

    try {
      const searchKey = this.generateSearchKey(query, filters)
      await this.db.put('searches', {
        query: searchKey,
        filters,
        results,
        timestamp: Date.now(),
      })

      // Cache individual properties from results
      for (const property of results) {
        await this.cacheProperty(property.id, property)
      }
    } catch (error) {
      console.error('Failed to cache search results:', error)
    }
  }

  async getCachedSearchResults(query: string, filters: any) {
    if (!this.db) return null

    try {
      const searchKey = this.generateSearchKey(query, filters)
      const search = await this.db.get('searches', searchKey)
      
      if (!search) return null

      // Check if results are still fresh (within 1 hour)
      const isStale = Date.now() - search.timestamp > 60 * 60 * 1000
      
      return {
        results: search.results,
        timestamp: search.timestamp,
        isStale,
      }
    } catch (error) {
      console.error('Failed to get cached search results:', error)
      return null
    }
  }

  private generateSearchKey(query: string, filters: any): string {
    return `${query}_${JSON.stringify(filters)}`
  }

  // Favorites management
  async addFavorite(propertyId: string, propertyData: any) {
    if (!this.db) return false

    try {
      await this.db.put('favorites', {
        propertyId,
        data: propertyData,
        addedAt: Date.now(),
        synced: false,
      })

      // Add to sync queue
      await this.addToSyncQueue('favorite', {
        action: 'add',
        propertyId,
        propertyData,
      })

      return true
    } catch (error) {
      console.error('Failed to add favorite:', error)
      return false
    }
  }

  async removeFavorite(propertyId: string) {
    if (!this.db) return false

    try {
      await this.db.delete('favorites', propertyId)

      // Add to sync queue
      await this.addToSyncQueue('favorite', {
        action: 'remove',
        propertyId,
      })

      return true
    } catch (error) {
      console.error('Failed to remove favorite:', error)
      return false
    }
  }

  async getFavorites() {
    if (!this.db) return []

    try {
      const favorites = await this.db.getAll('favorites')
      return favorites.map(f => f.data)
    } catch (error) {
      console.error('Failed to get favorites:', error)
      return []
    }
  }

  // Sync queue management
  async addToSyncQueue(type: string, data: any) {
    if (!this.db) return

    try {
      const id = `${type}_${Date.now()}_${Math.random()}`
      await this.db.put('sync_queue', {
        id,
        type: type as any,
        data,
        timestamp: Date.now(),
        retries: 0,
      })

      // Trigger sync if online
      if (navigator.onLine) {
        this.processSyncQueue()
      }
    } catch (error) {
      console.error('Failed to add to sync queue:', error)
    }
  }

  async processSyncQueue() {
    if (!this.db || this.syncInProgress || !navigator.onLine) return

    this.syncInProgress = true

    try {
      const items = await this.db.getAll('sync_queue')
      
      for (const item of items) {
        try {
          const success = await this.syncItem(item)
          
          if (success) {
            await this.db.delete('sync_queue', item.id)
          } else {
            // Increment retry count
            item.retries++
            if (item.retries < 3) {
              await this.db.put('sync_queue', item)
            } else {
              // Remove after 3 failed attempts
              await this.db.delete('sync_queue', item.id)
            }
          }
        } catch (error) {
          console.error('Failed to sync item:', error)
        }
      }
    } catch (error) {
      console.error('Failed to process sync queue:', error)
    } finally {
      this.syncInProgress = false
    }
  }

  private async syncItem(item: any): Promise<boolean> {
    try {
      // This would make actual API calls to sync data
      switch (item.type) {
        case 'favorite':
          return await this.syncFavorite(item.data)
        case 'search':
          return await this.syncSearch(item.data)
        case 'user_action':
          return await this.syncUserAction(item.data)
        default:
          return false
      }
    } catch (error) {
      console.error('Sync item failed:', error)
      return false
    }
  }

  private async syncFavorite(data: any): Promise<boolean> {
    try {
      const response = await fetch('/api/v1/favorites', {
        method: data.action === 'add' ? 'POST' : 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${await this.getAuthToken()}`,
        },
        body: JSON.stringify({
          propertyId: data.propertyId,
          propertyData: data.propertyData,
        }),
      })

      return response.ok
    } catch (error) {
      return false
    }
  }

  private async syncSearch(data: any): Promise<boolean> {
    // Sync search analytics
    try {
      await fetch('/api/v1/analytics/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${await this.getAuthToken()}`,
        },
        body: JSON.stringify(data),
      })
      return true
    } catch (error) {
      return false
    }
  }

  private async syncUserAction(data: any): Promise<boolean> {
    try {
      await fetch('/api/v1/analytics/action', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${await this.getAuthToken()}`,
        },
        body: JSON.stringify(data),
      })
      return true
    } catch (error) {
      return false
    }
  }

  // User data management
  async saveUserData(type: string, data: any) {
    if (!this.db) return

    try {
      await this.db.put('user_data', {
        type,
        data,
        timestamp: Date.now(),
        synced: false,
      })
    } catch (error) {
      console.error('Failed to save user data:', error)
    }
  }

  async getUserData(type: string) {
    if (!this.db) return null

    try {
      const userData = await this.db.get('user_data', type)
      return userData ? userData.data : null
    } catch (error) {
      console.error('Failed to get user data:', error)
      return null
    }
  }

  // Network event handlers
  private async handleOnline() {
    console.log('App is online - starting sync')
    
    // Trigger sync queue processing
    this.processSyncQueue()
    
    // Emit event for components
    window.dispatchEvent(new CustomEvent('sync-started'))
  }

  private handleOffline() {
    console.log('App is offline - data will be cached locally')
    
    // Show offline indicator
    window.dispatchEvent(new CustomEvent('offline-mode-active'))
  }

  // Background sync
  private startBackgroundSync() {
    // Process sync queue every 30 seconds when online
    setInterval(() => {
      if (navigator.onLine && !this.syncInProgress) {
        this.processSyncQueue()
      }
    }, 30000)
  }

  // Utility methods
  private async getAuthToken(): Promise<string> {
    try {
      return await localforage.getItem('auth_token') || ''
    } catch (error) {
      return ''
    }
  }

  async clearCache() {
    if (!this.db) return

    try {
      await this.db.clear('properties')
      await this.db.clear('searches')
      await localforage.clear()
      console.log('Cache cleared')
    } catch (error) {
      console.error('Failed to clear cache:', error)
    }
  }

  async getCacheSize(): Promise<number> {
    if (!this.db) return 0

    try {
      const properties = await this.db.count('properties')
      const searches = await this.db.count('searches')
      const favorites = await this.db.count('favorites')
      
      return properties + searches + favorites
    } catch (error) {
      return 0
    }
  }

  // Track user actions for analytics
  async trackUserAction(action: string, data: any) {
    await this.addToSyncQueue('user_action', {
      action,
      data,
      timestamp: Date.now(),
      userAgent: navigator.userAgent,
      url: window.location.href,
    })
  }
}

// Global offline manager instance
export const offlineManager = new OfflineManager()

// Initialize offline capabilities
export async function initializeOfflineStorage() {
  try {
    await offlineManager.initialize()
    console.log('Offline storage initialized')
  } catch (error) {
    console.error('Failed to initialize offline storage:', error)
  }
}

// Export utility functions
export {
  offlineManager as default,
  OfflineManager,
}