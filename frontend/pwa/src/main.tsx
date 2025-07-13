import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'
import { registerSW } from 'virtual:pwa-register'
import { initializeOfflineStorage } from './utils/offline'
import { initializeML } from './utils/ml'
import { initializeWebSocket } from './utils/websocket'
import { ErrorBoundary } from './components/ErrorBoundary'

// Initialize offline capabilities
initializeOfflineStorage()

// Initialize ML models
initializeML()

// Initialize WebSocket connection
initializeWebSocket()

// Register service worker
const updateSW = registerSW({
  onNeedRefresh() {
    // Show update available notification
    if (confirm('New version available. Refresh to update?')) {
      updateSW(true)
    }
  },
  onOfflineReady() {
    console.log('App ready to work offline')
    // Show offline ready notification
    const event = new CustomEvent('pwa-offline-ready')
    window.dispatchEvent(event)
  },
  immediate: true,
})

// Check for updates periodically
setInterval(() => {
  updateSW()
}, 60000) // Check every minute

// Handle network status changes
window.addEventListener('online', () => {
  console.log('App is back online')
  const event = new CustomEvent('network-online')
  window.dispatchEvent(event)
})

window.addEventListener('offline', () => {
  console.log('App went offline')
  const event = new CustomEvent('network-offline')
  window.dispatchEvent(event)
})

// Handle app visibility changes for battery optimization
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    // App is hidden, reduce background activity
    const event = new CustomEvent('app-hidden')
    window.dispatchEvent(event)
  } else {
    // App is visible, resume normal activity
    const event = new CustomEvent('app-visible')
    window.dispatchEvent(event)
  }
})

// Initialize app
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
)