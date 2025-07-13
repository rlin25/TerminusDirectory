import React, { Suspense, lazy } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from 'react-query'
import { ReactQueryDevtools } from 'react-query/devtools'
import { Toaster } from 'react-hot-toast'
import { MotionConfig } from 'framer-motion'

// Providers
import { AuthProvider } from './contexts/AuthContext'
import { OfflineProvider } from './contexts/OfflineContext'
import { LocationProvider } from './contexts/LocationContext'
import { ThemeProvider } from './contexts/ThemeContext'

// Components
import { Layout } from './components/Layout'
import { LoadingSpinner } from './components/LoadingSpinner'
import { NetworkStatus } from './components/NetworkStatus'
import { InstallPrompt } from './components/InstallPrompt'

// Lazy loaded pages for code splitting
const Home = lazy(() => import('./pages/Home'))
const Search = lazy(() => import('./pages/Search'))
const Property = lazy(() => import('./pages/Property'))
const Favorites = lazy(() => import('./pages/Favorites'))
const Recommendations = lazy(() => import('./pages/Recommendations'))
const Profile = lazy(() => import('./pages/Profile'))
const Camera = lazy(() => import('./pages/Camera'))
const Map = lazy(() => import('./pages/Map'))
const Offline = lazy(() => import('./pages/Offline'))
const Settings = lazy(() => import('./pages/Settings'))

// Create React Query client with offline support
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
      retry: (failureCount, error: any) => {
        // Don't retry on network errors when offline
        if (!navigator.onLine) return false
        return failureCount < 3
      },
      refetchOnWindowFocus: false,
      refetchOnReconnect: 'always',
    },
    mutations: {
      retry: (failureCount, error: any) => {
        if (!navigator.onLine) return false
        return failureCount < 2
      },
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <AuthProvider>
          <OfflineProvider>
            <LocationProvider>
              <Router>
                <MotionConfig transition={{ duration: 0.3 }}>
                  <div className="App min-h-screen bg-gray-50 dark:bg-gray-900">
                    {/* Global components */}
                    <NetworkStatus />
                    <InstallPrompt />
                    
                    {/* Toast notifications */}
                    <Toaster
                      position="top-right"
                      toastOptions={{
                        duration: 4000,
                        style: {
                          background: 'var(--toast-bg)',
                          color: 'var(--toast-color)',
                        },
                      }}
                    />
                    
                    {/* Main app layout */}
                    <Layout>
                      <Suspense fallback={<LoadingSpinner />}>
                        <Routes>
                          <Route path="/" element={<Home />} />
                          <Route path="/search" element={<Search />} />
                          <Route path="/property/:id" element={<Property />} />
                          <Route path="/favorites" element={<Favorites />} />
                          <Route path="/recommendations" element={<Recommendations />} />
                          <Route path="/profile" element={<Profile />} />
                          <Route path="/camera" element={<Camera />} />
                          <Route path="/map" element={<Map />} />
                          <Route path="/offline" element={<Offline />} />
                          <Route path="/settings" element={<Settings />} />
                        </Routes>
                      </Suspense>
                    </Layout>
                    
                    {/* Development tools */}
                    {process.env.NODE_ENV === 'development' && (
                      <ReactQueryDevtools initialIsOpen={false} />
                    )}
                  </div>
                </MotionConfig>
              </Router>
            </LocationProvider>
          </OfflineProvider>
        </AuthProvider>
      </ThemeProvider>
    </QueryClientProvider>
  )
}

export default App