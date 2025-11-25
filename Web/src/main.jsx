import React from 'react'
import { createRoot } from 'react-dom/client'
import { NextUIProvider } from '@nextui-org/react'
import App from './App'
import './styles.css'

function Main() {
  return (
    <NextUIProvider>
      <App />
    </NextUIProvider>
  )
}

createRoot(document.getElementById('root')).render(<Main />)
