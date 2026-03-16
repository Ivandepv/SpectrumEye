import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import SpectrumEyeDashboard from './SpectrumEyeDashboard'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <SpectrumEyeDashboard />
  </StrictMode>,
)
