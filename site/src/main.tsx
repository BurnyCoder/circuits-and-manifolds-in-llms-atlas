// Entry point: mounts the app inside a HashRouter (GitHub Pages serves static files only, so URLs look like
// .../#/entry/ioi; see https://reactrouter.com/api/declarative-routers/HashRouter).
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router-dom'
import 'katex/dist/katex.min.css'
import './index.css'
import App from './App'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <HashRouter>
      <App />
    </HashRouter>
  </StrictMode>,
)
