// App shell: header navigation + routes. Pages are small components in src/pages/.
import { useEffect, useState } from 'react'
import { NavLink, Route, Routes, useLocation } from 'react-router-dom'
import ModelExplorer from './components/ModelExplorer'
import Home from './pages/Home'
import EntryPage from './pages/EntryPage'
import Scoreboard from './pages/Scoreboard'
import Methods from './pages/Methods'
import Sources from './pages/Sources'
import About from './pages/About'

type Theme = 'light' | 'dark' | 'system'

function useTheme(): [Theme, (t: Theme) => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      return (localStorage.getItem('atlas-theme') as Theme) || 'system'
    } catch {
      return 'system'
    }
  })
  useEffect(() => {
    const root = document.documentElement
    if (theme === 'system') root.removeAttribute('data-theme')
    else root.setAttribute('data-theme', theme)
    try {
      localStorage.setItem('atlas-theme', theme)
    } catch {
      /* private mode: theme just isn't remembered */
    }
  }, [theme])
  return [theme, setTheme]
}

export default function App() {
  const [theme, setTheme] = useTheme()
  const { pathname } = useLocation()
  useEffect(() => {
    // Braces matter: newer browsers return a Promise from scrollTo, and an effect must return nothing or a cleanup fn.
    window.scrollTo(0, 0)
  }, [pathname])
  const next: Record<Theme, Theme> = { system: 'dark', dark: 'light', light: 'system' }
  return (
    <div className="app">
      <header className="topbar">
        <NavLink to="/" className="brand">
          <img src={`${import.meta.env.BASE_URL}favicon.svg`} alt="" width={28} height={28} />
          <span>Atlas of Circuits &amp; Manifolds</span>
        </NavLink>
        <nav>
          <NavLink to="/" end>
            Atlas
          </NavLink>
          <NavLink to="/models">Models</NavLink>
          <NavLink to="/replications">Replications</NavLink>
          <NavLink to="/methods">Methods</NavLink>
          <NavLink to="/sources">Sources</NavLink>
          <NavLink to="/about">About</NavLink>
          <button className="theme-btn" onClick={() => setTheme(next[theme])} title="Toggle theme">
            {theme === 'dark' ? '☾' : theme === 'light' ? '☀' : '◐'}
          </button>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/entry/:slug" element={<EntryPage />} />
          <Route path="/models" element={<ModelExplorer />} />
          <Route path="/replications" element={<Scoreboard />} />
          <Route path="/methods" element={<Methods />} />
          <Route path="/sources" element={<Sources />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Home />} />
        </Routes>
      </main>
      <footer className="footer">
        Atlas of Circuits &amp; Manifolds · by Libor “Burny” Burian ·{' '}
        <a href="https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas" target="_blank" rel="noopener">
          source &amp; data on GitHub
        </a>{' '}
        · every claim links to its source
      </footer>
    </div>
  )
}
