import React, { useEffect, useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import logoRounded from './assets/logo-rounded.svg'

const STORAGE_KEY = 'bugclassifier_chats_v1'
const THEME_KEY = 'bugclassifier_theme'

const sampleChats = [
  {
    id: 1,
    title: 'Hỗ trợ kỹ thuật',
    messages: [
      { id: 1, role: 'assistant', text: 'Chào bạn! Tôi có thể giúp gì cho bạn hôm nay?', time: '09:30' }
    ]
  },
  { id: 2, title: 'Ý tưởng sản phẩm', messages: [] }
]

export default function App() {
  const [theme, setTheme] = useState(() => {
    try {
      const stored = localStorage.getItem(THEME_KEY)
      if (stored) return stored
    } catch (e) {}
    // respect system preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) return 'dark'
    return 'dark'
  })

  const [showFileUpload, setShowFileUpload] = useState(false)

  useEffect(() => {
    document.documentElement.classList.toggle('theme-light', theme === 'light')
    document.documentElement.classList.toggle('theme-dark', theme === 'dark')
    try { localStorage.setItem(THEME_KEY, theme) } catch (e) {}
  }, [theme])

  // If user hasn't explicitly chosen a theme, listen to system changes
  useEffect(() => {
    let mq
    try {
      const stored = localStorage.getItem(THEME_KEY)
      if (!stored && window.matchMedia) {
        mq = window.matchMedia('(prefers-color-scheme: dark)')
        const handler = (e) => setTheme(e.matches ? 'dark' : 'light')
        if (mq.addEventListener) mq.addEventListener('change', handler)
        else mq.addListener && mq.addListener(handler)
        return () => {
          if (mq.removeEventListener) mq.removeEventListener('change', handler)
          else mq.removeListener && mq.removeListener(handler)
        }
      }
    } catch (e) {
      // ignore
    }
  }, [])

  function toggleTheme() {
    setTheme((t) => (t === 'dark' ? 'light' : 'dark'))
  }

  const [chats, setChats] = useState(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      return raw ? JSON.parse(raw) : sampleChats
    } catch (e) {
      return sampleChats
    }
  })

  const [currentChatId, setCurrentChatId] = useState(() => chats[0]?.id ?? null)

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(chats))
    } catch (e) {
      console.warn('Failed to save chats', e)
    }
  }, [chats])

  function createNewChat() {
    const id = Date.now()
    const newChat = { id, title: 'Untitled', messages: [] }
    setChats((s) => [newChat, ...s])
    setCurrentChatId(id)
  }

  function selectChat(id) {
    setCurrentChatId(id)
  }

  function updateChat(updated) {
    setChats((prev) => prev.map((c) => (c.id === updated.id ? updated : c)))
  }

  function renameChat(id, newTitle) {
    setChats((prev) => prev.map((c) => (c.id === id ? { ...c, title: newTitle } : c)))
  }

  function deleteChat(id) {
    const keep = chats.filter((c) => c.id !== id)
    setChats(keep)
    if (currentChatId === id) {
      setCurrentChatId(keep[0]?.id ?? null)
    }
  }

  const currentChat = chats.find((c) => c.id === currentChatId) ?? null

  return (
    <div className="app-wrapper">
      <div className="app-card">
        <header className="app-header">
          <div className="brand">
            <div className="brand-logo" aria-hidden>
              <img src={logoRounded} alt="BugClassifier" />
            </div>
            <div className="brand-text">
              <div className="brand-title">BugClassifier</div>
              <div className="brand-sub">Chat</div>
            </div>
          </div>
          <div className="header-actions">
            <button
              className={`theme-toggle ${theme === 'dark' ? 'is-dark' : 'is-light'}`}
              onClick={toggleTheme}
              title="Toggle theme"
              aria-pressed={theme === 'dark'}
            >
              <span className="switch-track" aria-hidden>
                <svg className="switch-icon switch-icon-left" viewBox="0 0 24 24" width="14" height="14" aria-hidden="true" focusable="false">
                  <path fill="currentColor" d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                </svg>
                <svg className="switch-icon switch-icon-right" viewBox="0 0 24 24" width="14" height="14" aria-hidden="true" focusable="false">
                  <circle cx="12" cy="12" r="4" fill="currentColor" />
                  <g stroke="currentColor" strokeWidth="1.2" strokeLinecap="round">
                    <line x1="12" y1="1" x2="12" y2="3" />
                    <line x1="12" y1="21" x2="12" y2="23" />
                    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                    <line x1="1" y1="12" x2="3" y2="12" />
                    <line x1="21" y1="12" x2="23" y2="12" />
                    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                  </g>
                </svg>
                <span className="switch-knob" />
              </span>
            </button>
            <span className="status-text">Online</span>
          </div>
        </header>

        <div className="app-body--layout">
          <Sidebar chats={chats} currentChatId={currentChatId} onNewChat={createNewChat} onSelect={selectChat} onDelete={deleteChat} onRename={renameChat} />
          <main className="main-chat">
            <ChatWindow chat={currentChat} onUpdateChat={updateChat} />
          </main>
        </div>
      </div>
    </div>
  )
}
