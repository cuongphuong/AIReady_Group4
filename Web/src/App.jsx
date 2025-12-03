import React, { useEffect, useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import logoRounded from './assets/logo-rounded.svg'

const THEME_KEY = 'bugclassifier_theme'

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

  const [chats, setChats] = useState([])
  const [loading, setLoading] = useState(true)
  const [currentChatId, setCurrentChatId] = useState(null)

  // Load chats từ database khi mount
  useEffect(() => {
    async function loadChatsFromDB() {
      try {
        const response = await fetch('http://localhost:8000/chat/sessions')
        if (!response.ok) throw new Error('Failed to load sessions')
        const data = await response.json()
        
        // Map sessions từ DB sang format của UI
        const loadedChats = await Promise.all(
          (data.sessions || []).map(async (session) => {
            // Load messages cho mỗi session
            try {
              const msgResponse = await fetch(
                `http://localhost:8000/chat/sessions/${session.session_id}/messages?limit=100`
              )
              const msgData = await msgResponse.json()
              
              return {
                id: session.id,
                sessionId: session.session_id,
                title: session.title,
                messages: (msgData.messages || []).map(msg => ({
                  id: msg.id,
                  role: msg.role,
                  text: msg.content,
                  time: new Date(msg.timestamp).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit'
                  }),
                  file_upload_id: msg.file_upload_id,
                  hasDownloadButton: msg.file_upload_id !== null && msg.file_upload_id !== undefined,
                  model: msg.model
                }))
              }
            } catch (err) {
              console.warn(`Failed to load messages for session ${session.session_id}:`, err)
              return {
                id: session.id,
                sessionId: session.session_id,
                title: session.title,
                messages: []
              }
            }
          })
        )
        
        setChats(loadedChats)
        if (loadedChats.length > 0 && !currentChatId) {
          setCurrentChatId(loadedChats[0].id)
        }
      } catch (err) {
        console.warn('Failed to load chats from database:', err)
        setChats([])
      } finally {
        setLoading(false)
      }
    }
    
    loadChatsFromDB()
  }, [])

  async function createNewChat() {
    const sessionId = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`
    const id = Date.now()
    
    console.log('Creating new chat:', { id, sessionId })
    
    // Tạo greeting message
    const greetingMsg = {
      id: 1,
      role: 'assistant',
      text: 'Chào bạn! Tôi có thể giúp gì cho bạn hôm nay?',
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
    
    const newChat = { id, sessionId, title: 'Untitled', messages: [greetingMsg] }
    setChats((s) => [newChat, ...s])
    setCurrentChatId(id)
    
    // Tạo session trong database
    try {
      console.log('Creating session in database...')
      const response = await fetch('http://localhost:8000/chat/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          title: 'Untitled'
        })
      })
      
      if (response.ok) {
        const result = await response.json()
        console.log('Session created in DB:', result)
        
        // Lưu greeting message vào database
        console.log('Saving greeting message...')
        const msgResponse = await fetch(`http://localhost:8000/chat/sessions/${sessionId}/messages`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            role: 'assistant',
            content: 'Chào bạn! Tôi có thể giúp gì cho bạn hôm nay?'
          })
        })
        
        if (msgResponse.ok) {
          const msgResult = await msgResponse.json()
          console.log('Greeting message saved:', msgResult)
        } else {
          console.error('Failed to save greeting message:', msgResponse.status)
        }
      } else {
        const errorText = await response.text()
        console.error('Failed to create session:', response.status, errorText)
      }
    } catch (err) {
      console.error('Failed to create session in database:', err)
    }
  }

  function selectChat(id) {
    setCurrentChatId(id)
  }

  function updateChat(updated) {
    // Cập nhật local state để UI reflect ngay
    setChats((prev) => prev.map((c) => (c.id === updated.id ? updated : c)))
  }

  async function renameChat(id, newTitle) {
    const chat = chats.find(c => c.id === id)
    setChats((prev) => prev.map((c) => (c.id === id ? { ...c, title: newTitle } : c)))
    
    // Update title trong database
    if (chat?.sessionId) {
      try {
        await fetch(`http://localhost:8000/chat/sessions/${chat.sessionId}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: newTitle })
        })
      } catch (err) {
        console.warn('Failed to update session title:', err)
      }
    }
  }

  async function deleteChat(id) {
    const chat = chats.find(c => c.id === id)
    const keep = chats.filter((c) => c.id !== id)
    setChats(keep)
    if (currentChatId === id) {
      setCurrentChatId(keep[0]?.id ?? null)
    }
    
    // Delete session trong database (cascade delete messages + uploads)
    if (chat?.sessionId) {
      try {
        await fetch(`http://localhost:8000/chat/sessions/${chat.sessionId}`, {
          method: 'DELETE'
        })
      } catch (err) {
        console.warn('Failed to delete session:', err)
      }
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
            {loading ? (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                color: 'var(--text-secondary)',
                fontSize: '14px'
              }}>
                Đang tải chats...
              </div>
            ) : (
              <ChatWindow chat={currentChat} onUpdateChat={updateChat} />
            )}
          </main>
        </div>
      </div>
    </div>
  )
}
