import React, { useState, useRef, useEffect } from 'react'

export default function Sidebar({ chats = [], currentChatId = null, onNewChat = () => {}, onSelect = () => {}, onDelete = () => {}, onRename = () => {} }) {
  const [editingId, setEditingId] = useState(null)
  const [editValue, setEditValue] = useState('')
  const inputRef = useRef(null)
  const [pendingDelete, setPendingDelete] = useState(null)

  useEffect(() => {
    if (editingId && inputRef.current) {
      inputRef.current.focus()
      inputRef.current.select()
    }
  }, [editingId])

  function startEditing(ev, chat) {
    ev.stopPropagation()
    setEditingId(chat.id)
    setEditValue(chat.title || 'Untitled')
  }

  function commitEdit(id) {
    const trimmed = (editValue || '').trim()
    if (trimmed.length === 0) {
      // don't allow empty title
      setEditValue('Untitled')
      onRename(id, 'Untitled')
    } else {
      onRename(id, trimmed)
    }
    setEditingId(null)
  }

  function cancelEdit() {
    setEditingId(null)
    setEditValue('')
  }

  function requestDelete(ev, chat) {
    ev.stopPropagation()
    setPendingDelete({ id: chat.id, title: chat.title || 'Untitled' })
  }

  function confirmDelete() {
    if (!pendingDelete) return
    onDelete(pendingDelete.id)
    setPendingDelete(null)
  }

  function cancelDelete() {
    setPendingDelete(null)
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-top">
        <button className="new-chat" onClick={onNewChat}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style={{marginRight:8}} xmlns="http://www.w3.org/2000/svg"><path d="M12 5v14M5 12h14" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
          New chat
        </button>
      </div>

      <div className="conversations">
        {chats.map((c) => {
          const last = c.messages?.slice(-1)[0]
          const lastText = last ? (last.text?.slice(0, 80) + (last.text.length > 80 ? '…' : '')) : 'No messages yet'
          const isActive = c.id === currentChatId
          const isEditing = editingId === c.id
          return (
            <div key={c.id} className={`conversation-item ${isActive ? 'active' : ''}${isEditing ? ' editing' : ''}`} onClick={() => onSelect(c.id)}>
              <div className="conv-body">
                {isEditing ? (
                  <input
                    ref={inputRef}
                    className="conv-title-input"
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    onBlur={() => commitEdit(c.id)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        commitEdit(c.id)
                      } else if (e.key === 'Escape') {
                        cancelEdit()
                      }
                    }}
                  />
                ) : (
                  <div className="conv-title" onDoubleClick={(ev) => startEditing(ev, c)} title="Double-click to rename">{c.title || 'Untitled'}</div>
                )}
                <div className="conv-last">{lastText}</div>
              </div>
              <button
                className="conv-delete"
                title="Delete conversation"
                onClick={(ev) => {
                  requestDelete(ev, c)
                }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                  <path d="M6 19c0 1.1.9 2 2 2h8a2 2 0 0 0 2-2V7H6v12z" />
                  <path d="M19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z" />
                </svg>
              </button>
            </div>
          )
        })}
      </div>

      <div className="sidebar-footer">BugClassifier • Chat UI</div>

      {pendingDelete && (
        <div className="modal-overlay" onClick={cancelDelete}>
          <div className="modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
            <div className="modal-title">Delete conversation?</div>
            <div className="modal-body">Are you sure you want to delete "{pendingDelete.title}"? This action cannot be undone.</div>
            <div className="modal-actions">
              <button className="btn btn-secondary" onClick={cancelDelete}>Cancel</button>
              <button className="btn btn-danger" onClick={confirmDelete}>Delete</button>
            </div>
          </div>
        </div>
      )}

    </aside>
  )
}
