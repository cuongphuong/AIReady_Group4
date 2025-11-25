import React from 'react'
import logoRounded from '../assets/logo-rounded.svg'

function Avatar({ role }) {
  if (role === 'user') return <div className="avatar user">U</div>
  return (
    <div className="avatar bot">
      <img src={logoRounded} alt="AI" />
    </div>
  )
}

export default function Message({ role = 'assistant', text = '', time = '', animate = false, typing = false, onDownload = null, hasDownloadButton = false }) {
  const isUser = role === 'user'
  const bubbleClass = [isUser ? 'bubble-user' : 'bubble-bot']
  if (animate) bubbleClass.push('is-entering')
  if (typing) bubbleClass.push('typing')

  return (
    <div className={`message-row ${isUser ? 'user' : 'assistant'}`}>
      {!isUser && <Avatar role={role} />}
      <div className={`bubble ${bubbleClass.join(' ')}`}>
        <div className="bubble-text">
          {typing ? (
            <span className="typing-dots"><span></span><span></span><span></span></span>
          ) : (
            text
          )}
        </div>
        {!isUser && hasDownloadButton && onDownload && (
          <button
            onClick={onDownload}
            className="export-btn download-excel-btn"
            type="button"
            style={{marginTop:12, display:'inline-flex', alignItems:'center', gap:8}}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" style={{verticalAlign:'middle'}} xmlns="http://www.w3.org/2000/svg">
              <rect x="3" y="3" width="18" height="18" rx="5" fill="var(--accent-2)"/>
              <path d="M12 8v5" stroke="#fff" strokeWidth="2" strokeLinecap="round"/>
              <path d="M9 13l3 3 3-3" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <span style={{fontWeight:600, fontSize:14, color:'var(--text)'}}>Tải Excel</span>
          </button>
        )}
        <div className="bubble-meta">{time}</div>
      </div>
      {isUser && <Avatar role={role} />}
    </div>
  )
}
