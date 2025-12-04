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

export default function Message({ role = 'assistant', text = '', time = '', animate = false, typing = false, onDownload = null, hasDownloadButton = false, model = null, onAssign = null, hasAssignButton = false }) {
  const isUser = role === 'user'
  const bubbleClass = [isUser ? 'bubble-user' : 'bubble-bot']
  if (animate) bubbleClass.push('is-entering')
  if (typing) bubbleClass.push('typing')

  return (
    <div className={`message-row ${isUser ? 'user' : 'assistant'}`}>
      {!isUser && <Avatar role={role} />}
      <div className={`bubble ${bubbleClass.join(' ')}`}>
        {!isUser && model && (
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '4px',
            padding: '2px 8px',
            backgroundColor: model === 'GPT-5' ? 'rgba(88, 101, 242, 0.15)' : 'rgba(250, 166, 26, 0.15)',
            border: `1px solid ${model === 'GPT-5' ? 'rgba(88, 101, 242, 0.3)' : 'rgba(250, 166, 26, 0.3)'}`,
            borderRadius: '6px',
            fontSize: '10px',
            fontWeight: '700',
            color: model === 'GPT-5' ? '#5865F2' : '#FAA61A',
            letterSpacing: '0.3px',
            marginBottom: '6px',
            textTransform: 'uppercase'
          }}>
            {model === 'GPT-5' ? '✨' : '⚡'} {model}
          </div>
        )}
        <div className="bubble-text">
          {typing ? (
            <span className="typing-dots"><span></span><span></span><span></span></span>
          ) : (
            text
          )}
        </div>
        {!isUser && (hasDownloadButton || hasAssignButton) && (
          <div style={{marginTop:12, display:'flex', gap:8, flexWrap:'wrap'}}>
            {hasDownloadButton && onDownload && (
              <button
                onClick={onDownload}
                className="export-btn download-excel-btn"
                type="button"
                style={{display:'inline-flex', alignItems:'center', gap:8}}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" style={{verticalAlign:'middle'}} xmlns="http://www.w3.org/2000/svg">
                  <rect x="3" y="3" width="18" height="18" rx="5" fill="var(--accent-2)"/>
                  <path d="M12 8v5" stroke="#fff" strokeWidth="2" strokeLinecap="round"/>
                  <path d="M9 13l3 3 3-3" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                <span style={{fontWeight:600, fontSize:14, color:'var(--text)'}}>Tải Excel</span>
              </button>
            )}
            {hasAssignButton && onAssign && (
              <button
                onClick={onAssign}
                className="export-btn assign-btn"
                type="button"
                style={{display:'inline-flex', alignItems:'center', gap:8}}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" style={{verticalAlign:'middle'}} xmlns="http://www.w3.org/2000/svg">
                  <circle cx="12" cy="8" r="3" stroke="var(--accent-2)" strokeWidth="2" fill="none"/>
                  <path d="M6 18c0-3 2.5-5 6-5s6 2 6 5" stroke="var(--accent-2)" strokeWidth="2" strokeLinecap="round"/>
                </svg>
                <span style={{fontWeight:600, fontSize:14, color:'var(--text)'}}>Assign</span>
              </button>
            )}
          </div>
        )}
        <div className="bubble-meta">{time}</div>
      </div>
      {isUser && <Avatar role={role} />}
    </div>
  )
}
