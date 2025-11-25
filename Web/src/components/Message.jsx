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

export default function Message({ role = 'assistant', text = '', time = '' }) {
  const isUser = role === 'user'
  return (
    <div className={`message-row ${isUser ? 'user' : 'assistant'}`}>
      {!isUser && <Avatar role={role} />}
      <div className={`bubble ${isUser ? 'bubble-user' : 'bubble-bot'}`}>
        <div className="bubble-text">{text}</div>
        <div className="bubble-meta">{time}</div>
      </div>
      {isUser && <Avatar role={role} />}
    </div>
  )
}
