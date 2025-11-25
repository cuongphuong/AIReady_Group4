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

export default function Message({ role = 'assistant', text = '', time = '', animate = false, typing = false }) {
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
        <div className="bubble-meta">{time}</div>
      </div>
      {isUser && <Avatar role={role} />}
    </div>
  )
}
