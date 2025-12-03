import React, { useState, useRef, useEffect } from 'react'
import Message from './Message'
// Note: papaparse and xlsx are optional dependencies. We dynamically import them
// when handling files so the dev server does not fail if they are not installed.

export default function ChatWindow({ chat = null, onUpdateChat = () => {} }) {
  const [messages, setMessages] = useState(chat?.messages ?? [])
  const [selectedModel, setSelectedModel] = useState('GPT-5')
  const [showModelMenu, setShowModelMenu] = useState(false)
  const modelMenuRef = useRef(null)
  
  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (modelMenuRef.current && !modelMenuRef.current.contains(event.target)) {
        setShowModelMenu(false)
      }
    }
    
    if (showModelMenu) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => {
        document.removeEventListener('mousedown', handleClickOutside)
      }
    }
  }, [showModelMenu])
  
  // Function để lưu message vào database
  async function saveMessageToDB(role, content, fileUploadId = null) {
    if (!chat?.sessionId) {
      console.warn('Cannot save message: no sessionId', { chat })
      return
    }
    
    console.log('Saving message to DB:', { 
      sessionId: chat.sessionId, 
      role, 
      content: content.substring(0, 50),
      fileUploadId 
    })
    
    try {
      const response = await fetch(`http://localhost:8000/chat/sessions/${chat.sessionId}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          role, 
          content,
          file_upload_id: fileUploadId
        })
      })
      
      if (!response.ok) {
        const errorText = await response.text()
        console.error('Failed to save message - Server error:', response.status, errorText)
      } else {
        const result = await response.json()
        console.log('Message saved successfully:', result)
      }
    } catch (err) {
      console.error('Failed to save message to DB:', err)
    }
  }
  const [notice, setNotice] = useState('')
  const [value, setValue] = useState('')
  const [sending, setSending] = useState(false)
  const [bulkFile, setBulkFile] = useState(null)
  const [bulkLoading, setBulkLoading] = useState(false)
  const [isDraggingBulk, setIsDraggingBulk] = useState(false)
  const listRef = useRef(null)
  const composeRef = useRef(null)
  const fileRef = useRef(null)
  const bulkFileRef = useRef(null)
  const pendingRef = useRef(null)

  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight
  }, [messages])

  // auto-resize composer textarea when value changes
  useEffect(() => {
    const ta = composeRef.current
    if (!ta) return
    try {
      // Default to single-line compact height when empty
      if (!value || value.trim() === '') {
        ta.style.height = '36px'
        ta.style.overflowY = 'hidden'
        // ensure pill is single-line
        const pillEl = ta.closest('.composer-pill')
        if (pillEl) pillEl.classList.remove('is-multiline')
        return
      }
      ta.style.height = 'auto'
      const newH = Math.min(ta.scrollHeight, 220)
      ta.style.height = newH + 'px'
      ta.style.overflowY = ta.scrollHeight > 220 ? 'auto' : 'hidden'
      // set pill multiline class when textarea is taller than single-line or contains newline
      const pillEl = ta.closest('.composer-pill')
      const isMulti = value.includes('\n') || ta.scrollHeight > 36
      if (pillEl) {
        if (isMulti) pillEl.classList.add('is-multiline')
        else pillEl.classList.remove('is-multiline')
      }
    } catch (e) {}
  }, [value])

  // sync when chat changes - load messages from database if needed
  useEffect(() => {
    async function loadMessages() {
      if (!chat?.sessionId) {
        setMessages([])
        return
      }
      
      // Nếu đã có messages từ initial load, dùng luôn
      if (chat.messages && chat.messages.length > 0) {
        setMessages(chat.messages)
        return
      }
      
      // Nếu chưa có, load từ database
      try {
        const response = await fetch(
          `http://localhost:8000/chat/sessions/${chat.sessionId}/messages?limit=100`
        )
        if (!response.ok) throw new Error('Failed to load messages')
        const data = await response.json()
        
        const loadedMessages = await Promise.all((data.messages || []).map(async (msg) => {
          // Kiểm tra nếu message có file_upload_id (là kết quả upload file)
          const hasFileUpload = msg.file_upload_id !== null && msg.file_upload_id !== undefined
          
          // Nếu có file_upload_id, load classification results từ DB
          if (hasFileUpload) {
            try {
              const resultsResponse = await fetch(
                `http://localhost:8000/classification-results/${msg.file_upload_id}`
              )
              if (resultsResponse.ok) {
                const resultsData = await resultsResponse.json()
                // Lưu vào window để download
                window.lastUploadResults = {
                  results: resultsData.results,
                  file_upload_id: msg.file_upload_id
                }
                console.log('Loaded classification results for file_upload_id:', msg.file_upload_id)
              }
            } catch (err) {
              console.warn('Failed to load classification results:', err)
            }
          }
          
          return {
            id: msg.id,
            role: msg.role,
            text: msg.content,
            time: new Date(msg.timestamp).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit'
            }),
            // Show download button nếu message có file_upload_id
            hasDownloadButton: hasFileUpload,
            file_upload_id: msg.file_upload_id
          }
        }))
        
        setMessages(loadedMessages)
        
        // Update chat object để không phải load lại
        if (onUpdateChat) {
          onUpdateChat({ ...chat, messages: loadedMessages })
        }
      } catch (err) {
        console.warn('Failed to load messages:', err)
        setMessages([])
      }
    }
    
    loadMessages()
  }, [chat?.id, chat?.sessionId])

  function sendMessage() {
    const text = value.trim()
    if (!text) return
    const now = new Date()
    const time = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    const userMsg = { id: Date.now(), role: 'user', text, time }
    const next = [...messages, userMsg]
    setMessages(next)
    setValue('')
    setSending(true)
    
    // Lưu user message vào database
    saveMessageToDB('user', text)
    // Append a placeholder assistant "typing" message so user sees immediate feedback
    const placeholder = { id: Date.now() + 9999, role: 'assistant', text: 'Đang phân loại...', time: '', animate: true, typing: true }
    pendingRef.current = placeholder.id
    const nextWithPlaceholder = [...next, placeholder]
    setMessages(nextWithPlaceholder)
    // send text to backend classifier
    ;(async () => {
      try {
        const resp = await fetch('http://localhost:8000/classify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        })
        if (!resp.ok) {
          const errText = await resp.text()
          throw new Error(errText || `HTTP ${resp.status}`)
        }
        const data = await resp.json()
        // Expecting { results: [ { text, label, raw }, ... ] }
        let botText = ''
        if (data && Array.isArray(data.results)) {
          botText = data.results
            .map((r, i) => {
              // show the original line and the label; if raw differs, include details
              const details = r.raw && r.raw !== r.label ? ` — ${r.raw}` : ''
              return `${i + 1}. ${r.label}${details}\n${r.text}`
            })
            .join('\n\n')
        } else if (data && data.label) {
          // fallback to legacy single-label response
          botText = `Phân loại: ${data.label}` + (data.raw && data.raw !== data.label ? `\n\nChi tiết: ${data.raw}` : '')
        } else {
          botText = 'Không nhận được kết quả hợp lệ từ server.'
        }

        const bot = {
          id: Date.now() + 1,
          role: 'assistant',
          text: botText,
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          animate: true
        }
        
        // Lưu assistant message vào database
        saveMessageToDB('assistant', botText)
        
        // replace placeholder message if present, otherwise append
        setMessages((prev) => {
          if (!pendingRef.current) return [...prev, bot]
          const idx = prev.findIndex((m) => m.id === pendingRef.current)
          if (idx >= 0) {
            const copy = [...prev]
            copy.splice(idx, 1, bot)
            pendingRef.current = null
            return copy
          }
          return [...prev, bot]
        })
      } catch (err) {
        const bot = {
          id: Date.now() + 2,
          role: 'assistant',
          text: `Lỗi khi gọi server: ${err.message}`,
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          animate: true
        }
        setMessages((prev) => {
          if (!pendingRef.current) return [...prev, bot]
          const idx = prev.findIndex((m) => m.id === pendingRef.current)
          if (idx >= 0) {
            const copy = [...prev]
            copy.splice(idx, 1, bot)
            pendingRef.current = null
            return copy
          }
          return [...prev, bot]
        })
      } finally {
        setSending(false)
      }
    })()
  }

  function handleKey(e) {
    // Enter sends message; Shift+Enter inserts newline
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  function triggerUpload() {
    // Show context menu or directly trigger bulk upload
    if (bulkFileRef.current) bulkFileRef.current.click()
  }

  async function handleBulkFile(e) {
    const f = e.target.files?.[0]
    if (!f) return

    const lower = f.name.toLowerCase()
    if (!lower.match(/\.(csv|xlsx|xls)$/)) {
      setNotice('❌ Chỉ hỗ trợ CSV, XLSX, XLS')
      setTimeout(() => setNotice(''), 3000)
      e.target.value = ''
      return
    }

    setBulkFile(f)
    e.target.value = ''
  }

  async function uploadBulkFile() {
    if (!bulkFile) {
      setNotice('❌ Chưa chọn file')
      return
    }

    setBulkLoading(true)
    setNotice('')
    
    try {
      const formData = new FormData()
      formData.append('file', bulkFile)

      // Tạo chat item của user khi upload file
      const now = new Date()
      const time = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      const userMsg = {
        id: Date.now(),
        role: 'user',
        text: `📄 Đã upload file: ${bulkFile.name}`,
        time
      }
      
      // Lưu upload message vào database
      saveMessageToDB('user', `📄 Đã upload file: ${bulkFile.name}`)
      
      // Thêm bubble bot trạng thái typing
      const botTyping = {
        id: Date.now() + 9999,
        role: 'assistant',
        text: '',
        time: '',
        animate: true,
        typing: true
      }
      const nextUser = [...messages, userMsg, botTyping]
      setMessages(nextUser)

      // Gọi API upload với session_id
      const uploadUrl = chat?.sessionId 
        ? `http://localhost:8000/upload?session_id=${chat.sessionId}`
        : 'http://localhost:8000/upload'
      const response = await fetch(uploadUrl, {
        method: 'POST',
        body: formData
      })
      if (!response.ok) {
        let errDetail = `HTTP ${response.status}`
        try {
          const errData = await response.json()
          errDetail = errData.detail || errDetail
        } catch (e) {
          const errText = await response.text()
          errDetail = errText || errDetail
        }
        throw new Error(errDetail)
      }
      const data = await response.json()
      const fileUploadId = data.file_upload_id
      
      // Lưu results vào memory để download
      window.lastUploadResults = {
        filename: bulkFile.name,
        results: data.results,
        file_upload_id: fileUploadId
      }
      
      // Tính số lượng bug cho từng label
      const labelCounts = {};
      (data.results || []).forEach(item => {
        const label = item.label || 'Chưa phân loại';
        labelCounts[label] = (labelCounts[label] || 0) + 1;
      });
      let labelSummary = Object.entries(labelCounts)
        .map(([label, count]) => `- ${label}: ${count} bug`)
        .join('\n');
      const bot = {
        id: Date.now() + 1,
        role: 'assistant',
        text: `✅ Đã phân loại ${data.classified_rows}/${data.total_rows} bugs.\n${labelSummary}`,
        time,
        animate: true,
        hasDownloadButton: true // Flag for download button
      }
      
      // Lưu classification result vào database với file_upload_id
      saveMessageToDB('assistant', `✅ Đã phân loại ${data.classified_rows}/${data.total_rows} bugs.\n${labelSummary}`, fileUploadId)
      
      // Xóa bubble bot typing cuối cùng, thêm bot kết quả
      setMessages((prev) => {
        const arr = [...prev]
        // Tìm và thay thế bubble bot typing cuối cùng
        const idx = arr.findIndex(m => m.role === 'assistant' && m.typing)
        if (idx >= 0) {
          arr.splice(idx, 1, bot)
          return arr
        }
        return [...arr, bot]
      })
      setBulkFile(null)
      setNotice('✅ Upload thành công! Có thể tải Excel')
      setTimeout(() => setNotice(''), 2000)
    } catch (err) {
      console.error('Upload error:', err)
      setNotice(`❌ Upload lỗi: ${err.message}`)
      setTimeout(() => setNotice(''), 3000)
    } finally {
      setBulkLoading(false)
    }
  }

  async function downloadExcel(messageWithUpload) {
    // Nếu có file_upload_id trong message, load results từ DB
    if (messageWithUpload?.file_upload_id && (!window.lastUploadResults || window.lastUploadResults.file_upload_id !== messageWithUpload.file_upload_id)) {
      try {
        const resultsResponse = await fetch(
          `http://localhost:8000/classification-results/${messageWithUpload.file_upload_id}`
        )
        if (resultsResponse.ok) {
          const resultsData = await resultsResponse.json()
          window.lastUploadResults = {
            results: resultsData.results,
            file_upload_id: messageWithUpload.file_upload_id
          }
        }
      } catch (err) {
        console.error('Failed to load classification results:', err)
      }
    }
    
    if (!window.lastUploadResults) {
      setNotice('❌ Không có kết quả để tải')
      return
    }

    try {
      const response = await fetch('http://localhost:8000/download-excel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ results: window.lastUploadResults.results })
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `classification_${Date.now()}.xlsx`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
      
      setNotice('✅ Tải Excel thành công!')
      setTimeout(() => setNotice(''), 2000)
    } catch (err) {
      setNotice(`❌ Tải Excel lỗi: ${err.message}`)
      setTimeout(() => setNotice(''), 3000)
    }
  }

  async function handleFile(e) {
    const f = e.target.files?.[0]
    if (!f) return

    const name = f.name
    const lower = name.toLowerCase()

    if (lower.endsWith('.csv')) {
      // Try dynamic import of papaparse, fallback to simple parser
      try {
        const mod = await import('papaparse')
        const Papa = mod.default ?? mod
        Papa.parse(f, {
          preview: 5,
          complete: (res) => {
            const preview = (res.data || []).map((r) => r.join(', ')).join('\n')
            appendFileMessage(name, preview)
          }
        })
      } catch (err) {
        // fallback: simple CSV read
        const reader = new FileReader()
        reader.onload = (ev) => {
          const text = ev.target.result || ''
          const rows = text
            .split(/\r?\n/)
            .filter(Boolean)
            .slice(0, 5)
            .map((r) => r.split(',').map((c) => c.trim()))
          const preview = rows.map((r) => r.join(', ')).join('\n')
          appendFileMessage(name, preview)
        }
        reader.readAsText(f)
      }
    } else if (lower.endsWith('.xlsx') || lower.endsWith('.xls')) {
      // Try dynamic import of xlsx, fallback to reading as text
      try {
        const XLSXmod = await import('xlsx')
        const XLSX = XLSXmod.default ?? XLSXmod
        const reader = new FileReader()
        reader.onload = (ev) => {
          const data = new Uint8Array(ev.target.result)
          const workbook = XLSX.read(data, { type: 'array' })
          const sheetName = workbook.SheetNames[0]
          const sheet = workbook.Sheets[sheetName]
          const rows = XLSX.utils.sheet_to_json(sheet, { header: 1 })
          const previewRows = rows.slice(0, 5).map((r) => (Array.isArray(r) ? r.join(', ') : String(r))).join('\n')
          appendFileMessage(name, previewRows)
        }
        reader.readAsArrayBuffer(f)
      } catch (err) {
        const reader = new FileReader()
        reader.onload = (ev) => {
          const text = ev.target.result || ''
          const preview = text.slice(0, 1000)
          appendFileMessage(name, preview)
        }
        reader.readAsText(f)
      }
    } else {
      appendFileMessage(name, '[Unsupported file type]')
    }
    // reset input
    e.target.value = ''
  }

  function appendFileMessage(filename, previewText) {
    const now = new Date()
    const time = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    const msg = { id: Date.now(), role: 'user', text: `Uploaded file: ${filename}\n\nPreview:\n${previewText}`, time }
    const next = [...messages, msg]
    setMessages(next)
  }

  // Extract possible bug items from messages and trigger CSV download
  function exportClassifiedBugs() {
    const items = []
    const rows = []
    const candidates = messages || []
    candidates.forEach((m) => {
      // look for preview block after 'Preview:' or lines in assistant responses
      const text = String(m.text || '')
      const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean)
      // if message contains 'Preview:' take following lines
      const previewIndex = lines.findIndex((l) => /preview[:]?/i.test(l))
      const toScan = previewIndex >= 0 ? lines.slice(previewIndex + 1) : lines
      toScan.forEach((line) => {
        // treat CSV-like lines (comma separated with >=2 fields)
        if ((line.match(/,/g) || []).length >= 1) {
          const parts = line.split(',').map((p) => p.trim())
          rows.push(parts)
        } else if (/^[\d]+[\).\-\s]/.test(line) || /^[-•\*]/.test(line)) {
          // numbered or bulleted lines
          const cleaned = line.replace(/^[\d]+[\).\-\s]*/, '').replace(/^[-•\*]\s*/, '')
          items.push([cleaned])
        } else if (/bug|issue|defect|lỗi/i.test(line) && line.length > 8) {
          items.push([line])
        }
      })
    })

    // Prefer structured rows if present
    const finalRows = rows.length ? rows : (items.length ? items : [])
    if (!finalRows.length) {
      setNotice('Không tìm thấy mục bug đã phân loại trong đoạn chat.')
      setTimeout(() => setNotice(''), 3000)
      return
    }

    // Build CSV content
    const csvLines = finalRows.map((r) => r.map((c) => '"' + String(c).replace(/"/g, '""') + '"').join(','))
    const csv = csvLines.join('\n')
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    const name = (chat?.title || 'chat').replace(/[^a-z0-9\-_]/gi, '_')
    a.download = `${name}_classified_bugs.csv`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
    setNotice('Đã tải xuống file CSV.')
    setTimeout(() => setNotice(''), 2000)
  }

  return (
    <div className="chat-root">
      <div className="chat-toolbar">
        <button className="export-btn" onClick={exportClassifiedBugs} title="Tải xuống bug đã phân loại">
          Tải bug đã phân loại
        </button>
        {notice && <div className="chat-notice">{notice}</div>}
      </div>

      <div className="messages" ref={listRef}>
        {messages.map((m) => (
          <Message 
            key={m.id} 
            role={m.role} 
            text={m.text} 
            time={m.time}
            animate={m.animate}
            typing={m.typing}
            hasDownloadButton={m.hasDownloadButton}
            onDownload={m.hasDownloadButton ? () => downloadExcel(m) : null}
          />
        ))}
      </div>

      <div className="composer">
        <input ref={fileRef} type="file" accept=".csv,.xlsx,.xls" onChange={handleFile} style={{ display: 'none' }} />
        <input 
          ref={bulkFileRef} 
          type="file" 
          accept=".csv,.xlsx,.xls" 
          onChange={handleBulkFile} 
          style={{ display: 'none' }} 
        />

        <div className="composer-pill" aria-hidden={!chat}>
          <textarea
            ref={composeRef}
            className="composer-input"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKey}
            placeholder={chat ? 'Mô tả lỗi hoặc dán danh sách bug; hoặc đính kèm file (.csv, .xlsx)' : 'Select or create a chat to start'}
            disabled={!chat}
            rows={1}
          />

          <div className="pill-actions">
            <button 
              className="icon-btn mic" 
              title="Voice"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 1v10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/><rect x="7" y="3" width="10" height="14" rx="5" stroke="currentColor" strokeWidth="1.8"/></svg>
            </button>
            <button
              className="icon-btn send"
              onClick={bulkFile ? uploadBulkFile : sendMessage}
              disabled={!chat || sending || (bulkFile ? bulkLoading : !value.trim())}
              title={bulkFile ? "📤 Upload & Gửi" : "Gửi"}
            >
              {bulkFile ? "📤" : (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M22 2L11 13" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/><path d="M22 2L15 22l-4-9-9-4 20-7z" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/></svg>
              )}
            </button>

            {/* Button + */}
            <button 
              onClick={triggerUpload} 
              disabled={!chat}
              title="Upload file"
              className="icon-btn"
              style={{
                width: '32px',
                height: '32px',
                minWidth: '32px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: 0,
                marginLeft: '4px'
              }}
            >
              <span style={{ fontSize: '18px', lineHeight: 1 }}>＋</span>
            </button>

            {/* Model Selector */}
            <div ref={modelMenuRef} style={{ position: 'relative', flexShrink: 0, marginLeft: '8px' }}>
              <button
                onClick={() => setShowModelMenu(!showModelMenu)}
                disabled={!chat}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  padding: '6px 10px',
                  height: '32px',
                  backgroundColor: 'transparent',
                  border: 'none',
                  color: 'var(--text)',
                  fontSize: '13px',
                  fontWeight: '600',
                  cursor: chat ? 'pointer' : 'not-allowed',
                  opacity: chat ? 1 : 0.5,
                  transition: 'all 0.2s ease',
                  whiteSpace: 'nowrap'
                }}
              >
                <span>Smart ({selectedModel})</span>
                <svg 
                  width="10" 
                  height="10" 
                  viewBox="0 0 24 24" 
                  fill="none" 
                  style={{ 
                    transform: showModelMenu ? 'rotate(180deg)' : 'rotate(0deg)',
                    transition: 'transform 0.2s ease',
                    flexShrink: 0
                  }}
                >
                  <path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>

            {/* Dropdown Menu */}
            {showModelMenu && chat && (
              <div
                style={{
                  position: 'absolute',
                  bottom: 'calc(100% + 8px)',
                  right: 0,
                  minWidth: '240px',
                  backgroundColor: 'var(--panel)',
                  backdropFilter: 'blur(20px)',
                  border: '1px solid var(--border)',
                  borderRadius: '12px',
                  boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
                  zIndex: 1000,
                  overflow: 'hidden',
                  animation: 'slideDown 0.15s ease'
                }}
              >
                {/* GPT-5 Option */}
                <div
                  onClick={() => {
                    setSelectedModel('GPT-5')
                    setShowModelMenu(false)
                  }}
                  style={{
                    padding: '12px 16px',
                    cursor: 'pointer',
                    backgroundColor: selectedModel === 'GPT-5' ? 'var(--primary-light)' : 'transparent',
                    transition: 'background-color 0.15s ease'
                  }}
                  onMouseEnter={(e) => {
                    if (selectedModel !== 'GPT-5') {
                      e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)'
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (selectedModel !== 'GPT-5') {
                      e.currentTarget.style.backgroundColor = 'transparent'
                    }
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div style={{ 
                      fontSize: '20px',
                      width: '28px',
                      height: '28px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}>
                      ✨
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ 
                        fontWeight: '600', 
                        fontSize: '13px',
                        marginBottom: '2px',
                        color: 'var(--text)'
                      }}>
                        GPT-5
                      </div>
                      <div style={{ 
                        fontSize: '11px', 
                        color: 'var(--text-secondary)',
                        lineHeight: '1.3'
                      }}>
                        Model AI online - Độ chính xác cao
                      </div>
                    </div>
                    {selectedModel === 'GPT-5' && (
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                        <path d="M20 6L9 17l-5-5" stroke="var(--primary)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    )}
                  </div>
                </div>

                {/* Llama Option */}
                <div
                  onClick={() => {
                    setSelectedModel('Llama')
                    setShowModelMenu(false)
                  }}
                  style={{
                    padding: '12px 16px',
                    cursor: 'pointer',
                    backgroundColor: selectedModel === 'Llama' ? 'var(--primary-light)' : 'transparent',
                    transition: 'background-color 0.15s ease'
                  }}
                  onMouseEnter={(e) => {
                    if (selectedModel !== 'Llama') {
                      e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)'
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (selectedModel !== 'Llama') {
                      e.currentTarget.style.backgroundColor = 'transparent'
                    }
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div style={{ 
                      fontSize: '20px',
                      width: '28px',
                      height: '28px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}>
                      ⚡
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ 
                        fontWeight: '600', 
                        fontSize: '13px',
                        marginBottom: '2px',
                        color: 'var(--text)'
                      }}>
                        Llama
                      </div>
                      <div style={{ 
                        fontSize: '11px', 
                        color: 'var(--text-secondary)',
                        lineHeight: '1.3'
                      }}>
                        Model nội bộ - Xử lý offline
                      </div>
                    </div>
                    {selectedModel === 'Llama' && (
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                        <path d="M20 6L9 17l-5-5" stroke="var(--primary)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    )}
                  </div>
                </div>
              </div>
            )}
            </div>
          </div>
        </div>

        {/* Bulk File Status */}
        {bulkFile && (
          <div style={{
            padding: '8px 12px',
            backgroundColor: 'var(--surface)',
            borderTop: '1px solid var(--border)',
            display: 'flex',
            gap: '8px',
            alignItems: 'center',
            fontSize: '13px'
          }}>
            <span style={{ flex: 1 }}>📄 {bulkFile.name}</span>
            <button
              onClick={uploadBulkFile}
              disabled={bulkLoading}
              style={{
                padding: '4px 10px',
                backgroundColor: '#28a745',
                color: 'white',
                border: 'none',
                borderRadius: '3px',
                cursor: bulkLoading ? 'not-allowed' : 'pointer',
                fontSize: '12px',
                fontWeight: 'bold',
                opacity: bulkLoading ? 0.6 : 1
              }}
            >
              {bulkLoading ? '⏳' : '📤'}
            </button>
            <button
              onClick={() => setBulkFile(null)}
              disabled={bulkLoading}
              style={{
                padding: '4px 8px',
                backgroundColor: '#6c757d',
                color: 'white',
                border: 'none',
                borderRadius: '3px',
                cursor: 'pointer',
                fontSize: '12px'
              }}
            >
              ✕
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
