import React, { useState, useRef, useEffect } from 'react'
import Message from './Message'
// Note: papaparse and xlsx are optional dependencies. We dynamically import them
// when handling files so the dev server does not fail if they are not installed.

export default function ChatWindow({ chat = null, onUpdateChat = () => {} }) {
  const [messages, setMessages] = useState(chat?.messages ?? [
    { id: 1, role: 'assistant', text: 'Chào bạn! Tôi có thể giúp gì cho bạn hôm nay?', time: '09:30' }
  ])
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

  // sync when chat changes
  useEffect(() => {
    setMessages(chat?.messages ?? [])
  }, [chat?.id])

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
    // Append a placeholder assistant "typing" message so user sees immediate feedback
    const placeholder = { id: Date.now() + 9999, role: 'assistant', text: 'Đang phân loại...', time: '', animate: true, typing: true }
    pendingRef.current = placeholder.id
    const nextWithPlaceholder = [...next, placeholder]
    setMessages(nextWithPlaceholder)
    if (chat) onUpdateChat({ ...chat, messages: next })
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
        if (chat) onUpdateChat({ ...chat, messages: [...next.filter(m=>m.role!=='assistant'), bot] })
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
        if (chat) onUpdateChat({ ...chat, messages: [...next.filter(m=>m.role!=='assistant'), bot] })
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
      if (chat) onUpdateChat({ ...chat, messages: nextUser })

      // Gọi API upload
      const response = await fetch('http://localhost:8000/upload', {
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
      window.lastUploadResults = {
        filename: bulkFile.name,
        results: data.results
      }
      // Thay thế bubble bot typing bằng kết quả bot
      const bot = {
        id: Date.now() + 1,
        role: 'assistant',
        text: `✅ Đã phân loại ${data.classified_rows}/${data.total_rows} bugs.`,
        time,
        animate: true,
        hasDownloadButton: true // Flag for download button
      }
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
      if (chat) onUpdateChat({ ...chat, messages: [...nextUser.filter(m=>!(m.role==='assistant'&&m.typing)), bot] })
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

  async function downloadExcel() {
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
    if (chat) onUpdateChat({ ...chat, messages: next })
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
            onDownload={m.hasDownloadButton ? downloadExcel : null}
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
          <button className="pill-add" onClick={triggerUpload} title="Add file" aria-label="Add">
            <span className="plus">＋</span>
          </button>

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
