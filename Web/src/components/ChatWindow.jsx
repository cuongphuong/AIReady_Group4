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
  const listRef = useRef(null)
  const fileRef = useRef(null)

  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight
  }, [messages])

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
    if (chat) onUpdateChat({ ...chat, messages: next })

    // simulate bot reply
    setTimeout(() => {
      const bot = {
        id: Date.now() + 1,
        role: 'assistant',
        text: `Phản hồi mẫu: "${text}"`,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
      const next2 = [...messages, userMsg, bot]
      setMessages(next2)
      setSending(false)
      if (chat) onUpdateChat({ ...chat, messages: next2 })
    }, 800)
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  function triggerUpload() {
    if (fileRef.current) fileRef.current.click()
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
          <Message key={m.id} role={m.role} text={m.text} time={m.time} />
        ))}
      </div>

      <div className="composer">
        <input ref={fileRef} type="file" accept=".csv,.xlsx,.xls" onChange={handleFile} style={{ display: 'none' }} />

        <div className="composer-pill" aria-hidden={!chat}>
          <button className="pill-add" onClick={triggerUpload} title="Add file" aria-label="Add">
            <span className="plus">＋</span>
          </button>

          <input
            type="text"
            className="composer-input"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKey}
            placeholder={chat ? 'Mô tả lỗi hoặc dán danh sách bug; hoặc đính kèm file (.csv, .xlsx)' : 'Select or create a chat to start'}
            disabled={!chat}
          />

          <div className="pill-actions">
            <button className="icon-btn mic" title="Voice">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 1v10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/><rect x="7" y="3" width="10" height="14" rx="5" stroke="currentColor" strokeWidth="1.8"/></svg>
            </button>
            <button className="icon-btn send" onClick={sendMessage} disabled={!chat || sending || !value.trim()} title="Gửi">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M22 2L11 13" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/><path d="M22 2L15 22l-4-9-9-4 20-7z" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/></svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
