import React, { useState, useRef, useEffect } from 'react'
import { Send, AlertCircle, BookOpen, Cpu, RefreshCw } from 'lucide-react'

function ScoreBadge({ value, label }) {
  const cls = value >= 7.5 ? 'score-green' : value >= 6 ? 'score-yellow' : 'score-red'
  return (
    <span className={`score-badge ${cls}`}>
      {label}: {value?.toFixed(1)}
    </span>
  )
}

function Message({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`msg-row ${isUser ? 'msg-user' : 'msg-agent'} animate-fade-in`}>
      {!isUser && (
        <div className="msg-avatar">
          <Cpu size={14} />
        </div>
      )}
      <div className="msg-bubble">
        {isUser ? (
          <p>{msg.content}</p>
        ) : (
          <>
            <p style={{ whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>{msg.content}</p>
            {msg.scores && (
              <div className="msg-scores">
                {Object.entries(msg.scores).map(([k, v]) => (
                  <ScoreBadge key={k} label={k.replace(/_/g,' ')} value={v} />
                ))}
                <span className="score-badge" style={{ background:'rgba(124,58,237,0.15)', color:'#9d6ef8', border:'1px solid rgba(124,58,237,0.3)' }}>
                  avg: {msg.avg?.toFixed(1)}
                </span>
              </div>
            )}
            {msg.citations?.length > 0 && (
              <div className="msg-citations">
                <BookOpen size={11} />
                {msg.citations.map((c, i) => <span key={i}>{c}</span>)}
              </div>
            )}
            {msg.version && (
              <div className="msg-meta">
                <span>Prompt {msg.version}</span>
                {msg.evolution && <span style={{ color:'#f59e0b' }}>⚡ Evolution triggered</span>}
                <span>{msg.time_ms}ms</span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default function ChatInterface({ onNewResult }) {
  const [messages, setMessages] = useState([
    {
      role: 'agent',
      content: "Hello! I'm MedTrace, your AI medical information assistant powered by Gemini + LangGraph. I answer medical questions, evaluate my own responses, and autonomously improve via Phoenix MCP.\n\nAsk me about drug interactions, symptoms, dosages, or treatment protocols.",
    }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async () => {
    const q = input.trim()
    if (!q || loading) return
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: q }])
    setLoading(true)
    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q }),
      })
      const data = await res.json()
      setMessages(prev => [...prev, {
        role: 'agent',
        content: data.answer,
        scores: data.eval_scores,
        avg: data.avg_score,
        citations: data.citations,
        version: data.prompt_version,
        evolution: data.evolution_triggered,
        time_ms: Math.round(data.processing_time_ms),
      }])
      onNewResult?.(data)
    } catch (e) {
      setMessages(prev => [...prev, {
        role: 'agent',
        content: `⚠️ Connection error: ${e.message}. Is the backend running on port 8000?`,
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="chat-container glass">
      <div className="chat-header">
        <div className="chat-header-left">
          <div className="status-dot" />
          <span>MedTrace Agent</span>
        </div>
        <span style={{ fontSize:'0.75rem', color:'var(--text-muted)' }}>Gemini 2.0 Flash + LangGraph</span>
      </div>

      <div className="chat-messages">
        {messages.map((m, i) => <Message key={i} msg={m} />)}
        {loading && (
          <div className="msg-row msg-agent animate-fade-in">
            <div className="msg-avatar"><Cpu size={14} /></div>
            <div className="msg-bubble">
              <div className="typing-dots"><span/><span/><span/></div>
              <p style={{ fontSize:'0.75rem', color:'var(--text-muted)', marginTop:4 }}>Thinking + evaluating…</p>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-row">
        <input
          className="chat-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
          placeholder="Ask a medical question…"
          disabled={loading}
        />
        <button className="btn-primary" onClick={send} disabled={loading || !input.trim()} id="send-btn">
          {loading ? <RefreshCw size={16} className="spin" /> : <Send size={16} />}
        </button>
      </div>
      <p style={{ fontSize:'0.7rem', color:'var(--text-muted)', padding:'6px 16px 12px', textAlign:'center' }}>
        ⚠️ For informational purposes only. Not a substitute for professional medical advice.
      </p>
    </div>
  )
}
