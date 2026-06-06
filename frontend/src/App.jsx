import React, { useState, useEffect, useCallback } from 'react'
import { Activity } from 'lucide-react'
import ChatInterface from './components/ChatInterface'
import EvolutionGraph from './components/EvolutionGraph'
import MetricsPanel from './components/MetricsPanel'
import TraceViewer from './components/TraceViewer'
import GoldenDataset from './components/GoldenDataset'

export default function App() {
  const [metrics, setMetrics] = useState(null)
  const [traces, setTraces] = useState([])
  const [evolutionHistory, setEvolutionHistory] = useState([])
  const [goldenDataset, setGoldenDataset] = useState([])

  const fetchDashboardData = useCallback(async () => {
    try {
      const [mRes, tRes, eRes, gRes] = await Promise.all([
        fetch('/api/metrics').then(r => r.json()),
        fetch('/api/traces').then(r => r.json()),
        fetch('/api/evolution/history').then(r => r.json()),
        fetch('/api/golden-dataset').then(r => r.json())
      ])
      setMetrics(mRes)
      setTraces(tRes)
      setEvolutionHistory(eRes)
      setGoldenDataset(gRes)
    } catch (e) {
      console.error('Failed to fetch dashboard data:', e)
    }
  }, [])

  useEffect(() => {
    fetchDashboardData()
    const interval = setInterval(fetchDashboardData, 15000)
    return () => clearInterval(interval)
  }, [fetchDashboardData])

  const triggerEvolution = async () => {
    try {
      await fetch('/api/evolution/trigger', { method: 'POST' })
      fetchDashboardData()
    } catch (e) {
      console.error('Failed to trigger evolution:', e)
    }
  }

  return (
    <div className="app-layout">
      <header className="app-header">
        <div style={{ display:'flex', alignItems:'center', gap:12 }}>
          <div style={{ background:'linear-gradient(135deg, var(--purple), var(--cyan))', padding:8, borderRadius:12 }}>
            <Activity color="white" size={24} />
          </div>
          <div>
            <h1 style={{ fontSize:'1.4rem', fontWeight:800, margin:0, letterSpacing:'-0.02em', background:'linear-gradient(to right, #fff, #a5b4fc)', WebkitBackgroundClip:'text', color:'transparent' }}>MedTrace</h1>
            <p style={{ fontSize:'0.8rem', color:'var(--text-muted)', margin:0 }}>Self-Evolving Medical AI Agent</p>
          </div>
        </div>
        <div style={{ fontSize:'0.75rem', display:'flex', gap:16, color:'var(--text-muted)' }}>
          <span style={{ display:'flex', alignItems:'center', gap:6 }}><div style={{ width:8, height:8, borderRadius:'50%', background:'var(--green)' }}/> Phoenix MCP Connected</span>
          <span style={{ display:'flex', alignItems:'center', gap:6 }}><div style={{ width:8, height:8, borderRadius:'50%', background:'var(--green)' }}/> ChromaDB Active</span>
        </div>
      </header>

      <main className="dashboard-grid">
        {/* Left Column - Chat */}
        <div className="chat-section">
          <ChatInterface onNewResult={fetchDashboardData} />
        </div>

        {/* Right Column - Dashboard */}
        <div className="metrics-section">
          <MetricsPanel metrics={metrics} onTriggerEvolution={triggerEvolution} />
          <EvolutionGraph scoreHistory={traces.slice().reverse()} evolutionEvents={evolutionHistory} />
        </div>

        {/* Bottom Section */}
        <div className="traces-section">
          <TraceViewer traces={traces} />
        </div>
        <div className="golden-section">
          <GoldenDataset dataset={goldenDataset} />
        </div>
      </main>

      <style>{`
        .app-layout {
          max-width: 1400px;
          margin: 0 auto;
          padding: 24px;
        }
        .app-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 24px;
          padding: 16px 24px;
          background: rgba(255,255,255,0.02);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          backdrop-filter: blur(12px);
        }
        .dashboard-grid {
          display: grid;
          grid-template-columns: 1.2fr 1fr;
          grid-template-areas:
            "chat metrics"
            "chat traces"
            "golden golden";
          gap: 20px;
        }
        .chat-section { grid-area: chat; display: flex; flex-direction: column; min-height: 600px; }
        .metrics-section { grid-area: metrics; display: flex; flex-direction: column; gap: 20px; }
        .traces-section { grid-area: traces; }
        .golden-section { grid-area: golden; margin-top: 10px; }

        @media (max-width: 1024px) {
          .dashboard-grid {
            grid-template-columns: 1fr;
            grid-template-areas:
              "metrics"
              "chat"
              "traces"
              "golden";
          }
          .chat-section { min-height: 500px; }
        }

        /* Chat Specific Styles */
        .chat-container {
          display: flex;
          flex-direction: column;
          height: 100%;
          overflow: hidden;
        }
        .chat-header {
          padding: 14px 20px;
          border-bottom: 1px solid var(--border);
          display: flex;
          justify-content: space-between;
          align-items: center;
          background: rgba(0,0,0,0.2);
        }
        .chat-header-left {
          display: flex;
          align-items: center;
          gap: 10px;
          font-weight: 600;
          font-size: 0.9rem;
        }
        .status-dot {
          width: 8px; height: 8px;
          background: var(--green);
          border-radius: 50%;
          box-shadow: 0 0 10px var(--green);
        }
        .chat-messages {
          flex: 1;
          overflow-y: auto;
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 20px;
        }
        .msg-row { display: flex; gap: 12px; max-width: 85%; }
        .msg-user { align-self: flex-end; flex-direction: row-reverse; }
        .msg-agent { align-self: flex-start; }
        
        .msg-avatar {
          width: 28px; height: 28px;
          border-radius: 8px;
          background: linear-gradient(135deg, var(--purple), var(--cyan));
          display: flex; alignItems: center; justify-content: center;
          flex-shrink: 0;
          margin-top: 4px;
        }
        
        .msg-bubble {
          padding: 14px 18px;
          border-radius: 16px;
          font-size: 0.9rem;
          position: relative;
        }
        .msg-user .msg-bubble {
          background: rgba(255,255,255,0.08);
          border-bottom-right-radius: 4px;
        }
        .msg-agent .msg-bubble {
          background: rgba(124,58,237,0.08);
          border: 1px solid rgba(124,58,237,0.2);
          border-bottom-left-radius: 4px;
        }

        .msg-scores {
          display: flex; flex-wrap: wrap; gap: 6px;
          margin-top: 12px; padding-top: 12px;
          border-top: 1px solid rgba(255,255,255,0.05);
        }
        .msg-citations {
          display: flex; flex-wrap: wrap; gap: 8px; alignItems: center;
          margin-top: 10px; font-size: 0.75rem; color: var(--cyan);
        }
        .msg-meta {
          display: flex; gap: 12px; margin-top: 10px;
          font-size: 0.7rem; color: var(--text-muted); fontFamily: 'JetBrains Mono', monospace;
        }

        .chat-input-row {
          padding: 16px 20px 8px;
          display: flex; gap: 12px;
        }
        .chat-input {
          flex: 1;
          background: rgba(0,0,0,0.3);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 12px 16px;
          color: white;
          font-family: inherit;
          font-size: 0.95rem;
          transition: border-color 0.2s;
        }
        .chat-input:focus {
          outline: none;
          border-color: var(--purple-light);
        }
        .spin { animation: spin 1s linear infinite; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}
