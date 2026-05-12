import React from 'react'
import { Activity } from 'lucide-react'

function ScoreBar({ value }) {
  const color = value >= 7.5 ? 'var(--green)' : value >= 6 ? 'var(--yellow)' : 'var(--red)'
  return (
    <div style={{ display:'flex', alignItems:'center', gap:6 }}>
      <div style={{ flex:1, height:4, background:'rgba(255,255,255,0.08)', borderRadius:2, overflow:'hidden' }}>
        <div style={{ width:`${(value/10)*100}%`, height:'100%', background:color, borderRadius:2, transition:'width 0.5s' }} />
      </div>
      <span style={{ fontSize:'0.72rem', color, fontWeight:600, minWidth:24 }}>{value?.toFixed(1)}</span>
    </div>
  )
}

export default function TraceViewer({ traces }) {
  return (
    <div className="glass" style={{ padding:20 }}>
      <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:14 }}>
        <Activity size={18} color="var(--purple-light)" />
        <span style={{ fontWeight:600 }}>Recent Traces</span>
        <span style={{ marginLeft:'auto', fontSize:'0.73rem', color:'var(--text-muted)' }}>Last {traces?.length ?? 0}</span>
      </div>

      {!traces?.length ? (
        <div style={{ textAlign:'center', padding:'20px 0', color:'var(--text-muted)', fontSize:'0.85rem' }}>
          <Activity size={28} style={{ opacity:0.3, display:'block', margin:'0 auto 8px' }} />
          No traces yet — ask a question
        </div>
      ) : (
        <div style={{ display:'flex', flexDirection:'column', gap:8 }}>
          {traces.map((t, i) => (
            <div key={i} className="glass-hover" style={{ padding:'10px 14px', borderRadius:8, border:'1px solid var(--border)', transition:'all 0.2s', cursor:'default' }}>
              <div style={{ display:'flex', justifyContent:'space-between', marginBottom:4 }}>
                <span style={{ fontSize:'0.8rem', color:'var(--text-primary)', fontWeight:500, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', maxWidth:'60%' }}>{t.query}</span>
                <div style={{ display:'flex', gap:6, flexShrink:0 }}>
                  <span style={{ fontSize:'0.7rem', color:'var(--text-muted)', fontFamily:'JetBrains Mono, monospace' }}>{t.prompt_version}</span>
                  {t.evolution_triggered && <span title="Evolution triggered" style={{ color:'var(--yellow)', fontSize:'0.7rem' }}>⚡</span>}
                </div>
              </div>
              <ScoreBar value={t.avg_score ?? 0} />
              <div style={{ fontSize:'0.68rem', color:'var(--text-muted)', marginTop:4 }}>
                {new Date(t.timestamp).toLocaleTimeString()}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
