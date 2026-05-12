import React from 'react'
import { BarChart2, Zap, Database, Award, RefreshCw } from 'lucide-react'

function Metric({ icon: Icon, label, value, color, sub }) {
  return (
    <div className="glass glass-hover" style={{ padding:'16px 20px', display:'flex', alignItems:'center', gap:14 }}>
      <div style={{ width:40, height:40, borderRadius:10, background:`rgba(${color},0.15)`, display:'flex', alignItems:'center', justifyContent:'center', flexShrink:0 }}>
        <Icon size={18} color={`rgb(${color})`} />
      </div>
      <div>
        <div style={{ fontSize:'1.5rem', fontWeight:700, color:`rgb(${color})`, lineHeight:1 }}>{value}</div>
        <div style={{ fontSize:'0.75rem', color:'var(--text-muted)', marginTop:2 }}>{label}</div>
        {sub && <div style={{ fontSize:'0.7rem', color:'var(--text-muted)', opacity:0.7 }}>{sub}</div>}
      </div>
    </div>
  )
}

export default function MetricsPanel({ metrics, onTriggerEvolution }) {
  if (!metrics) return null
  const { total_queries, avg_score, evolutions_run, golden_examples, current_prompt_version, evolution_running } = metrics

  const scoreColor = avg_score >= 7.5 ? '16,185,129' : avg_score >= 6 ? '245,158,11' : '239,68,68'

  return (
    <div className="glass" style={{ padding:20 }}>
      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:16 }}>
        <div style={{ display:'flex', alignItems:'center', gap:8 }}>
          <BarChart2 size={18} color="var(--purple-light)" />
          <span style={{ fontWeight:600 }}>Live Metrics</span>
        </div>
        <button
          className="btn-secondary"
          onClick={onTriggerEvolution}
          disabled={evolution_running}
          id="trigger-evolution-btn"
          style={{ fontSize:'0.75rem' }}
        >
          {evolution_running
            ? <><RefreshCw size={12} className="spin" /> Evolving…</>
            : <><Zap size={12} /> Trigger Evolution</>
          }
        </button>
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:10 }}>
        <Metric icon={BarChart2} label="Total Queries" value={total_queries ?? 0} color="124,58,237" />
        <Metric icon={Award}    label="Avg Score" value={avg_score != null ? avg_score.toFixed(1) : '–'} color={scoreColor} sub="/10" />
        <Metric icon={Zap}      label="Evolutions" value={evolutions_run ?? 0} color="245,158,11" sub="cycles run" />
        <Metric icon={Database} label="Golden Examples" value={golden_examples ?? 0} color="6,182,212" sub="curated" />
      </div>

      <div style={{ marginTop:12, padding:'8px 12px', background:'rgba(124,58,237,0.08)', borderRadius:8, display:'flex', justifyContent:'space-between', fontSize:'0.78rem' }}>
        <span style={{ color:'var(--text-muted)' }}>Active Prompt</span>
        <span style={{ color:'var(--purple-light)', fontWeight:600, fontFamily:'JetBrains Mono, monospace' }}>{current_prompt_version ?? 'v1'}</span>
      </div>
    </div>
  )
}
