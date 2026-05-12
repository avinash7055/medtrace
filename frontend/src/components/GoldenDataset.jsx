import React from 'react'
import { Database, Star } from 'lucide-react'

export default function GoldenDataset({ dataset }) {
  return (
    <div className="glass" style={{ padding:20, gridColumn:'1 / -1' }}>
      <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:16 }}>
        <Database size={18} color="var(--cyan)" />
        <span style={{ fontWeight:600 }}>Golden Dataset</span>
        <span style={{ marginLeft:'auto', fontSize:'0.75rem', color:'var(--text-muted)' }}>
          Curated automatically from high-scoring production traces
        </span>
      </div>

      {!dataset?.length ? (
        <div style={{ textAlign:'center', padding:'30px 0', color:'var(--text-muted)', fontSize:'0.85rem' }}>
          <Star size={32} style={{ opacity:0.3, display:'block', margin:'0 auto 10px' }} />
          No golden examples yet. High-scoring answers (≥8.0) will appear here.
        </div>
      ) : (
        <div style={{ overflowX:'auto' }}>
          <table style={{ width:'100%', borderCollapse:'collapse', fontSize:'0.8rem', textAlign:'left' }}>
            <thead>
              <tr style={{ borderBottom:'1px solid var(--border)' }}>
                <th style={{ padding:'10px 14px', color:'var(--text-muted)', fontWeight:500 }}>Score</th>
                <th style={{ padding:'10px 14px', color:'var(--text-muted)', fontWeight:500, width:'30%' }}>Question</th>
                <th style={{ padding:'10px 14px', color:'var(--text-muted)', fontWeight:500 }}>Answer Snippet</th>
                <th style={{ padding:'10px 14px', color:'var(--text-muted)', fontWeight:500 }}>Version</th>
              </tr>
            </thead>
            <tbody>
              {dataset.map((d, i) => (
                <tr key={i} style={{ borderBottom:'1px solid rgba(255,255,255,0.03)' }} className="glass-hover">
                  <td style={{ padding:'10px 14px' }}>
                    <span style={{ color:'var(--green)', fontWeight:600 }}>{d.avg_score?.toFixed(1)}</span>
                  </td>
                  <td style={{ padding:'10px 14px', color:'var(--text-primary)' }}>{d.query}</td>
                  <td style={{ padding:'10px 14px', color:'var(--text-secondary)' }}>
                    {d.answer.substring(0, 100)}…
                  </td>
                  <td style={{ padding:'10px 14px', fontFamily:'JetBrains Mono, monospace', color:'var(--purple-light)', fontSize:'0.75rem' }}>
                    {d.prompt_version}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
