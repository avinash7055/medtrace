import React from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer, Legend } from 'recharts'
import { TrendingUp } from 'lucide-react'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background:'rgba(15,15,30,0.95)', border:'1px solid rgba(124,58,237,0.4)', borderRadius:8, padding:'10px 14px', fontSize:'0.8rem' }}>
      <p style={{ color:'var(--text-muted)', marginBottom:4 }}>Query #{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: <strong>{p.value?.toFixed(2)}</strong>
        </p>
      ))}
    </div>
  )
}

export default function EvolutionGraph({ scoreHistory, evolutionEvents }) {
  const data = scoreHistory.map((s, i) => ({
    query: i + 1,
    score: s.avg_score,
    accuracy: s.eval_scores?.medical_accuracy,
    safety: s.eval_scores?.safety,
  }))

  const hasData = data.length > 0
  const latestScore = hasData ? data[data.length - 1].score : 0
  const firstScore = hasData ? data[0].score : 0
  const improvement = latestScore - firstScore

  return (
    <div className="glass" style={{ padding:20 }}>
      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:16 }}>
        <div style={{ display:'flex', alignItems:'center', gap:8 }}>
          <TrendingUp size={18} color="var(--purple-light)" />
          <span style={{ fontWeight:600 }}>Score Evolution</span>
        </div>
        {hasData && (
          <div style={{ display:'flex', gap:12, fontSize:'0.8rem' }}>
            <span style={{ color:'var(--text-muted)' }}>Latest: <span style={{ color: latestScore >= 7.5 ? 'var(--green)' : latestScore >= 6 ? 'var(--yellow)' : 'var(--red)', fontWeight:700 }}>{latestScore.toFixed(1)}</span></span>
            {improvement !== 0 && <span style={{ color: improvement > 0 ? 'var(--green)' : 'var(--red)' }}>{improvement > 0 ? '▲' : '▼'} {Math.abs(improvement).toFixed(1)}</span>}
          </div>
        )}
      </div>

      {!hasData ? (
        <div style={{ height:200, display:'flex', alignItems:'center', justifyContent:'center', color:'var(--text-muted)', flexDirection:'column', gap:8 }}>
          <TrendingUp size={32} opacity={0.3} />
          <p style={{ fontSize:'0.85rem' }}>Ask a question to see score evolution</p>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={data} margin={{ top:5, right:10, bottom:5, left:-20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="query" tick={{ fill:'var(--text-muted)', fontSize:11 }} label={{ value:'Query #', position:'insideBottom', fill:'var(--text-muted)', fontSize:11, offset:-2 }} />
            <YAxis domain={[0, 10]} tick={{ fill:'var(--text-muted)', fontSize:11 }} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize:'0.75rem', color:'var(--text-muted)' }} />
            <ReferenceLine y={6.5} stroke="rgba(239,68,68,0.5)" strokeDasharray="4 4" label={{ value:'Threshold', fill:'rgba(239,68,68,0.7)', fontSize:10, position:'insideTopRight' }} />
            <Line type="monotone" dataKey="score" stroke="var(--purple-light)" strokeWidth={2.5} dot={{ fill:'var(--purple)', r:3 }} activeDot={{ r:5 }} name="Avg Score" />
            <Line type="monotone" dataKey="accuracy" stroke="var(--cyan)" strokeWidth={1.5} dot={false} name="Accuracy" strokeDasharray="4 2" />
            <Line type="monotone" dataKey="safety" stroke="var(--green)" strokeWidth={1.5} dot={false} name="Safety" strokeDasharray="4 2" />
          </LineChart>
        </ResponsiveContainer>
      )}

      {evolutionEvents?.length > 0 && (
        <div style={{ marginTop:12 }}>
          <p style={{ fontSize:'0.75rem', color:'var(--text-muted)', marginBottom:6 }}>Recent Evolutions</p>
          {evolutionEvents.slice(-3).map((e, i) => (
            <div key={i} style={{ display:'flex', justifyContent:'space-between', padding:'6px 10px', background:'rgba(124,58,237,0.08)', borderRadius:6, marginBottom:4, fontSize:'0.75rem' }}>
              <span style={{ color:'var(--purple-light)' }}>{e.old_version} → {e.new_version}</span>
              <span style={{ color:'var(--green)' }}>+{e.improvement?.toFixed(1)}</span>
              <span style={{ color:'var(--text-muted)', maxWidth:200, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{e.root_cause}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
