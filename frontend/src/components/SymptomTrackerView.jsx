import { useCallback, useEffect, useState } from 'react'
import { Activity, Bot, Lightbulb, MessageSquare, Plus } from 'lucide-react'
import { fetchHistory, fetchSymptomInsights, saveSymptom } from '../services/api'

function formatDate(value) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? 'Date unavailable' : date.toLocaleString()
}

export default function SymptomTrackerView({ user, city, onDiscussWithBot }) {
  const [history, setHistory] = useState([])
  const [insight, setInsight] = useState('')
  const [historyStats, setHistoryStats] = useState({
    conversation_count: 0,
    symptom_count: 0,
    ai_response_count: 0,
  })
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const [insightError, setInsightError] = useState('')
  const [showLogModal, setShowLogModal] = useState(false)
  const [newSymptom, setNewSymptom] = useState('')
  const [newIntensity, setNewIntensity] = useState('')
  const [newNotes, setNewNotes] = useState('')

  const loadHistory = useCallback(async () => {
    setLoading(true)
    setHistoryError('')
    setInsightError('')
    const [historyResult, insightResult] = await Promise.allSettled([
      fetchHistory(user),
      fetchSymptomInsights(user),
    ])

    if (historyResult.status === 'fulfilled') {
      const conversation = historyResult.value
      setHistory(conversation.messages)
      setHistoryStats({
        conversation_count: conversation.conversation_count,
        symptom_count: conversation.symptom_count,
        ai_response_count: conversation.ai_response_count,
      })
    } else {
      setHistory([])
      setHistoryStats({ conversation_count: 0, symptom_count: 0, ai_response_count: 0 })
      setHistoryError(historyResult.reason.message || 'Unable to load your health history.')
    }

    if (insightResult.status === 'fulfilled') {
      setInsight(insightResult.value.insight)
    } else {
      setInsight('')
      setInsightError(insightResult.reason.message || 'AI analysis is temporarily unavailable.')
    }
    setLoading(false)
  }, [user])

  useEffect(() => {
    loadHistory()
  }, [loadHistory])

  const handleAddLog = async (event) => {
    event.preventDefault()
    if (!newSymptom.trim() || saving) return
    setSaving(true)
    setHistoryError('')
    try {
      const note = newNotes.trim() ? ` Notes: ${newNotes.trim()}` : ''
      await saveSymptom(`${newSymptom.trim()} (self-reported intensity ${newIntensity}/10).${note}`, city, user)
      setNewSymptom('')
      setNewIntensity('')
      setNewNotes('')
      setShowLogModal(false)
      await loadHistory()
    } catch (saveError) {
      setHistoryError(saveError.message || 'Unable to save your symptom log.')
    } finally {
      setSaving(false)
    }
  }

  const lastActivity = history.length ? formatDate(history[history.length - 1].timestamp) : 'No activity yet'

  return (
    <div className="content-body">
      <div className="symptom-top-row">
        <div>
          <h1 className="page-title">Symptom Tracker</h1>
          <p className="page-subtitle">Your private chat and symptom history, with insights from what you have shared. Records are retained for 30 days.</p>
        </div>
        <button className="log-symptom-btn" onClick={() => setShowLogModal(true)}>
          <Plus size={18} />
          <span>Log Symptom</span>
        </button>
      </div>

      {historyError && <p className="chat-error" role="alert">{historyError}</p>}

      <div className="tracker-grid">
        <section className="insight-card">
          <div className="insight-badge"><Lightbulb size={16} /><span>AI INSIGHT FROM YOUR HISTORY</span></div>
          <p className="insight-desc">
            {loading ? 'Reviewing your conversation history…' : insight || 'No insight is available yet.'}
          </p>
          {insightError && <p className="chat-error" role="status">{insightError}</p>}
          <div className="insight-actions">
            <button className="discuss-btn" onClick={() => onDiscussWithBot?.()}>
              <MessageSquare size={16} /><span>Discuss with MediBot</span>
            </button>
          </div>
        </section>

        <div className="metrics-column">
          <div className="metric-card">
            <div className="metric-icon-box cyan"><Activity size={24} /></div>
            <div>
              <div className="metric-label">CHAT TURNS</div>
              <div className="metric-value-row"><span className="metric-number">{historyStats.conversation_count}</span></div>
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-icon-box dark"><MessageSquare size={24} /></div>
            <div>
              <div className="metric-label">SYMPTOM LOGS</div>
              <div className="metric-value-row"><span className="metric-number">{historyStats.symptom_count}</span></div>
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-icon-box cyan"><Bot size={24} /></div>
            <div>
              <div className="metric-label">AI RESPONSES</div>
              <div className="metric-value-row"><span className="metric-number">{historyStats.ai_response_count}</span></div>
            </div>
          </div>
        </div>
      </div>

      <section className="history-section-card">
        <div className="history-header">
          <h3>Your History</h3>
          <span className="timeline-subtext">Last activity: {lastActivity}</span>
        </div>
        {loading ? <p className="timeline-subtext">Loading your private history…</p> : history.length === 0 ? (
          <p className="timeline-subtext">Your conversations and symptom logs will appear here.</p>
        ) : (
          <div className="timeline-list">
            {[...history].reverse().map((item, index) => (
              <article className="timeline-item" key={`${item.timestamp}-${index}`}>
                <div className={`timeline-dot ${item.role === 'symptom' ? 'cyan' : ''}`} />
                <div className="timeline-title-row">
                  <div className="timeline-title">{item.role === 'assistant' ? 'MediBot response' : item.role === 'symptom' ? 'Symptom log' : 'You'}</div>
                  <time className="timeline-date">{formatDate(item.timestamp)}</time>
                </div>
                <div className="timeline-quote">{item.content}</div>
              </article>
            ))}
          </div>
        )}
      </section>

      {showLogModal && (
        <div className="modal-backdrop" onClick={() => setShowLogModal(false)}>
          <div className="modal-box" onClick={(event) => event.stopPropagation()}>
            <h2>Log a symptom</h2>
            <form onSubmit={handleAddLog}>
              <div className="form-group">
                <label className="form-label" htmlFor="symptom-description">Symptom description</label>
                <input id="symptom-description" className="form-input" value={newSymptom} onChange={(event) => setNewSymptom(event.target.value)} required />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="symptom-intensity">Self-reported intensity (1–10)</label>
                <input id="symptom-intensity" type="number" min="1" max="10" value={newIntensity} onChange={(event) => setNewIntensity(event.target.value)} required />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="symptom-notes">Notes</label>
                <textarea id="symptom-notes" className="form-input" value={newNotes} onChange={(event) => setNewNotes(event.target.value)} />
              </div>
              <div className="symptom-modal-actions">
                <button type="button" className="filter-chip" onClick={() => setShowLogModal(false)}>Cancel</button>
                <button type="submit" className="book-btn" disabled={saving}>{saving ? 'Saving…' : 'Save symptom log'}</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
