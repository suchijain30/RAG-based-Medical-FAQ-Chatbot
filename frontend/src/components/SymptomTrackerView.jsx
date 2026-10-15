import React, { useState } from 'react';
import { 
  Plus, 
  Lightbulb, 
  MessageSquare, 
  Heart, 
  Moon, 
  Calendar,
  CheckCircle2
} from 'lucide-react';

export default function SymptomTrackerView({ onDiscussWithBot }) {
  const [logs, setLogs] = useState([
    {
      id: 1,
      title: 'Mild Headache & Fatigue',
      date: 'TODAY, 9:41 AM',
      type: 'Logged via text input.',
      tags: ['Headache', 'Fatigue'],
      intensity: '4/10',
      dotColor: 'cyan',
    },
    {
      id: 2,
      title: 'Voice Log Transcription',
      date: 'YESTERDAY, 8:15 PM',
      quote: '"Feeling a bit feverish this evening, throat is slightly scratchy."',
      tags: ['Fever', 'Sore Throat'],
      dotColor: 'gray',
    },
    {
      id: 3,
      title: 'Normal Baseline',
      date: 'OCT 12, 10:00 AM',
      type: 'Automated check-in. No symptoms reported.',
      tags: [],
      dotColor: 'gray',
    }
  ]);

  const [showLogModal, setShowLogModal] = useState(false);
  const [newSymptom, setNewSymptom] = useState('');
  const [newIntensity, setNewIntensity] = useState('5');
  const [newNotes, setNewNotes] = useState('');

  const handleAddLog = (e) => {
    e.preventDefault();
    if (!newSymptom.trim()) return;

    const newEntry = {
      id: Date.now(),
      title: newSymptom,
      date: 'JUST NOW',
      type: newNotes || 'Logged via web entry.',
      tags: [newSymptom],
      intensity: `${newIntensity}/10`,
      dotColor: 'cyan',
    };

    setLogs([newEntry, ...logs]);
    setNewSymptom('');
    setNewNotes('');
    setShowLogModal(false);
  };

  return (
    <div className="content-body">
      <div className="symptom-top-row">
        <div>
          <h1 className="page-title">Symptom Tracker</h1>
          <p className="page-subtitle">
            Monitor your health patterns, review past logs, and receive AI-driven clinical insights based on your recent activity.
          </p>
        </div>
        <button className="log-symptom-btn" onClick={() => setShowLogModal(true)}>
          <Plus size={18} />
          <span>Log Symptom</span>
        </button>
      </div>

      {/* Grid: AI Insight + Metrics */}
      <div className="tracker-grid">
        {/* AI Insight Card */}
        <div className="insight-card">
          <div className="insight-badge">
            <Lightbulb size={16} />
            <span>AI INSIGHT</span>
          </div>
          <h3 className="insight-title">Seasonal Pattern Detected</h3>
          <p className="insight-desc">
            Your recent logs of <strong>dry cough</strong> and <strong>fatigue</strong> align closely with seasonal allergies, similar to patterns noted last Spring. We recommend monitoring pollen counts in your area over the next two weeks.
          </p>
          <div className="insight-actions">
            <button className="discuss-btn" onClick={() => onDiscussWithBot && onDiscussWithBot('Can you explain seasonal allergy patterns and what treatments help?')}>
              <MessageSquare size={16} />
              <span>Discuss with MediBot</span>
            </button>
            <span className="secondary-link" onClick={() => alert('Viewing historical 12-month allergy pattern correlations.')}>
              View Past Patterns
            </span>
          </div>
        </div>

        {/* Right Metrics Column */}
        <div className="metrics-column">
          <div className="metric-card">
            <div className="metric-icon-box cyan">
              <Heart size={24} />
            </div>
            <div>
              <div className="metric-label">AVG RESTING HR</div>
              <div className="metric-value-row">
                <span className="metric-number">68</span>
                <span className="metric-unit">bpm</span>
              </div>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-icon-box dark">
              <Moon size={24} />
            </div>
            <div>
              <div className="metric-label">SLEEP QUALITY</div>
              <div className="metric-value-row">
                <span className="metric-number">7.2</span>
                <span className="metric-unit">hrs/night</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent History Timeline */}
      <div className="history-section-card">
        <div className="history-header">
          <h3>Recent History</h3>
          <span className="secondary-link" onClick={() => alert('Displaying complete log repository.')}>
            View All Logs
          </span>
        </div>

        <div className="timeline-list">
          {logs.map((item) => (
            <div key={item.id} className="timeline-item">
              <div className={`timeline-dot ${item.dotColor === 'cyan' ? 'cyan' : ''}`} />
              <div className="timeline-title-row">
                <div className="timeline-title">{item.title}</div>
                <div className="timeline-date">{item.date}</div>
              </div>

              {item.type && <div className="timeline-subtext">{item.type}</div>}
              {item.quote && <div className="timeline-quote">{item.quote}</div>}

              <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
                {item.tags.map((tag, idx) => (
                  <span key={idx} className="tag-pill">{tag}</span>
                ))}
                {item.intensity && (
                  <span className="intensity-badge">Intensity: {item.intensity}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Log Symptom Modal */}
      {showLogModal && (
        <div className="modal-backdrop" onClick={() => setShowLogModal(false)}>
          <div className="modal-box" onClick={(e) => e.stopPropagation()}>
            <h2 style={{ fontSize: '1.3rem', fontWeight: '700', marginBottom: '1rem' }}>
              Log New Symptom
            </h2>
            <form onSubmit={handleAddLog}>
              <div className="form-group">
                <label className="form-label">Symptom Description</label>
                <input 
                  type="text" 
                  className="form-input" 
                  style={{ paddingLeft: '1rem' }}
                  placeholder="e.g. Throbbing migraine, Sore throat, Cough"
                  value={newSymptom}
                  onChange={(e) => setNewSymptom(e.target.value)}
                  required
                />
              </div>

              <div className="form-group">
                <label className="form-label">Intensity: {newIntensity} / 10</label>
                <input 
                  type="range" 
                  min="1" 
                  max="10" 
                  value={newIntensity} 
                  onChange={(e) => setNewIntensity(e.target.value)} 
                  style={{ width: '100%' }}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Notes & Circumstances</label>
                <textarea 
                  className="form-input"
                  style={{ paddingLeft: '1rem', minHeight: '80px' }}
                  placeholder="e.g. Started after working on laptop for 6 hours"
                  value={newNotes}
                  onChange={(e) => setNewNotes(e.target.value)}
                />
              </div>

              <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end', marginTop: '1.5rem' }}>
                <button type="button" className="filter-chip" onClick={() => setShowLogModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="book-btn">
                  Save Symptom Log
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
