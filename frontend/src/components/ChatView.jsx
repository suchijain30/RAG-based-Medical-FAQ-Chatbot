import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, 
  Paperclip, 
  Mic, 
  MicOff, 
  CheckCircle2, 
  Info, 
  Stethoscope, 
  ExternalLink,
  Loader2
} from 'lucide-react';
import { fetchHistory, sendChatMessage, sendVoiceChat, sendImageAnalysis } from '../services/api';

export default function ChatView({ city, user, onFindSpecialist }) {
  const [messages, setMessages] = useState([]);

  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [error, setError] = useState('');
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const fileInputRef = useRef(null);
  const chatBottomRef = useRef(null);

  useEffect(() => {
    let active = true
    setHistoryLoading(true)
    fetchHistory(user)
      .then(({ messages: history }) => {
        if (active) setMessages(history.map((message, index) => ({
          ...message,
          id: `${message.timestamp}-${index}`,
          role: message.role === 'symptom' ? 'user' : message.role,
        })))
      })
      .catch((error) => {
        if (active) setError(`Unable to load your conversation history: ${error.message}`)
      })
      .finally(() => {
        if (active) setHistoryLoading(false)
      })
    return () => { active = false }
  }, [user])

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async (textToSend = input) => {
    const query = textToSend.trim();
    if (!query || loading) return;

    const userMsg = {
      id: String(Date.now()),
      role: 'user',
      content: query,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    setError('');

    try {
      const res = await sendChatMessage(query, city, user);
      
      const botMsg = {
        id: String(Date.now() + 1),
        role: 'assistant',
        badge: 'CLINICAL GUIDELINES APPLIED',
        content: res.answer,
        specialty: res.specialty,
        doctorLinks: res.doctor_links,
      };

      setMessages((prev) => [...prev, botMsg]);
    } catch (e) {
      console.error(e);
      setError(e.message || 'Unable to send your message. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleChipClick = (chip) => {
    if (chip === 'Find a specialist' && onFindSpecialist) {
      onFindSpecialist();
      return;
    }
    handleSend(chip);
  };

  // Voice recording
  const toggleRecording = async () => {
    if (recording) {
      mediaRecorderRef.current?.stop();
      setRecording(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorderRef.current = new MediaRecorder(stream);
        audioChunksRef.current = [];

        mediaRecorderRef.current.ondataavailable = (e) => {
          if (e.data.size > 0) audioChunksRef.current.push(e.data);
        };

        mediaRecorderRef.current.onstop = async () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
          setLoading(true);
          setError('');
          try {
            const res = await sendVoiceChat(audioBlob, city, user);
            const userMsg = {
              id: String(Date.now()),
              role: 'user',
              content: res.corrected_query || '🎙️ Voice note recorded',
            };
            const botMsg = {
              id: String(Date.now() + 1),
              role: 'assistant',
              badge: 'CLINICAL GUIDELINES APPLIED',
              content: res.answer,
              specialty: res.specialty,
              doctorLinks: res.doctor_links,
            };
            setMessages((prev) => [...prev, userMsg, botMsg]);
          } catch (err) {
            console.error(err);
            setError(err.message || 'Unable to process your voice message. Please try again.');
          } finally {
            setLoading(false);
          }
        };

        mediaRecorderRef.current.start();
        setRecording(true);
      } catch (err) {
        alert('Could not access microphone: ' + err.message);
      }
    }
  };

  // Image Upload
  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const userMsg = {
      id: String(Date.now()),
      role: 'user',
      content: `📷 [Uploaded image: ${file.name}]`,
      imagePreview: URL.createObjectURL(file),
    };

    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    setError('');

    try {
      const res = await sendImageAnalysis(file, 'Please analyze this medical image / report', user);
      const botMsg = {
        id: String(Date.now() + 1),
        role: 'assistant',
        badge: 'IMAGE & LAB ANALYSIS',
        content: res.analysis,
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      console.error(err);
      setError(err.message || 'Unable to analyze this file. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      {/* Thread */}
      <div className="chat-thread">
        {!historyLoading && messages.length === 0 && (
          <div className="bot-card-bubble">Hello. I’m MediBot. How can I help you today?</div>
        )}
        {messages.map((msg) => (
          <div key={msg.id} className={`message-row ${msg.role}`}>
            {msg.role === 'assistant' && (
              <div className="avatar-circle bot">
                <Stethoscope size={18} />
              </div>
            )}

            {msg.role === 'user' ? (
              <div className="user-bubble">
                {msg.imagePreview && (
                  <img 
                    src={msg.imagePreview} 
                    alt="Upload preview" 
                    style={{ maxWidth: '240px', borderRadius: '8px', marginBottom: '8px', display: 'block' }} 
                  />
                )}
                {msg.content}
              </div>
            ) : (
              <div className="bot-card-bubble">
                {msg.badge && (
                  <div className="clinical-badge">
                    <CheckCircle2 size={14} />
                    <span>{msg.badge}</span>
                  </div>
                )}

                <div className="bot-message-text" style={{ whiteSpace: 'pre-line' }}>
                  {msg.content}
                </div>

                {msg.recommendations && (
                  <div className="recommendation-callout">
                    <h5>Recommended Actions:</h5>
                    <ul>
                      {msg.recommendations.map((rec, i) => (
                        <li key={i}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {msg.warning && (
                  <div className="warning-callout">
                    <Info size={16} style={{ flexShrink: 0, marginTop: '2px' }} />
                    <span>{msg.warning}</span>
                  </div>
                )}

                {msg.chips && (
                  <div className="chips-row">
                    {msg.chips.map((chip, idx) => (
                      <button key={idx} className="chip-btn" onClick={() => handleChipClick(chip)}>
                        {chip}
                      </button>
                    ))}
                  </div>
                )}

                {msg.doctorLinks && msg.doctorLinks.practo && (
                  <div style={{ marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid #f1f5f9' }}>
                    <div style={{ fontSize: '0.85rem', fontWeight: '600', color: '#004b99', marginBottom: '0.4rem' }}>
                      🏥 Recommended Specialist near {city || 'your area'}:
                    </div>
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      <a 
                        href={msg.doctorLinks.practo} 
                        target="_blank" 
                        rel="noreferrer" 
                        className="filter-chip"
                        style={{ textDecoration: 'none' }}
                      >
                        Book on Practo <ExternalLink size={13} />
                      </a>
                    </div>
                  </div>
                )}
              </div>
            )}

            {msg.role === 'user' && (
              <div className="avatar-circle user-av" aria-label={user?.fullName || user?.email}>
                {(user?.fullName || user?.email || 'U').charAt(0).toUpperCase()}
              </div>
            )}
          </div>
        ))}

        {(loading || historyLoading) && (
          <div className="message-row assistant">
            <div className="avatar-circle bot">
              <Stethoscope size={18} />
            </div>
            <div className="bot-card-bubble" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#64748b' }}>
              <Loader2 className="animate-spin" size={18} />
              <span>{historyLoading ? 'Loading your private history…' : 'Analyzing symptoms & medical database…'}</span>
            </div>
          </div>
        )}

        <div ref={chatBottomRef} />
      </div>

      {/* Input Area */}
      <div className="chat-input-area">
        {error && <p className="chat-error" role="alert">{error}</p>}
        <div className="input-capsule">
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            accept="image/*,.pdf" 
            style={{ display: 'none' }} 
          />
          
          <button 
            className="input-icon-btn" 
            title="Attach symptom photo / lab report"
            onClick={() => fileInputRef.current?.click()}
          >
            <Paperclip size={18} />
          </button>

          <input
            type="text"
            className="input-field"
            placeholder="Describe your symptoms or ask a medical question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          />

          <div className="input-actions">
            <button 
              className={`input-icon-btn ${recording ? 'recording' : ''}`}
              title={recording ? 'Stop Recording' : 'Voice input'}
              onClick={toggleRecording}
            >
              {recording ? <MicOff size={18} /> : <Mic size={18} />}
            </button>

            <button 
              className="send-btn" 
              onClick={() => handleSend()}
              disabled={!input.trim() && !loading}
            >
              <Send size={16} />
            </button>
          </div>
        </div>

        <div className="disclaimer-text">
          MediBot provides general information and is not a substitute for professional medical advice, diagnosis, or treatment.
        </div>
      </div>
    </div>
  );
}
