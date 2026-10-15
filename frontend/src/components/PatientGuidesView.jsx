import React, { useState } from 'react';
import { 
  Download, 
  ArrowRight, 
  Utensils, 
  Activity, 
  Moon, 
  Plus, 
  BookOpen,
  Wrench,
  Clock
} from 'lucide-react';

export default function PatientGuidesView({ onRequestNewGuide }) {
  const [selectedGuide, setSelectedGuide] = useState(null);

  const guides = [
    {
      id: 1,
      featured: true,
      title: 'Post-Surgery Recovery Guide',
      tags: ['Post-Op', 'New'],
      excerpt: 'A comprehensive 6-week recovery plan following your recent procedure, including mobility exercises, pain management schedules, and dietary guidance.',
      generatedDate: 'Oct 24, 2023',
      content: `## Post-Surgery Recovery Protocol\n\n### 1. Wound Care & Hygiene\n• Keep surgical incisions dry for the first 48 hours.\n• Inspect daily for signs of increased redness or warmth.\n\n### 2. Pain & Medication Management\n• Take prescribed analgesics on a scheduled basis for the first 3 days.\n• Transition to acetaminophen as comfort improves.\n\n### 3. Progressive Mobility\n• Week 1-2: Light indoor walking for 10 minutes every 2 hours.\n• Week 3-4: Gentle stretches avoiding abdominal strain.\n• Week 5-6: Resume normal daily activities under clinical guidance.`
    },
    {
      id: 2,
      category: 'NUTRITION',
      icon: 'utensils',
      iconClass: 'cyan',
      title: 'Dietary Plan: Iron Deficiency',
      excerpt: 'Personalized meal strategies to boost iron absorption, featuring iron-rich recipes and vitamin C pairings.',
      content: `## Iron Deficiency Nutritional Protocol\n\n### High-Iron Foods\n• Spinach, lentils, fortified oats, and lean poultry.\n• Pair plant iron with Vitamin C (oranges, bell peppers) to boost absorption by up to 300%.\n\n### Items to Avoid Around Meals\n• Coffee and black tea within 1 hour of meals (tannins inhibit absorption).`
    },
    {
      id: 3,
      category: 'PHYSICAL THERAPY',
      icon: 'activity',
      iconClass: 'blue',
      title: 'Lower Back Mobility Protocol',
      excerpt: 'Gentle stretching routines and ergonomic advice to alleviate chronic lower back tension during desk work.',
      content: `## Lower Back Mobility Guide\n\n• Cat-cow stretches (10 reps daily).\n• Ergonomic desk setup with lumbar support.\n• Stand and stretch every 45 minutes.`
    },
    {
      id: 4,
      category: 'WELLNESS',
      icon: 'moon',
      iconClass: 'dark',
      title: 'Sleep Hygiene Blueprint',
      excerpt: 'Techniques for establishing a restorative circadian rhythm, managing pre-sleep anxiety, and optimizing bedroom environment.',
      content: `## Sleep Hygiene Protocol\n\n• Consistent wake-up time 7 days a week.\n• Avoid blue light 60 minutes before bed.\n• Keep bedroom temperature around 65°F (18°C).`
    }
  ];

  return (
    <div className="content-body">
      <h1 className="page-title">Patient Guides</h1>
      <p className="page-subtitle">
        Your personalized medical library. Structured summaries, recovery plans, and dietary advice generated from your consultations.
      </p>

      <div className="guides-grid">
        {/* Featured Card */}
        {guides.filter(g => g.featured).map((g) => (
          <div key={g.id} className="guide-card featured-card">
            <div className="guide-featured-banner">
              <div style={{ display: 'flex', gap: '0.4rem' }}>
                <span className="tag-pill" style={{ background: 'rgba(255,255,255,0.2)', color: 'white' }}>
                  🔧 Post-Op
                </span>
                <span className="tag-pill" style={{ background: '#64eedf', color: '#083344', fontWeight: '700' }}>
                  ⏱ New
                </span>
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3 className="guide-title">{g.title}</h3>
                <button className="icon-btn" title="Download PDF" onClick={() => alert('Downloading PDF guide...')}>
                  <Download size={18} />
                </button>
              </div>
              <p className="guide-excerpt">{g.excerpt}</p>
            </div>

            <div className="guide-footer">
              <span style={{ fontSize: '0.8rem', color: '#64748b' }}>Generated: {g.generatedDate}</span>
              <button 
                className="secondary-link" 
                style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', fontWeight: '600', color: '#004b99' }}
                onClick={() => setSelectedGuide(g)}
              >
                <span>Read Guide</span>
                <ArrowRight size={15} />
              </button>
            </div>
          </div>
        ))}

        {/* Regular Guides */}
        {guides.filter(g => !g.featured).map((g) => (
          <div key={g.id} className="guide-card">
            <div>
              <div className={`guide-icon-circle ${g.iconClass}`}>
                {g.icon === 'utensils' && <Utensils size={22} />}
                {g.icon === 'activity' && <Activity size={22} />}
                {g.icon === 'moon' && <Moon size={22} />}
              </div>
              <h3 className="guide-title">{g.title}</h3>
              <p className="guide-excerpt">{g.excerpt}</p>
            </div>

            <div className="guide-footer">
              <span className="tag-pill">{g.category}</span>
              <button className="icon-btn" title="Download Guide" onClick={() => setSelectedGuide(g)}>
                <Download size={16} />
              </button>
            </div>
          </div>
        ))}

        {/* Request New Guide Card */}
        <div 
          className="guide-card guide-request-card" 
          onClick={() => {
            const condition = prompt('What medical topic or condition would you like a guide for? (e.g. Type 2 Diabetes Diet, Hypertension Care)');
            if (condition && onRequestNewGuide) {
              onRequestNewGuide(condition);
            }
          }}
        >
          <div className="guide-request-icon">
            <Plus size={22} />
          </div>
          <h3 style={{ fontSize: '1.1rem', fontWeight: '700', marginBottom: '0.35rem', color: '#0f172a' }}>
            Request New Guide
          </h3>
          <p style={{ fontSize: '0.85rem', color: '#64748b', lineHeight: 1.4 }}>
            Ask MediBot to summarize a recent consultation into a structured guide.
          </p>
        </div>
      </div>

      {/* Full Guide Viewer Modal */}
      {selectedGuide && (
        <div className="modal-backdrop" onClick={() => setSelectedGuide(null)}>
          <div className="modal-box" style={{ maxWidth: '650px' }} onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h2 style={{ fontSize: '1.35rem', fontWeight: '700' }}>{selectedGuide.title}</h2>
              <button className="icon-btn" onClick={() => setSelectedGuide(null)}>✕</button>
            </div>
            
            <div style={{ background: '#f8fafc', padding: '1.25rem', borderRadius: '8px', lineHeight: '1.6', fontSize: '0.92rem', whiteSpace: 'pre-line', marginBottom: '1.25rem' }}>
              {selectedGuide.content}
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
              <button className="filter-chip" onClick={() => setSelectedGuide(null)}>Close</button>
              <button className="book-btn" onClick={() => alert('PDF document generated and downloaded.')}>Download PDF</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
