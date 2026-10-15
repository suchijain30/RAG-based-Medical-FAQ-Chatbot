import React from 'react';
import { 
  MessageSquarePlus, 
  Activity, 
  MapPin, 
  BookOpen, 
  Sparkles, 
  User, 
  LogOut,
  Stethoscope
} from 'lucide-react';

export default function Sidebar({ 
  currentTab, 
  setCurrentTab, 
  onOpenProfile, 
  onLogout,
  recentChats = [],
  onSelectRecentChat
}) {
  const navItems = [
    { id: 'chat', label: 'New Chat', icon: MessageSquarePlus },
    { id: 'symptoms', label: 'Symptom Tracker', icon: Activity },
    { id: 'doctors', label: 'Doctor Search', icon: MapPin },
    { id: 'guides', label: 'Patient Guides', icon: BookOpen },
  ];

  return (
    <aside className="sidebar">
      {/* Brand */}
      <div className="brand-header">
        <div className="brand-icon">
          <Stethoscope size={22} />
        </div>
        <div>
          <div className="brand-title">MediBot AI</div>
          <div className="brand-subtitle">Clinical Assistant</div>
        </div>
      </div>

      {/* Upgrade to Pro Button */}
      <button className="upgrade-btn" onClick={() => alert('MediBot Pro: Advanced multimodal diagnostics, priority doctor bookings, and full medical records sync.')}>
        <Sparkles size={16} />
        <span>Upgrade to Pro</span>
      </button>

      {/* Navigation */}
      <nav className="nav-menu">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentTab === item.id;
          return (
            <button
              key={item.id}
              className={`nav-item ${isActive ? 'active' : ''}`}
              onClick={() => setCurrentTab(item.id)}
            >
              <Icon size={19} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Recent Conversations (Visible in Chat) */}
      <div className="recent-section">
        <div className="recent-heading">Recent Conversations</div>
        <button className="recent-item" onClick={() => onSelectRecentChat && onSelectRecentChat('Persistent headache guidelines')}>
          Persistent headache guidelines
        </button>
        <button className="recent-item" onClick={() => onSelectRecentChat && onSelectRecentChat('Vitamin D deficiency symptoms')}>
          Vitamin D deficiency symptoms
        </button>
        <button className="recent-item" onClick={() => onSelectRecentChat && onSelectRecentChat('Post-op care instructions')}>
          Post-op care instructions
        </button>
      </div>

      {/* Bottom Actions */}
      <div className="sidebar-bottom">
        <button className="nav-item" onClick={onOpenProfile}>
          <User size={18} />
          <span>Profile</span>
        </button>
        <button className="nav-item" onClick={onLogout}>
          <LogOut size={18} />
          <span>Log Out</span>
        </button>
      </div>
    </aside>
  );
}
