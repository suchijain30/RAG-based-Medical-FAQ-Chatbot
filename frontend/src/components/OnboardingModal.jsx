import React, { useState } from 'react';
import { User, Calendar, MapPin, ArrowRight, ShieldCheck } from 'lucide-react';

export default function OnboardingModal({ user, onComplete, onClose }) {
  const [fullName, setFullName] = useState(user?.fullName || '');
  const [dob, setDob] = useState(user?.dob || '');
  const [gender, setGender] = useState(user?.gender || '');
  const [location, setLocation] = useState(user?.city || '');
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSaving(true);
    setError('');
    const updated = {
      ...user,
      fullName: fullName.trim(),
      dob,
      gender,
      city: location.trim(),
    };
    try {
      await onComplete?.(updated);
    } catch (saveError) {
      setError(saveError.message || 'Unable to save your profile.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="onboarding-wrap" onClick={(e) => e.stopPropagation()}>
        {/* Step indicator */}
        <div className="onboarding-progress">
          <span>PROFILE</span>
          <span>PERSONAL DETAILS</span>
        </div>

        <div className="progress-bar-line">
          <div className="progress-fill" />
        </div>

        {/* Avatar icon */}
        <div className="onboarding-icon-box">
          <User size={28} />
        </div>

        <h2 style={{ fontSize: '1.4rem', fontWeight: '700', marginBottom: '0.5rem', color: '#0f172a' }}>
          {user?.fullName ? 'Update your profile.' : "Let's complete your profile."}
        </h2>
        <p style={{ fontSize: '0.9rem', color: '#64748b', marginBottom: '1.75rem', lineHeight: '1.4' }}>
          This information helps us personalize your health insights.
        </p>

        <form onSubmit={handleSubmit}>
          {/* Full Name */}
          <div className="form-group">
            <label className="form-label">Full Name</label>
            <div className="input-with-icon">
              <User className="input-icon" size={18} />
              <input 
                type="text" 
                className="form-input" 
                placeholder="Your name"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                required
              />
            </div>
          </div>

          {/* Date of Birth */}
          <div className="form-group">
            <label className="form-label">Date of Birth</label>
            <div className="input-with-icon">
              <Calendar className="input-icon" size={18} />
              <input 
                type="date" 
                className="form-input" 
                value={dob}
                onChange={(e) => setDob(e.target.value)}
                max={new Date().toISOString().slice(0, 10)}
                required
              />
            </div>
          </div>

          {/* Gender */}
          <div className="form-group">
            <label className="form-label">Gender</label>
            <select 
              className="form-input" 
              style={{ paddingLeft: '1rem' }}
              value={gender}
              onChange={(e) => setGender(e.target.value)}
              required
            >
              <option value="">Choose an option</option>
              <option value="Female">Female</option>
              <option value="Male">Male</option>
              <option value="Transgender">Transgender</option>
            </select>
          </div>

          {/* Location */}
          <div className="form-group">
            <label className="form-label">Location</label>
            <div className="input-with-icon">
              <MapPin className="input-icon" size={18} />
              <input 
                type="text" 
                className="form-input" 
                placeholder="City, Country"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                required
              />
            </div>
          </div>

          {/* Continue button */}
          <button 
            type="submit" 
            className="upgrade-btn" 
            style={{ width: '100%', padding: '0.85rem', fontSize: '1rem', marginTop: '1.5rem', borderRadius: '8px' }}
          >
            <span>{saving ? 'Saving…' : 'Save profile'}</span>
            <ArrowRight size={18} />
          </button>
        </form>
        {error && <p className="auth-error" role="alert">{error}</p>}

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem', fontSize: '0.75rem', color: '#94a3b8', marginTop: '1.25rem' }}>
          <ShieldCheck size={14} />
          <span>Your profile is stored with your account and can be updated any time.</span>
        </div>
      </div>
    </div>
  );
}
