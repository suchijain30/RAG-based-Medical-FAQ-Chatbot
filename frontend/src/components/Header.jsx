import React from 'react';
import { MapPin, Bell, Settings } from 'lucide-react';

export default function Header({ city, onCityChange, onOpenProfile, user }) {
  return (
    <header className="top-header">
      {/* Location Pill */}
      <button 
        className="location-pill" 
        onClick={() => {
          const newCity = prompt('Enter your city for nearby doctor recommendations:', city);
          if (newCity && onCityChange) onCityChange(newCity);
        }}
        title="Click to change location"
      >
        <MapPin size={15} color="#004b99" />
        <span>{city || 'New York, NY'}</span>
      </button>

      {/* Actions */}
      <div className="header-actions">
        <button className="icon-btn" title="Notifications" onClick={() => alert('No new medical alerts.')}>
          <Bell size={19} />
          <span className="badge-dot"></span>
        </button>
        <button className="icon-btn" title="Settings" onClick={onOpenProfile}>
          <Settings size={19} />
        </button>
        <button
          type="button"
          className="user-avatar-btn"
          onClick={onOpenProfile}
          title={user?.fullName || user?.email || 'Profile'}
          aria-label="Open profile"
        >
          {(user?.fullName || user?.email || 'U').charAt(0).toUpperCase()}
        </button>
      </div>
    </header>
  );
}
