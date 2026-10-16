import React from 'react';
import { MapPin, Moon, Sun } from 'lucide-react';

export default function Header({ city, onCityChange, onOpenProfile, user, theme, onToggleTheme }) {
  return (
    <header className="top-header">
      {/* Location Pill */}
      <button 
        className="location-pill" 
        onClick={() => {
          const newCity = prompt('Enter your city for nearby healthcare provider listings:', city);
          if (newCity && onCityChange) onCityChange(newCity);
        }}
        title="Click to change location"
      >
        <MapPin size={15} color="#004b99" />
        <span>{city || 'Set your location'}</span>
      </button>

      {/* Actions */}
      <div className="header-actions">
        <button className="icon-btn" title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`} aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`} onClick={onToggleTheme}>
          {theme === 'light' ? <Moon size={19} /> : <Sun size={19} />}
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
