import React, { useState } from 'react';
import { 
  Lightbulb, 
  SlidersHorizontal, 
  ChevronDown, 
  Star, 
  MapPin, 
  Phone,
  Plus,
  Minus
} from 'lucide-react';

export default function DoctorSearchView({ city }) {
  const [selectedSpecialty, setSelectedSpecialty] = useState('Dermatology');
  const [selectedDistance, setSelectedDistance] = useState('Under 5 miles');
  const [selectedInsurance, setSelectedInsurance] = useState('Aetna');
  const [bookingDoctor, setBookingDoctor] = useState(null);

  const doctors = [
    {
      id: 1,
      name: 'Dr. Sarah Jenkins',
      rating: 4.9,
      specialty: 'Dermatologist',
      experience: '15 yrs exp',
      distance: '1.2 miles',
      insurance: 'Accepts Aetna',
      telehealth: true,
      nextAvailable: 'Today, 2:30 PM',
      avatar: 'https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=120&auto=format&fit=crop&q=80',
    },
    {
      id: 2,
      name: 'Dr. Michael Chen',
      rating: 4.7,
      specialty: 'Dermatologist',
      experience: '8 yrs exp',
      distance: '3.5 miles',
      insurance: 'Accepts Aetna',
      telehealth: false,
      nextAvailable: 'Tomorrow, 9:00 AM',
      avatar: 'https://images.unsplash.com/photo-1622253692010-333f2da6031d?w=120&auto=format&fit=crop&q=80',
    },
    {
      id: 3,
      name: 'Dr. Evelyn Reed',
      rating: 4.8,
      specialty: 'Cardiologist',
      experience: '12 yrs exp',
      distance: '4.1 miles',
      insurance: 'Accepts Aetna',
      telehealth: true,
      nextAvailable: 'Thursday, 11:15 AM',
      avatar: 'https://images.unsplash.com/photo-1594824813687-99520b22a00c?w=120&auto=format&fit=crop&q=80',
    }
  ];

  return (
    <div className="content-body">
      <h1 className="page-title">Find a Specialist</h1>

      {/* AI Recommendation Banner */}
      <div className="ai-recommendation-box">
        <div className="icon-wrap">
          <Lightbulb size={18} />
        </div>
        <div>
          <h4>AI Recommendation</h4>
          <p>
            Based on your recent symptom logs concerning skin irritation, we recommend consulting a Dermatologist. Here are specialists nearby in {city || 'New York'}.
          </p>
        </div>
      </div>

      {/* Filters Bar */}
      <div className="filter-bar">
        <button className="filter-chip active">
          <SlidersHorizontal size={14} />
          <span>Filters</span>
        </button>

        <button className="filter-chip" onClick={() => {
          const spec = prompt('Enter specialty (Dermatology, Cardiology, Neurology, ENT, Orthopedic):', selectedSpecialty);
          if (spec) setSelectedSpecialty(spec);
        }}>
          <span>Specialty: {selectedSpecialty}</span>
          <ChevronDown size={14} />
        </button>

        <button className="filter-chip" onClick={() => {
          const dist = prompt('Select max distance (Under 5 miles, Under 10 miles, Any):', selectedDistance);
          if (dist) setSelectedDistance(dist);
        }}>
          <span>Distance: {selectedDistance}</span>
          <ChevronDown size={14} />
        </button>

        <button className="filter-chip" onClick={() => {
          const ins = prompt('Select insurance (Aetna, BlueCross, Cigna, Medicare):', selectedInsurance);
          if (ins) setSelectedInsurance(ins);
        }}>
          <span>Insurance: {selectedInsurance}</span>
          <ChevronDown size={14} />
        </button>
      </div>

      {/* Doctor Search Layout */}
      <div className="doctor-layout">
        {/* Map / Phone simulation */}
        <div className="map-container">
          <div className="map-phone-frame">
            <div style={{ padding: '0.75rem 1rem', background: 'white', display: 'flex', alignItems: 'center', gap: '0.5rem', borderBottom: '1px solid #e2e8f0', fontSize: '0.82rem', fontWeight: '600' }}>
              <MapPin size={16} color="#004b99" />
              <span>Specialists in {city || 'New York'}</span>
            </div>

            <div className="map-graphic">
              {/* Pins */}
              <div className="map-pin" style={{ top: '80px', left: '140px' }}>🏥</div>
              <div className="map-pin" style={{ top: '160px', left: '80px' }}>🩺</div>
              <div className="map-pin" style={{ top: '220px', left: '200px' }}>💊</div>
              <div className="map-pin" style={{ top: '300px', left: '120px' }}>⭐</div>

              {/* Popup */}
              <div className="map-info-popup">
                <img src="https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=80&auto=format&fit=crop&q=80" alt="Clinic" />
                <div>
                  <strong>UCSF Health Clinic</strong>
                  <div style={{ color: '#64748b' }}>Mon-Fri 8am-5pm • 1.2 mi</div>
                </div>
              </div>

              {/* Bottom Card */}
              <div className="map-bottom-card">
                <div style={{ fontSize: '0.75rem', fontWeight: '700', color: '#64748b', marginBottom: '0.25rem' }}>
                  Nearby Specialists ({doctors.length})
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <img 
                    src={doctors[0].avatar} 
                    alt="Doc" 
                    style={{ width: '32px', height: '32px', borderRadius: '50%' }} 
                  />
                  <div>
                    <div style={{ fontSize: '0.82rem', fontWeight: '700' }}>{doctors[0].name}</div>
                    <div style={{ fontSize: '0.72rem', color: '#64748b' }}>{doctors[0].specialty}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="map-zoom-controls">
            <button title="Zoom in"><Plus size={16} /></button>
            <button title="Zoom out"><Minus size={16} /></button>
          </div>
        </div>

        {/* Doctor Cards */}
        <div className="doctor-cards-list">
          {doctors.map((doc) => (
            <div key={doc.id} className="doctor-card-item">
              <div className="doctor-card-top">
                <img src={doc.avatar} alt={doc.name} className="doctor-avatar" />
                <div className="doctor-info-text">
                  <div className="doctor-name-row">
                    <h3 className="doctor-name">{doc.name}</h3>
                    <div className="doctor-rating">
                      <Star size={12} fill="#1e40af" />
                      <span>{doc.rating}</span>
                    </div>
                  </div>
                  <div className="doctor-specialty-line">
                    {doc.specialty} • {doc.experience}
                  </div>
                  <div className="doctor-tags">
                    <span className="tag-pill">{doc.distance}</span>
                    <span className="tag-pill">{doc.insurance}</span>
                    {doc.telehealth && <span className="tag-pill">Telehealth</span>}
                  </div>
                </div>
              </div>

              <div className="doctor-card-bottom">
                <div className="availability-text">
                  Next available: <strong>{doc.nextAvailable}</strong>
                </div>
                <button className="book-btn" onClick={() => setBookingDoctor(doc)}>
                  Book
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Booking Confirmation Modal */}
      {bookingDoctor && (
        <div className="modal-backdrop" onClick={() => setBookingDoctor(null)}>
          <div className="modal-box" onClick={(e) => e.stopPropagation()}>
            <h2 style={{ fontSize: '1.3rem', fontWeight: '700', marginBottom: '0.5rem' }}>
              Confirm Appointment
            </h2>
            <p style={{ fontSize: '0.9rem', color: '#64748b', marginBottom: '1.25rem' }}>
              Booking consultation with <strong>{bookingDoctor.name}</strong> ({bookingDoctor.specialty}).
            </p>

            <div style={{ background: '#f8fafc', padding: '1rem', borderRadius: '8px', marginBottom: '1.25rem' }}>
              <div>📅 <strong>Time:</strong> {bookingDoctor.nextAvailable}</div>
              <div style={{ marginTop: '0.35rem' }}>📍 <strong>Location:</strong> {city || 'New York Clinic'} & Telehealth</div>
              <div style={{ marginTop: '0.35rem' }}>🛡️ <strong>Insurance:</strong> {bookingDoctor.insurance} Verified</div>
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
              <button className="filter-chip" onClick={() => setBookingDoctor(null)}>
                Cancel
              </button>
              <button 
                className="book-btn" 
                onClick={() => {
                  alert(`Appointment successfully confirmed with ${bookingDoctor.name} for ${bookingDoctor.nextAvailable}!`);
                  setBookingDoctor(null);
                }}
              >
                Confirm Booking
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
