import { useEffect, useState } from 'react'
import { ExternalLink, MapPin, Phone, Search, Star } from 'lucide-react'
import { searchDoctors } from '../services/api'

const specialties = [
  'Cardiology', 'Dermatology', 'Endocrinology', 'Family Medicine',
  'Neurology', 'Obstetrics and Gynecology', 'Oncology', 'Ophthalmology',
  'Orthopedics', 'Pediatrics', 'Psychiatry', 'Dentistry',
]

export default function DoctorSearchView({ city, user }) {
  const [specialty, setSpecialty] = useState('')
  const [doctors, setDoctors] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!city?.trim()) {
      setDoctors([])
      setError('Add a city to your profile to search nearby healthcare providers.')
      return undefined
    }

    const controller = new AbortController()
    setLoading(true)
    setError('')
    searchDoctors(city, specialty, user, controller.signal)
      .then((results) => setDoctors(results.doctors))
      .catch((searchError) => {
        if (!controller.signal.aborted) {
          setError(searchError.message || 'Unable to search healthcare providers.')
          setDoctors([])
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })

    return () => controller.abort()
  }, [city, specialty, user])

  return (
    <main className="content-body">
      <h1 className="page-title">Find a Healthcare Provider</h1>
      <p className="page-subtitle">Live places directory results for {city || 'your location'}.</p>

      <div className="filter-bar">
        <label className="doctor-filter-label" htmlFor="doctor-specialty"><Search size={16} /> Specialty</label>
        <select id="doctor-specialty" className="doctor-filter-select" value={specialty} onChange={(event) => setSpecialty(event.target.value)}>
          <option value="">All healthcare providers</option>
          {specialties.map((item) => <option key={item} value={item}>{item}</option>)}
        </select>
        {specialty && <button className="filter-chip" onClick={() => setSpecialty('')}>Clear specialty</button>}
      </div>

      {error && <p className="chat-error" role="alert">{error}</p>}
      {loading && <p className="timeline-subtext" aria-live="polite">Searching current listings…</p>}
      {!loading && !error && doctors.length === 0 && <p className="timeline-subtext">No providers were returned for this search. Try another city or specialty.</p>}

      <section className="doctor-results" aria-live="polite">
        {doctors.map((doctor) => (
          <article className="doctor-card-item" key={doctor.id}>
            <div className="doctor-card-top">
              <div className="doctor-result-icon"><MapPin size={22} /></div>
              <div className="doctor-info-text">
                <div className="doctor-name-row">
                  <h2 className="doctor-name">{doctor.name}</h2>
                  {doctor.rating != null && (
                    <div className="doctor-rating"><Star size={13} fill="currentColor" /><span>{doctor.rating}{doctor.reviewCount ? ` (${doctor.reviewCount})` : ''}</span></div>
                  )}
                </div>
                <div className="doctor-specialty-line">{doctor.specialty}</div>
                {doctor.address && <p className="doctor-address">{doctor.address}</p>}
                {doctor.phone && <a className="doctor-phone" href={`tel:${doctor.phone}`}><Phone size={14} />{doctor.phone}</a>}
              </div>
            </div>
            <div className="doctor-card-bottom">
              <span className="timeline-subtext">Information provided by Google Places</span>
              <div className="doctor-result-links">
                {doctor.website && <a className="filter-chip" href={doctor.website} target="_blank" rel="noreferrer">Website <ExternalLink size={13} /></a>}
                {doctor.mapsUrl && <a className="book-btn" href={doctor.mapsUrl} target="_blank" rel="noreferrer">View directions <ExternalLink size={13} /></a>}
              </div>
            </div>
          </article>
        ))}
      </section>
      <p className="provider-attribution">Listings, ratings, and links are returned by Google Places. Verify services, availability, and credentials with the provider.</p>
    </main>
  )
}
