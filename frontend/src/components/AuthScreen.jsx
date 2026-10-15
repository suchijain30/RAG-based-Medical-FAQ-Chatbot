import { useState } from 'react'
import { createUserWithEmailAndPassword, signInWithEmailAndPassword, updateProfile } from 'firebase/auth'
import { HeartPulse, Loader2, LockKeyhole, Mail, UserRound } from 'lucide-react'
import { auth } from '../services/firebase'

function getAuthError(error) {
  const messages = {
    'auth/email-already-in-use': 'An account already exists for this email.',
    'auth/invalid-credential': 'Email or password is incorrect.',
    'auth/invalid-email': 'Enter a valid email address.',
    'auth/weak-password': 'Use a password with at least 6 characters.',
    'auth/too-many-requests': 'Too many attempts. Try again later.',
    'auth/network-request-failed': 'Unable to reach Firebase. Check your connection and try again.',
    'auth/operation-not-allowed': 'Email/password sign-in is not enabled for this Firebase project.',
    'auth/user-disabled': 'This account has been disabled. Contact support for help.',
  }
  return messages[error.code] || 'Authentication failed. Please try again.'
}

export default function AuthScreen({ configError, initialError }) {
  const [isSignup, setIsSignup] = useState(false)
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(configError || initialError || '')
  const [busy, setBusy] = useState(false)

  const handleSubmit = async (event) => {
    event.preventDefault()
    setError('')
    setBusy(true)

    try {
      if (isSignup) {
        const credential = await createUserWithEmailAndPassword(auth, email, password)
        if (name.trim()) await updateProfile(credential.user, { displayName: name.trim() })
      } else {
        await signInWithEmailAndPassword(auth, email, password)
      }
    } catch (authError) {
      setError(getAuthError(authError))
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-card" aria-labelledby="auth-title">
        <div className="auth-brand-icon"><HeartPulse size={26} /></div>
        <h1 id="auth-title">MediBot AI</h1>
        <p className="auth-subtitle">
          {isSignup ? 'Create an account to get started.' : 'Sign in to continue to your clinical assistant.'}
        </p>

        <form className="auth-form" onSubmit={handleSubmit}>
          {isSignup && (
            <label className="auth-field">
              <span>Name</span>
              <span className="auth-input-wrap"><UserRound size={18} /><input autoComplete="name" value={name} onChange={(event) => setName(event.target.value)} placeholder="Your name" /></span>
            </label>
          )}
          <label className="auth-field">
            <span>Email</span>
            <span className="auth-input-wrap"><Mail size={18} /><input type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} required placeholder="you@example.com" /></span>
          </label>
          <label className="auth-field">
            <span>Password</span>
            <span className="auth-input-wrap"><LockKeyhole size={18} /><input type="password" autoComplete={isSignup ? 'new-password' : 'current-password'} value={password} onChange={(event) => setPassword(event.target.value)} minLength={6} required placeholder="At least 6 characters" /></span>
          </label>
          {error && <p className="auth-error" role="alert">{error}</p>}
          <button className="auth-submit" type="submit" disabled={busy || Boolean(configError)}>
            {busy && <Loader2 size={18} className="auth-spinner" />}
            {busy ? 'Please wait…' : isSignup ? 'Create account' : 'Sign in'}
          </button>
        </form>

        <p className="auth-toggle">
          {isSignup ? 'Already have an account?' : 'New to MediBot?'}{' '}
          <button type="button" onClick={() => { setIsSignup(!isSignup); setError('') }} disabled={Boolean(configError)}>
            {isSignup ? 'Sign in' : 'Create account'}
          </button>
        </p>
        <p className="auth-disclaimer">For informational purposes only. Not a substitute for professional medical care.</p>
      </section>
    </main>
  )
}
