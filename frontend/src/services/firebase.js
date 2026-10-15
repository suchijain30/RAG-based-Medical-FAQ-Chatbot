import { getApp, getApps, initializeApp } from 'firebase/app'
import { getAuth } from 'firebase/auth'

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
}

export const firebaseConfigError = [
  ['FIREBASE_API_KEY', firebaseConfig.apiKey],
  ['FIREBASE_AUTH_DOMAIN', firebaseConfig.authDomain],
  ['FIREBASE_PROJECT_ID', firebaseConfig.projectId],
]
  .filter(([, value]) => !value)
  .map(([name]) => name)

export const auth = firebaseConfigError.length
  ? null
  : getAuth(getApps().length ? getApp() : initializeApp(firebaseConfig))
