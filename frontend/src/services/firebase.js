import { getApp, getApps, initializeApp } from 'firebase/app'
import { getAuth } from 'firebase/auth'

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
  measurementId: import.meta.env.VITE_FIREBASE_MEASUREMENT_ID,
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

export async function getDatabase() {
  if (!auth) return null
  const { getFirestore } = await import('firebase/firestore')
  return getFirestore(auth.app)
}
