import { getDatabase } from './firebase'

export async function fetchProfile(user) {
  if (!user?.uid) throw new Error('Firebase profile storage is unavailable.')
  const db = await getDatabase()
  if (!db) throw new Error('Firebase profile storage is unavailable.')
  const { doc, getDoc } = await import('firebase/firestore')
  const snapshot = await getDoc(doc(db, 'users', user.uid))
  return snapshot.exists() ? snapshot.data() : {}
}

export async function saveProfile(profile, user) {
  if (!user?.uid) throw new Error('Firebase profile storage is unavailable.')
  const db = await getDatabase()
  if (!db) throw new Error('Firebase profile storage is unavailable.')
  const { doc, setDoc } = await import('firebase/firestore')
  const data = {
    fullName: profile.fullName.trim(),
    city: profile.city.trim(),
    gender: profile.gender,
    dob: profile.dob,
  }
  await setDoc(doc(db, 'users', user.uid), data, { merge: true })
  return data
}
