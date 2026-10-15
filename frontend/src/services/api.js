import { auth } from './firebase'

const API_BASE = import.meta.env.VITE_API_BASE_URL?.replace(/\/+$/, '') || '/api'

async function getHeaders(user, json = true) {
  const currentUser = auth?.currentUser
  if (!currentUser || currentUser.uid !== user?.uid) {
    throw new Error('Your session has expired. Please sign in again.')
  }

  const headers = {
    'X-User-Id': currentUser.uid,
    'X-Id-Token': await currentUser.getIdToken(),
  }
  if (json) headers['Content-Type'] = 'application/json'
  return headers
}

async function readResponse(response) {
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.detail || `Request failed (${response.status}).`)
  }
  return data
}

export async function sendChatMessage(message, city = '', user) {
  const headers = await getHeaders(user)
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ message, city: city || user.city || '' }),
  })
  return readResponse(response)
}

export async function sendVoiceChat(audioBlob, city = '', user) {
  const formData = new FormData()
  formData.append('audio', audioBlob, 'recording.wav')
  formData.append('city', city || user.city || '')

  const response = await fetch(`${API_BASE}/chat/voice`, {
    method: 'POST',
    headers: await getHeaders(user, false),
    body: formData,
  })
  return readResponse(response)
}

export async function sendImageAnalysis(imageFile, question = '', user) {
  const formData = new FormData()
  formData.append('image', imageFile)
  if (question) formData.append('question', question)

  const response = await fetch(`${API_BASE}/chat/image`, {
    method: 'POST',
    headers: await getHeaders(user, false),
    body: formData,
  })
  return readResponse(response)
}

export async function fetchHistory(user) {
  const response = await fetch(`${API_BASE}/history`, {
    headers: await getHeaders(user),
  })
  return readResponse(response)
}

export async function clearHistory(user) {
  const response = await fetch(`${API_BASE}/history`, {
    method: 'DELETE',
    headers: await getHeaders(user),
  })
  return readResponse(response)
}
