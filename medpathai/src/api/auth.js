const USER_ID_KEY = 'medpath_user_id'

// Generate a stable userId for this browser.
// In a real auth system this would come from a login token.
export function getUserId() {
  let userId = localStorage.getItem(USER_ID_KEY)
  if (!userId) {
    userId = 'user_' + Math.random().toString(36).slice(2, 11)
    localStorage.setItem(USER_ID_KEY, userId)
  }
  return userId
}

export function clearUserId() {
  localStorage.removeItem(USER_ID_KEY)
}

export function isRegistered() {
  return !!localStorage.getItem('medpath_registered')
}

export function markRegistered() {
  localStorage.setItem('medpath_registered', '1')
}

export function clearRegistration() {
  localStorage.removeItem('medpath_registered')
}