import client from './client'

export const registerUser = (payload) =>
  client.post('/api/register', payload)

export const getProfile = (userId) =>
  client.get(`/api/profile/${userId}`)