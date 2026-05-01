import client from './client'

export const sendMessage = (payload) =>
  client.post('/api/chat', payload)
// payload shape:
// { message, user_id, session_id?, selected_hospital?, user_lat?, user_lon? }