import client from './client'

export const saveFinancials = (payload) =>
  client.post('/api/financials', payload)