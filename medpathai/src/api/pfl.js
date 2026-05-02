import client from './client'

export function getPFLApplications() {
  return client.get('/api/pfl/applications')
}

export function decidePFLApplication(referenceId, decision, officerNote = '') {
  return client.post('/api/pfl/decide', null, {
    params: {
      reference_id: referenceId,
      decision,
      officer_note: officerNote,
    },
  })
}
