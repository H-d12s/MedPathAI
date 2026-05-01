// PFL EMI at 9.99% p.a. reducing balance
export function calcEMI(principal, tenureMonths) {
  const r = 9.99 / 100 / 12   // monthly rate
  const emi = (principal * r * Math.pow(1 + r, tenureMonths)) /
              (Math.pow(1 + r, tenureMonths) - 1)
  return Math.round(emi)
}

// Returns Tailwind-compatible class names for eligibility decision
export function eligibilityMeta(decision) {
  switch (decision) {
    case 'GREEN':
      return {
        label: 'Pre-Approved',
        badgeClass: 'badge-green',
        icon: '✓',
        description: 'You are pre-approved. A PFL representative will contact you within 24 hours.',
      }
    case 'YELLOW':
      return {
        label: 'Needs Verification',
        badgeClass: 'badge-yellow',
        icon: '~',
        description: 'Your application needs further verification. Our team will reach out shortly.',
      }
    case 'RED':
      return {
        label: 'Not Eligible',
        badgeClass: 'badge-red',
        icon: '✕',
        description: 'Unfortunately, you don\'t meet the eligibility criteria at this time.',
      }
    default:
      return {
        label: 'Unknown',
        badgeClass: 'badge-yellow',
        icon: '?',
        description: 'Please upload your financial documents to check eligibility.',
      }
  }
}

// Format procedure name for display: "knee_replacement" → "Knee Replacement"
export function formatProcedure(name) {
  if (!name) return ''
  return name
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ')
}