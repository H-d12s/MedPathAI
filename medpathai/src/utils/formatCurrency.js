// Formats a number as Indian rupees: ₹1,23,456
export function formatINR(amount, compact = false) {
  if (amount === null || amount === undefined) return '—'

  if (compact && amount >= 100000) {
    const lakhs = amount / 100000
    return `₹${lakhs % 1 === 0 ? lakhs : lakhs.toFixed(1)}L`
  }

  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 0,
  }).format(amount)
}

// e.g. formatINR(123456) → "₹1,23,456"
// e.g. formatINR(150000, true) → "₹1.5L"