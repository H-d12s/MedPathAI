// ── Spinner ──────────────────────────────────────────────────
export function Spinner({ size = 20, color = 'var(--color-accent)' }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      style={{ animation: 'spin 0.7s linear infinite', flexShrink: 0 }}
    >
      <circle cx="12" cy="12" r="10" stroke={color} strokeWidth="2" strokeOpacity="0.2" />
      <path
        d="M12 2a10 10 0 0 1 10 10"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  )
}

// ── Badge ────────────────────────────────────────────────────
export function Badge({ children, variant = 'default', className = '' }) {
  const variantMap = {
    nabh:      'badge-nabh',
    jci:       'badge-jci',
    emergency: 'badge-emergency',
    metro:     'badge-metro',
    tier2:     'badge-tier2',
    green:     'badge-green',
    yellow:    'badge-yellow',
    red:       'badge-red',
    default:   '',
  }
  return (
    <span className={`badge ${variantMap[variant] || ''} ${className}`}>
      {children}
    </span>
  )
}

// ── Rating Stars ─────────────────────────────────────────────
export function RatingStars({ rating }) {
  const full  = Math.floor(rating)
  const half  = rating % 1 >= 0.5 ? 1 : 0
  const empty = 5 - full - half
  const star = (type, i) => (
    <span
      key={type + i}
      style={{
        fontSize: '13px',
        color: type === 'empty' ? 'var(--gray-300)' : 'var(--amber-400)',
      }}
    >
      {type === 'half' ? '⭐' : type === 'full' ? '★' : '☆'}
    </span>
  )
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 1 }}>
      {Array.from({ length: full }, (_, i) => star('full', i))}
      {half ? star('half', 0) : null}
      {Array.from({ length: empty }, (_, i) => star('empty', i))}
      <span style={{
        marginLeft: 4,
        fontSize: 'var(--text-xs)',
        color: 'var(--color-text-secondary)',
        fontWeight: 'var(--weight-medium)',
      }}>
        {rating.toFixed(1)}
      </span>
    </span>
  )
}

// ── Toast container ──────────────────────────────────────────
import { useUIStore } from '../store/uiStore'

export function ToastContainer() {
  const { toasts, dismissToast } = useUIStore()
  if (!toasts.length) return null

  const typeStyles = {
    success: { background: 'var(--green-600)', color: '#fff' },
    error:   { background: 'var(--red-600)',   color: '#fff' },
    warning: { background: 'var(--amber-500)', color: '#fff' },
    info:    { background: 'var(--navy-800)',  color: '#fff' },
  }

  return (
    <div style={{
      position: 'fixed', bottom: 24, right: 24, zIndex: 1000,
      display: 'flex', flexDirection: 'column', gap: 8,
    }}>
      {toasts.map((t) => (
        <div
          key={t.id}
          className="animate-slide-up"
          onClick={() => dismissToast(t.id)}
          style={{
            ...typeStyles[t.type],
            padding: '10px 16px',
            borderRadius: 'var(--radius-lg)',
            fontSize: 'var(--text-sm)',
            fontWeight: 'var(--weight-medium)',
            boxShadow: 'var(--shadow-lg)',
            cursor: 'pointer',
            maxWidth: 360,
            lineHeight: 1.5,
          }}
        >
          {t.message}
        </div>
      ))}
    </div>
  )
}

// ── Eligibility pill ─────────────────────────────────────────
export function EligibilityPill({ decision }) {
  const map = {
    GREEN:  { label: 'Pre-Approved',       variant: 'green' },
    YELLOW: { label: 'Needs Verification', variant: 'yellow' },
    RED:    { label: 'Not Eligible',       variant: 'red' },
  }
  const m = map[decision] || { label: decision, variant: 'default' }
  return <Badge variant={m.variant}>{m.label}</Badge>
}