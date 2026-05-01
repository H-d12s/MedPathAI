import { Check } from 'lucide-react'

// steps: [{ label }]
// current: 0-based index
export default function StepIndicator({ steps, current }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 0,
      marginBottom: 'var(--space-8)',
    }}>
      {steps.map((step, i) => {
        const done   = i < current
        const active = i === current

        return (
          <div key={i} style={{ display: 'flex', alignItems: 'center', flex: i < steps.length - 1 ? 1 : 'none' }}>
            {/* Circle */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
              <div style={{
                width: 32, height: 32,
                borderRadius: '50%',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                flexShrink: 0,
                fontSize: 'var(--text-xs)',
                fontWeight: 'var(--weight-semibold)',
                transition: 'all var(--transition-base)',
                background: done
                  ? 'var(--teal-500)'
                  : active
                    ? 'var(--navy-900)'
                    : 'var(--gray-100)',
                color: done || active ? '#fff' : 'var(--gray-400)',
                border: active ? '2px solid var(--navy-900)' : '2px solid transparent',
                boxShadow: active ? '0 0 0 3px rgba(10,22,40,0.1)' : 'none',
              }}>
                {done ? <Check size={14} strokeWidth={2.5} /> : i + 1}
              </div>
              <span style={{
                fontSize: 'var(--text-xs)',
                fontWeight: active ? 'var(--weight-medium)' : 'var(--weight-regular)',
                color: active
                  ? 'var(--color-text-primary)'
                  : done
                    ? 'var(--teal-600)'
                    : 'var(--color-text-muted)',
                whiteSpace: 'nowrap',
              }}>
                {step.label}
              </span>
            </div>

            {/* Connector line */}
            {i < steps.length - 1 && (
              <div style={{
                flex: 1,
                height: 2,
                marginBottom: 22, // align with circles not labels
                background: done ? 'var(--teal-300)' : 'var(--gray-200)',
                transition: 'background var(--transition-base)',
              }} />
            )}
          </div>
        )
      })}
    </div>
  )
}