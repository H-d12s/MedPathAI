import { Stethoscope } from 'lucide-react'

export default function PossibleCauses({ causes = [], icd10 }) {
  const signals = Array.from(new Set((causes || []).filter(Boolean))).slice(0, 4)

  if (!signals.length && !icd10) return null

  return (
    <div className="card" style={{
      padding: '12px 14px',
      borderRadius: 'var(--radius-lg)',
      boxShadow: 'var(--shadow-sm)',
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 7,
        marginBottom: signals.length ? 8 : 0,
      }}>
        <Stethoscope size={15} color="var(--teal-700)" />
        <h3 style={{
          fontSize: 'var(--text-sm)',
          fontWeight: 700,
          letterSpacing: 0,
        }}>
          Clinical Signals
        </h3>
      </div>
      {signals.length > 0 && (
        <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
          {signals.map((signal) => (
            <span
              key={signal}
              className="badge badge-metro"
              style={{
                border: '1px solid rgba(176, 200, 228, 0.35)',
                borderRadius: 'var(--radius-full)',
                padding: '4px 9px',
                fontSize: 'var(--text-xs)',
                fontWeight: 500,
                letterSpacing: 0,
              }}
            >
              {signal}
            </span>
          ))}
        </div>
      )}
      {icd10 && (
        <p style={{ marginTop: 7, fontSize: 'var(--text-xs)', color: 'var(--color-text-secondary)' }}>
          ICD-10 hint: <strong>{icd10}</strong>
        </p>
      )}
    </div>
  )
}
