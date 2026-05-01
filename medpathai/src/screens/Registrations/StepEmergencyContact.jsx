import { ArrowLeft } from 'lucide-react'
import { Spinner } from '../../components/ui'

export default function StepEmergencyContact({ form, update, loading, onBack, onFinish }) {
  return (
    <div>
      <h3 style={{
        fontFamily: 'var(--font-display)',
        fontSize: 'var(--text-lg)',
        fontWeight: 600,
        marginBottom: 8,
      }}>
        Emergency contact
      </h3>
      <p style={{
        color: 'var(--color-text-secondary)',
        fontSize: 'var(--text-sm)',
        marginBottom: 'var(--space-6)',
        lineHeight: 'var(--leading-relaxed)',
      }}>
        Shown to you during emergency mode so you can reach someone quickly. Optional.
      </p>

      <div style={{ marginBottom: 'var(--space-5)' }}>
        <label className="form-label">Contact name</label>
        <input
          className="form-input"
          placeholder="e.g. Rahul Sharma"
          value={form.emergency_contact_name}
          onChange={(e) => update({ emergency_contact_name: e.target.value })}
        />
      </div>

      <div style={{ marginBottom: 'var(--space-8)' }}>
        <label className="form-label">Contact phone</label>
        <input
          className="form-input"
          type="tel"
          placeholder="e.g. +91 98765 43210"
          value={form.emergency_contact_phone}
          onChange={(e) => update({ emergency_contact_phone: e.target.value })}
        />
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <button className="btn btn-outline" onClick={onBack} disabled={loading}>
          <ArrowLeft size={15} /> Back
        </button>
        <button
          className="btn btn-primary btn-lg"
          onClick={onFinish}
          disabled={loading}
        >
          {loading
            ? <><Spinner size={16} color="#fff" /> Saving…</>
            : 'Finish setup'}
        </button>
      </div>
    </div>
  )
}