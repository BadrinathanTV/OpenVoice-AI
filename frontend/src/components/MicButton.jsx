/**
 * MicButton — Circular microphone toggle button with pulse animation.
 * Follows SRP: only renders the button, delegates action via callback.
 */
export function MicButton({ isActive, onClick, disabled }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <button
        className={`mic-button ${isActive ? 'mic-button--active' : ''}`}
        onClick={onClick}
        disabled={disabled}
        aria-label={isActive ? 'Stop recording' : 'Start recording'}
        id="mic-toggle-btn"
      >
        {isActive ? (
          // Stop icon
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" />
          </svg>
        ) : (
          // Mic icon
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" fill="currentColor" />
            <path d="M19 10v2a7 7 0 0 1-14 0v-2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            <line x1="12" y1="19" x2="12" y2="23" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            <line x1="8" y1="23" x2="16" y2="23" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        )}
      </button>
      <div className="mic-button__label">
        {disabled ? 'Connecting…' : isActive ? 'Tap to Stop' : 'Tap to Speak'}
      </div>
    </div>
  );
}
