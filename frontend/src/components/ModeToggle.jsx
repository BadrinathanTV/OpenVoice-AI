/**
 * ModeToggle — Switch between Voice and Text modes.
 * Follows SRP: only renders the toggle UI.
 */
export function ModeToggle({ mode, onSwitch }) {
  return (
    <div className="mode-toggle">
      <button
        className={`mode-toggle__btn ${mode === 'voice' ? 'mode-toggle__btn--active' : ''}`}
        onClick={() => onSwitch('voice')}
      >
        🎤 Voice
      </button>
      <button
        className={`mode-toggle__btn ${mode === 'text' ? 'mode-toggle__btn--active' : ''}`}
        onClick={() => onSwitch('text')}
      >
        💬 Text
      </button>
    </div>
  );
}
