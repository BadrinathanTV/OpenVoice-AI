/**
 * NoiseReductionToggle — Toggle DeepFilterNet3 Noise Reduction.
 */
export function NoiseReductionToggle({ isEnabled, onToggle }) {
  return (
    <div className="mode-toggle">
      <button
        className={`mode-toggle__btn ${isEnabled ? 'mode-toggle__btn--active' : ''}`}
        onClick={onToggle}
      >
        🛡️ Noise Reduction: {isEnabled ? 'ON' : 'OFF'}
      </button>
    </div>
  );
}
