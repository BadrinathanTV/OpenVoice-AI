/**
 * ConnectionStatus — Header badge showing realtime transport state.
 * Follows SRP: pure presentational component.
 */
export function ConnectionStatus({ status }) {
  const labels = {
    connected: 'Connected',
    connecting: 'Connecting…',
    disconnected: 'Disconnected',
  };

  return (
    <div className="app-header__status">
      <span className={`status-dot status-dot--${status}`} />
      <span>{labels[status] || 'Unknown'}</span>
    </div>
  );
}
