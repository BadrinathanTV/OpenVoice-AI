import { VoiceOrb } from './components/VoiceOrb';
import { AgentLabel } from './components/AgentLabel';
import { MicButton } from './components/MicButton';
import { TranscriptPanel } from './components/TranscriptPanel';
import { TextInputBar } from './components/TextInputBar';
import { ConnectionStatus } from './components/ConnectionStatus';
import { ModeToggle } from './components/ModeToggle';
import { useVoicePipeline } from './hooks/useVoicePipeline';
import './index.css';

/**
 * App — Root component that assembles the full UI from subcomponents.
 * Follows Dependency Inversion: components receive data via props from the pipeline hook,
 * rather than reaching into global state.
 */
function App() {
  const {
    activeAgent,
    pipelineStatus,
    messages,
    mode,
    connectionStatus,
    toggleVoice,
    sendTextMessage,
    switchMode,
  } = useVoicePipeline();

  const isRecording = pipelineStatus === 'recording';

  return (
    <div className="app-layout">
      {/* ─── Header ─── */}
      <header className="app-header">
        <div className="app-header__logo">
          <div className="app-header__logo-icon">🎙</div>
          <span>OpenVoice AI</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <ModeToggle mode={mode} onSwitch={switchMode} />
          <ConnectionStatus status={connectionStatus} />
        </div>
      </header>

      {/* ─── Voice Panel (Left) ─── */}
      <main className="voice-panel">
        <VoiceOrb agentName={activeAgent} state={pipelineStatus} />
        <AgentLabel agentName={activeAgent} status={pipelineStatus} />

        {mode === 'voice' && (
          <MicButton
            isActive={isRecording}
            onClick={toggleVoice}
            disabled={connectionStatus !== 'connected'}
          />
        )}

        {mode === 'text' && (
          <div style={{ marginTop: 48, width: '100%', maxWidth: 400 }}>
            <TextInputBar
              onSend={sendTextMessage}
              disabled={connectionStatus !== 'connected'}
            />
          </div>
        )}
      </main>

      {/* ─── Transcript Panel (Right) ─── */}
      <TranscriptPanel messages={messages} />
    </div>
  );
}

export default App;
