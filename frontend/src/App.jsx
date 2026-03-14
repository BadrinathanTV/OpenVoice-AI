import { VoiceOrb } from './components/VoiceOrb';
import { MicButton } from './components/MicButton';
import { TranscriptPanel } from './components/TranscriptPanel';
import { TextInputBar } from './components/TextInputBar';
import { ConnectionStatus } from './components/ConnectionStatus';
import { ModeToggle } from './components/ModeToggle';
import { useVoicePipeline } from './hooks/useVoicePipeline';
import { AGENTS } from './config/agents';
import './index.css';

function App() {
  const {
    activeAgent,
    pipelineStatus,
    messages,
    mode,
    connectionStatus,
    isUserSpeaking,
    toggleVoice,
    sendTextMessage,
    switchMode,
  } = useVoicePipeline();

  const isRecording = pipelineStatus === 'recording';
  const userOrbState = isUserSpeaking ? 'speaking' : 'idle';
  const agentOrbState = pipelineStatus === 'speaking' ? 'speaking' : 'idle';
  const activeAgentInfo = AGENTS[activeAgent] || AGENTS.CustomerCare;
  const inactiveAgents = ['OrderOps', 'Shopper']
    .map((key) => AGENTS[key])
    .filter((agent) => agent.name !== activeAgent);

  return (
    <div className="app-layout">
      <header className="app-header">
        <div className="app-header__logo">
          <div className="app-header__logo-icon">O</div>
          <span>OpenVoice AI</span>
        </div>
        <div className="app-header__controls">
          <ModeToggle mode={mode} onSwitch={switchMode} />
          <ConnectionStatus status={connectionStatus} />
        </div>
      </header>

      <main className="dashboard">
        <section className="dashboard__main">
          <div className="dashboard__hero">
            <div className="dashboard-card dashboard-card--active-user glass-panel">
              <div className="dashboard-card__title">Active User</div>
              <div className="dashboard-card__body dashboard-card__body--user">
                <VoiceOrb variant="user" state={userOrbState} size="md" />
                <div className="dashboard-identity">
                  <div className="dashboard-identity__name">User</div>
                  <div className="dashboard-identity__badge">
                    {isRecording ? 'Listening' : 'Ready'}
                  </div>
                </div>
                {mode === 'voice' && (
                  <MicButton
                    isActive={isRecording}
                    onClick={toggleVoice}
                    disabled={wsStatus !== 'connected'}
                  />
                )}
              </div>
            </div>

            <div className="dashboard-card dashboard-card--active-agent glass-panel">
              <div className="dashboard-card__title">Active Voice Agent</div>
              <div className="dashboard-card__body dashboard-card__body--agent">
                <VoiceOrb agentName={activeAgent} state={agentOrbState} size="md" />
                <div className="dashboard-identity">
                  <div
                    className="dashboard-identity__name dashboard-identity__name--agent"
                    style={{ color: activeAgentInfo.color }}
                  >
                    {activeAgentInfo.label}
                  </div>
                  <div
                    className="dashboard-identity__badge"
                    style={{
                      color: activeAgentInfo.color,
                      borderColor: `${activeAgentInfo.color}55`,
                      backgroundColor: `${activeAgentInfo.color}1a`,
                    }}
                  >
                    {pipelineStatus === 'speaking' ? 'Speaking' : 'Active'}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="dashboard-card dashboard-card--inactive glass-panel">
            <div className="dashboard-card__title">Inactive Agents</div>
            <div className="inactive-agent-list inactive-agent-list--row">
              {inactiveAgents.map((agent) => (
                <div className="inactive-agent" key={agent.name}>
                  <div
                    className="inactive-agent__wave"
                    style={{
                      '--mini-wave-color': agent.color,
                      '--mini-wave-glow': agent.glow,
                    }}
                  >
                    <span className="inactive-agent__wave-ring inactive-agent__wave-ring--1" />
                    <span className="inactive-agent__wave-ring inactive-agent__wave-ring--2" />
                    <span className="inactive-agent__wave-ring inactive-agent__wave-ring--3" />
                  </div>
                  <div className="inactive-agent__name">{agent.label}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <aside className="dashboard-debug">
          <div className="dashboard-card dashboard-card--transcript glass-panel">
            <TranscriptPanel messages={messages} />
            {mode === 'text' && (
              <TextInputBar
                onSend={sendTextMessage}
                disabled={wsStatus !== 'connected'}
              />
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;
