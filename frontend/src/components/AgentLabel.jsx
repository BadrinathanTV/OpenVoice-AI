import { AGENTS } from '../config/agents';

/**
 * AgentLabel — Shows the active agent name and status badge below the orb.
 * Follows SRP: pure presentational component.
 */
export function AgentLabel({ agentName, status }) {
  const agent = AGENTS[agentName] || AGENTS.CustomerCare;

  const statusLabels = {
    idle: 'Ready',
    recording: 'Listening…',
    processing: 'Transcribing…',
    thinking: 'Thinking…',
    speaking: 'Speaking…',
  };

  return (
    <div className="agent-label">
      <div className="agent-label__name">{agent.label}</div>
      <div className="agent-label__status">
        <span className="agent-label__badge">
          {statusLabels[status] || 'Ready'}
        </span>
      </div>
    </div>
  );
}
