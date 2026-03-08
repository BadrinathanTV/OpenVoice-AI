import { useRef, useEffect } from 'react';
import { AGENTS } from '../config/agents';

/**
 * TranscriptPanel — Right sidebar showing conversation messages.
 * Follows SRP: only renders messages, no state management.
 *
 * @param {Array} messages - Array of { role: 'user'|'ai', text: string, agent?: string }
 * @param {string} activeAgent - Current active agent key
 */
export function TranscriptPanel({ messages, activeAgent }) {
  const scrollRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="transcript-panel">
      <div className="transcript-panel__header">
        Session Transcript
      </div>
      <div className="transcript-panel__messages" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="transcript-panel__empty">
            Start a conversation by clicking{' '}
            the microphone or typing a message below.
          </div>
        ) : (
          messages.map((msg, i) => (
            <MessageBubble key={i} message={msg} />
          ))
        )}
      </div>
    </div>
  );
}

/**
 * MessageBubble — Individual message in the transcript.
 * Extracted as a separate component following SRP.
 */
function MessageBubble({ message }) {
  const { role, text, agent } = message;
  const agentInfo = agent ? AGENTS[agent] : null;

  return (
    <div className={`message message--${role}`}>
      <span className="message__sender">
        {role === 'user' ? 'You' : agentInfo?.label || 'AI'}
      </span>
      <div className="message__bubble">
        {text}
        {role === 'ai' && agentInfo && (
          <div className="message__agent-tag" style={{ color: agentInfo.color }}>
            via {agentInfo.label}
          </div>
        )}
      </div>
    </div>
  );
}
