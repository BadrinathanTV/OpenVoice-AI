import { useState, useCallback } from 'react';

/**
 * TextInputBar — Text message input at the bottom of the transcript panel.
 * Follows SRP: only handles text input UI, delegates sending via callback.
 */
export function TextInputBar({ onSend, disabled }) {
  const [text, setText] = useState('');

  const handleSubmit = useCallback(
    (e) => {
      e.preventDefault();
      if (text.trim() && !disabled) {
        onSend(text.trim());
        setText('');
      }
    },
    [text, onSend, disabled]
  );

  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit(e);
      }
    },
    [handleSubmit]
  );

  return (
    <form className="text-input-bar" onSubmit={handleSubmit}>
      <input
        className="text-input-bar__input"
        type="text"
        placeholder="Type a message…"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        id="text-chat-input"
        autoComplete="off"
      />
      <button
        className="text-input-bar__send"
        type="submit"
        disabled={disabled || !text.trim()}
        aria-label="Send message"
        id="send-message-btn"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
    </form>
  );
}
