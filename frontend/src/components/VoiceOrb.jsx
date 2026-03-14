import { useRef, useEffect } from 'react';
import { AGENTS } from '../config/agents';

/**
 * VoiceOrb — The animated agent orb with particle effects.
 * Follows SRP: only renders the orb visualization, no business logic.
 *
 * @param {string} agentName - Active agent key (e.g., 'CustomerCare')
 * @param {string} state - Animation state: 'idle' | 'recording' | 'processing' | 'thinking' | 'speaking'
 */
export function VoiceOrb({ agentName, state = 'idle' }) {
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animFrameRef = useRef(null);
  const canvasSizeRef = useRef({ width: 0, height: 0 });
  const agent = AGENTS[agentName] || AGENTS.CustomerCare;

  // Update CSS custom properties for the agent color
  useEffect(() => {
    document.documentElement.style.setProperty('--agent-color', agent.color);
    document.documentElement.style.setProperty('--agent-glow', agent.glow);
  }, [agent]);

  // Particle system for speaking state
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      const { width, height } = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;

      canvasSizeRef.current = { width, height };
      canvas.width = Math.max(1, Math.round(width * dpr));
      canvas.height = Math.max(1, Math.round(height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
    };
    resize();
    window.addEventListener('resize', resize);

    const hexToRgb = (hex) => {
      const r = parseInt(hex.slice(1, 3), 16);
      const g = parseInt(hex.slice(3, 5), 16);
      const b = parseInt(hex.slice(5, 7), 16);
      return { r, g, b };
    };

    const spawnParticle = () => {
      const angle = Math.random() * Math.PI * 2;
      const speed = 0.3 + Math.random() * 1.2;
      const { width, height } = canvasSizeRef.current;
      const centerX = width / 2;
      const centerY = height / 2;
      return {
        x: centerX + (Math.random() - 0.5) * 40,
        y: centerY + (Math.random() - 0.5) * 40,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 1.0,
        decay: 0.005 + Math.random() * 0.015,
        size: 1.5 + Math.random() * 3,
      };
    };

    const animate = () => {
      const { width, height } = canvasSizeRef.current;
      ctx.clearRect(0, 0, width, height);

      if (state === 'speaking') {
        // Spawn 2–3 particles per frame
        for (let i = 0; i < 3; i++) {
          if (particlesRef.current.length < 50) {
            particlesRef.current.push(spawnParticle());
          }
        }
      }

      const { r, g, b } = hexToRgb(agent.color);

      particlesRef.current = particlesRef.current.filter((p) => {
        p.x += p.vx;
        p.y += p.vy;
        p.life -= p.decay;

        if (p.life <= 0) return false;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${p.life * 0.6})`;
        ctx.fill();

        // Glow effect
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * p.life * 2.5, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${p.life * 0.15})`;
        ctx.fill();

        return true;
      });

      animFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resize);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [state, agent.color]);

  return (
    <div className="orb-container" data-state={state}>
      <canvas ref={canvasRef} className="orb-particles" />
      <div className="orb-ring orb-ring--1" />
      <div className="orb-ring orb-ring--2" />
      <div className="orb-ring orb-ring--3" />
      <div className="orb-core" />
    </div>
  );
}
