import { useEffect, useRef } from 'react';
import { AGENTS } from '../config/agents';

export function VoiceOrb({ agentName, state = 'idle', variant = 'agent', size = 'lg' }) {
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animFrameRef = useRef(null);
  const canvasSizeRef = useRef({ width: 0, height: 0 });
  const agent = AGENTS[agentName] || AGENTS.CustomerCare;
  const palette = variant === 'user'
    ? { color: '#B86CFF', glow: 'rgba(184, 108, 255, 0.42)' }
    : agent;

  useEffect(() => {
    if (variant !== 'agent') return undefined;

    document.documentElement.style.setProperty('--agent-color', agent.color);
    document.documentElement.style.setProperty('--agent-glow', agent.glow);
    return undefined;
  }, [agent, variant]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;

    const ctx = canvas.getContext('2d');
    if (!ctx) return undefined;

    const resize = () => {
      const { width, height } = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;

      canvasSizeRef.current = { width, height };
      canvas.width = Math.max(1, Math.round(width * dpr));
      canvas.height = Math.max(1, Math.round(height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
    };

    const hexToRgb = (hex) => ({
      r: parseInt(hex.slice(1, 3), 16),
      g: parseInt(hex.slice(3, 5), 16),
      b: parseInt(hex.slice(5, 7), 16),
    });

    const spawnParticle = () => {
      const angle = Math.random() * Math.PI * 2;
      const speed = 0.3 + Math.random() * 1.1;
      const { width, height } = canvasSizeRef.current;
      const centerX = width / 2;
      const centerY = height / 2;

      return {
        x: centerX + (Math.random() - 0.5) * 28,
        y: centerY + (Math.random() - 0.5) * 28,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 1,
        decay: 0.006 + Math.random() * 0.012,
        size: 1.4 + Math.random() * 2.4,
      };
    };

    const { r, g, b } = hexToRgb(palette.color);

    const animate = () => {
      const { width, height } = canvasSizeRef.current;
      ctx.clearRect(0, 0, width, height);

      if (state === 'speaking') {
        for (let i = 0; i < 3; i += 1) {
          if (particlesRef.current.length < 45) {
            particlesRef.current.push(spawnParticle());
          }
        }
      }

      particlesRef.current = particlesRef.current.filter((particle) => {
        particle.x += particle.vx;
        particle.y += particle.vy;
        particle.life -= particle.decay;

        if (particle.life <= 0) return false;

        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size * particle.life, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${particle.life * 0.62})`;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size * particle.life * 2.5, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${particle.life * 0.12})`;
        ctx.fill();

        return true;
      });

      animFrameRef.current = requestAnimationFrame(animate);
    };

    resize();
    window.addEventListener('resize', resize);
    animate();

    return () => {
      window.removeEventListener('resize', resize);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [palette.color, state]);

  return (
    <div
      className={`orb-container orb-container--${size}`}
      data-state={state}
      style={{
        '--orb-color': palette.color,
        '--orb-glow': palette.glow,
      }}
    >
      <canvas ref={canvasRef} className="orb-particles" />
      <div className="orb-halo" />
      <div className="orb-ring orb-ring--1" />
      <div className="orb-ring orb-ring--2" />
      <div className="orb-ring orb-ring--3" />
      <div className="orb-wave orb-wave--1" />
      <div className="orb-wave orb-wave--2" />
      <div className="orb-wave orb-wave--3" />
      <div className="orb-core" />
    </div>
  );
}
