/**
 * Agent configuration — single source of truth for agent identities.
 * Following the Single Responsibility Principle: this module only defines agent metadata.
 */

export const AGENTS = {
  CustomerCare: {
    name: 'CustomerCare',
    label: 'Customer Care',
    description: 'General help, returns, refunds, policies',
    color: '#6C63FF',
    glow: 'rgba(108, 99, 255, 0.4)',
  },
  Shopper: {
    name: 'Shopper',
    label: 'Shopper',
    description: 'Product search and recommendations',
    color: '#00C9A7',
    glow: 'rgba(0, 201, 167, 0.4)',
  },
  OrderOps: {
    name: 'OrderOps',
    label: 'Order Ops',
    description: 'Order tracking and operations',
    color: '#FF6B6B',
    glow: 'rgba(255, 107, 107, 0.4)',
  },
};

export const DEFAULT_AGENT = 'CustomerCare';

const sanitizeBaseUrl = (value) => value?.replace(/\/+$/, '');

const configuredApiBaseUrl = sanitizeBaseUrl(import.meta.env.VITE_API_BASE_URL);
const configuredWsBaseUrl = sanitizeBaseUrl(import.meta.env.VITE_WS_BASE_URL);
const defaultHttpProtocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
const defaultWsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

export const API_BASE_URL =
  configuredApiBaseUrl ||
  `${defaultHttpProtocol}//${window.location.hostname}:8000`;

export const WS_BASE_URL =
  configuredWsBaseUrl ||
  (configuredApiBaseUrl
    ? configuredApiBaseUrl.replace(/^http/i, 'ws')
    : `${defaultWsProtocol}//${window.location.hostname}:8000`);
