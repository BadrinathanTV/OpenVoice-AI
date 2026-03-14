export const AGENTS = {
  CustomerCare: {
    name: 'CustomerCare',
    label: 'Customer Care',
    description: 'General help, returns, refunds, policies',
    color: '#4B8DFF',
    glow: 'rgba(75, 141, 255, 0.45)',
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
    color: '#FF6FAE',
    glow: 'rgba(255, 111, 174, 0.42)',
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
