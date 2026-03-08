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

export const WS_BASE_URL = `ws://${window.location.hostname}:8000`;
export const API_BASE_URL = `http://${window.location.hostname}:8000`;
