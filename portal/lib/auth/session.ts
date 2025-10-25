import { createHmac, timingSafeEqual } from 'crypto';

const SESSION_COOKIE_NAME = 'hmco_portal';
const DEFAULT_SESSION_TTL_MS = 12 * 60 * 60 * 1000; // 12 hours

interface SessionPayload {
  username: string;
  issuedAt: number;
  expiresAt: number;
}

function getSessionSecret(): string {
  const secret = process.env.PORTAL_SESSION_SECRET;
  if (!secret) {
    console.warn(
      'PORTAL_SESSION_SECRET is not set. Falling back to insecure default secret.'
    );
  }
  return secret || 'development-insecure-secret';
}

function encodePayload(payload: SessionPayload): string {
  return Buffer.from(JSON.stringify(payload)).toString('base64url');
}

function decodePayload(encoded: string): SessionPayload | null {
  try {
    const json = Buffer.from(encoded, 'base64url').toString('utf8');
    return JSON.parse(json) as SessionPayload;
  } catch (error) {
    console.error('Failed to decode session payload:', error);
    return null;
  }
}

function sign(data: string): string {
  return createHmac('sha256', getSessionSecret())
    .update(data)
    .digest('base64url');
}

export function createSessionToken(username: string): string {
  const now = Date.now();
  const ttl =
    Number(process.env.PORTAL_SESSION_TTL_MS) || DEFAULT_SESSION_TTL_MS;

  const payload: SessionPayload = {
    username,
    issuedAt: now,
    expiresAt: now + ttl,
  };

  const encoded = encodePayload(payload);
  const signature = sign(encoded);
  return `${encoded}.${signature}`;
}

export function verifySessionToken(
  token?: string | null
): SessionPayload | null {
  if (!token) return null;
  const parts = token.split('.');
  if (parts.length !== 2) return null;

  const [encoded, signature] = parts;
  const expectedSignature = sign(encoded);

  try {
    const provided = Buffer.from(signature, 'base64url');
    const expected = Buffer.from(expectedSignature, 'base64url');

    if (
      provided.length !== expected.length ||
      !timingSafeEqual(provided, expected)
    ) {
      return null;
    }
  } catch (error) {
    console.error('Session signature comparison failed:', error);
    return null;
  }

  const payload = decodePayload(encoded);
  if (!payload) return null;

  if (payload.expiresAt < Date.now()) {
    return null;
  }

  return payload;
}

export function getSessionCookieName(): string {
  return SESSION_COOKIE_NAME;
}

export function getSessionCookieOptions() {
  const ttl =
    Number(process.env.PORTAL_SESSION_TTL_MS) || DEFAULT_SESSION_TTL_MS;

  return {
    httpOnly: true as const,
    secure: process.env.NODE_ENV === 'production' ? true : false,
    sameSite: 'lax' as const,
    path: '/',
    maxAge: Math.floor(ttl / 1000),
  };
}

export type SessionUser = {
  username: string;
};
