import { createRemoteJWKSet, jwtVerify, JWTPayload } from 'jose';
import type { NextRequest } from 'next/server';

export type AccessUser = {
  email: string;
  name?: string;
  sub?: string;
  issuedAt?: number;
  expiresAt?: number;
};

const accessAudiences = (process.env.CLOUDFLARE_ACCESS_AUD ?? process.env.CLOUDFLARE_ACCESS_AUDIENCE ?? '')
  .split(',')
  .map(value => value.trim())
  .filter(Boolean);

const devBypassEmail =
  process.env.PORTAL_DEV_BYPASS_USER ??
  process.env.PORTAL_DEV_USER_EMAIL ??
  process.env.PORTAL_DEV_BYPASS_EMAIL ??
  '';

function normalizeDomain(domain?: string | null): string | null {
  if (!domain) return null;
  return domain.replace(/^https?:\/\//, '').replace(/\/+$/, '').trim() || null;
}

const accessDomain =
  normalizeDomain(process.env.CLOUDFLARE_ACCESS_TEAM_DOMAIN) ??
  normalizeDomain(process.env.NEXT_PUBLIC_CLOUDFLARE_ACCESS_DOMAIN);

const portalBaseUrl =
  process.env.NEXT_PUBLIC_PORTAL_URL ??
  process.env.PORTAL_EXTERNAL_URL ??
  process.env.PORTAL_URL ??
  '';

let remoteJwks: ReturnType<typeof createRemoteJWKSet> | null = null;

function getRemoteJwks() {
  if (!accessDomain) {
    throw new Error('Cloudflare Access team domain is not configured.');
  }

  if (!remoteJwks) {
    remoteJwks = createRemoteJWKSet(
      new URL(`https://${accessDomain}/cdn-cgi/access/certs`)
    );
  }

  return remoteJwks;
}

export function isCloudflareAccessEnabled(): boolean {
  return Boolean(accessDomain && accessAudiences.length > 0);
}

export function extractAccessToken(req: NextRequest): string | null {
  const headerToken =
    req.headers.get('cf-access-jwt-assertion') ??
    req.headers.get('Cf-Access-Jwt-Assertion') ??
    req.headers.get('CF_Authorization');

  if (headerToken) {
    return headerToken;
  }

  const bearer = req.headers.get('authorization');
  if (bearer?.startsWith('Bearer ')) {
    return bearer.slice('Bearer '.length);
  }

  const cookieTokenCandidates = new Set([
    'CF_Authorization',
    'CF_Authorization',
    'CF_AUTHORIZATION',
    'cf_authorization',
    'cf_access_token',
  ]);

  for (const name of cookieTokenCandidates) {
    const value = req.cookies.get(name)?.value;
    if (value) {
      return value;
    }
  }

  return null;
}

function mapJwtPayload(payload: JWTPayload): AccessUser {
  const identity = (payload.identity ??
    payload.user ??
    {}) as Partial<{ email: string; name: string }>;

  const email =
    (payload.email as string | undefined) ??
    identity.email ??
    (payload.sub as string | undefined);

  if (!email) {
    throw new Error('Cloudflare Access token missing email claim');
  }

  return {
    email,
    name:
      (payload.name as string | undefined) ??
      identity.name ??
      (payload.common_name as string | undefined),
    sub: payload.sub,
    issuedAt: payload.iat ? payload.iat * 1000 : undefined,
    expiresAt: payload.exp ? payload.exp * 1000 : undefined,
  };
}

export async function verifyAccessToken(token: string): Promise<AccessUser> {
  if (!isCloudflareAccessEnabled()) {
    throw new Error('Cloudflare Access is not configured.');
  }

  const { payload } = await jwtVerify(token, getRemoteJwks(), {
    audience: accessAudiences.length > 0 ? accessAudiences : undefined,
  });

  return mapJwtPayload(payload);
}

export async function getUserFromRequest(
  req: NextRequest
): Promise<AccessUser | null> {
  const token = extractAccessToken(req);

  if (token) {
    try {
      return await verifyAccessToken(token);
    } catch (error) {
      console.error('Failed to verify Cloudflare Access token', error);
      return null;
    }
  }

  if (!isCloudflareAccessEnabled() && devBypassEmail) {
    return {
      email: devBypassEmail,
      name: 'Dev Bypass User',
      sub: 'dev-bypass',
    };
  }

  return null;
}

export function buildCloudflareLoginUrl(returnTo?: string | null): string {
  if (!accessDomain) {
    throw new Error('Cloudflare Access team domain is not configured.');
  }

  const url = new URL(`https://${accessDomain}/cdn-cgi/access/login`);
  const redirectTarget =
    returnTo ??
    (portalBaseUrl
      ? portalBaseUrl
      : typeof window !== 'undefined'
      ? window.location.href
      : '');

  if (redirectTarget) {
    url.searchParams.set('redirect_url', redirectTarget);
  }

  return url.toString();
}

export function buildCloudflareLogoutUrl(returnTo?: string | null): string {
  if (!accessDomain) {
    throw new Error('Cloudflare Access team domain is not configured.');
  }

  const url = new URL(`https://${accessDomain}/cdn-cgi/access/logout`);
  const redirectTarget =
    returnTo ??
    (portalBaseUrl
      ? portalBaseUrl
      : typeof window !== 'undefined'
      ? window.location.origin
      : '');

  if (redirectTarget) {
    url.searchParams.set('returnTo', redirectTarget);
  }

  return url.toString();
}
