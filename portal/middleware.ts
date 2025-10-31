import { NextRequest, NextResponse } from 'next/server';
import {
  buildCloudflareLoginUrl,
  getUserFromRequest,
  isCloudflareAccessEnabled,
} from '@/lib/auth/cloudflare';

const PUBLIC_PATHS = new Set([
  '/login',
  '/api/auth/session',
  '/api/auth/login',
  '/api/auth/logout',
  '/_next/static',
  '/_next/image',
  '/favicon.ico',
]);

// E2E bypass configuration - only active when E2E_BYPASS=1
const E2E_BYPASS_ENABLED = process.env.E2E_BYPASS === '1';
const E2E_BYPASS_HEADER = 'x-e2e-bypass';
const E2E_BYPASS_USER_EMAIL = process.env.E2E_BYPASS_USER_EMAIL || 'e2e-test@example.com';

function isPublicPath(pathname: string) {
  if (pathname === '/') return false;
  if (PUBLIC_PATHS.has(pathname)) return true;
  return Array.from(PUBLIC_PATHS).some(
    (path) => path !== '/' && pathname.startsWith(`${path}/`)
  );
}

/**
 * Check if request has E2E bypass header (dev/test mode only)
 * This allows E2E tests to bypass Cloudflare Access authentication
 * Only works when E2E_BYPASS=1 environment variable is set
 */
function hasE2EBypass(req: NextRequest): boolean {
  if (!E2E_BYPASS_ENABLED) {
    return false;
  }
  
  const bypassHeader = req.headers.get(E2E_BYPASS_HEADER);
  return !!bypassHeader;
}

export async function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  if (
    pathname.startsWith('/_next/') ||
    pathname.startsWith('/static/') ||
    pathname.match(/\.(png|jpg|jpeg|gif|webp|svg|ico)$/i)
  ) {
    return NextResponse.next();
  }

  // Check for E2E bypass (dev/test mode only)
  const hasE2EBypassHeader = hasE2EBypass(req);
  if (hasE2EBypassHeader) {
    // Allow E2E tests to bypass authentication
    const requestHeaders = new Headers(req.headers);
    requestHeaders.set('x-authenticated-user-email', E2E_BYPASS_USER_EMAIL);
    requestHeaders.set('x-authenticated-user-name', 'E2E Test User');
    return NextResponse.next({
      request: {
        headers: requestHeaders,
      },
    });
  }

  const user = await getUserFromRequest(req);
  const isPublic = isPublicPath(pathname);

  if (user && pathname === '/login') {
    const url = req.nextUrl.clone();
    url.pathname = '/';
    url.searchParams.delete('next');
    return NextResponse.redirect(url);
  }

  if (!user && !isPublic) {
    if (!isCloudflareAccessEnabled()) {
      // Allow access when Cloudflare Access is not configured (e.g., local development).
      return NextResponse.next();
    }

    if (pathname.startsWith('/api/')) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    try {
      const redirectUrl = buildCloudflareLoginUrl(req.nextUrl.toString());
      return NextResponse.redirect(redirectUrl);
    } catch (error) {
      console.error('Failed to build Cloudflare Access login URL', error);
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
  }

  if (user) {
    const requestHeaders = new Headers(req.headers);
    requestHeaders.set('x-authenticated-user-email', user.email);
    if (user.name) {
      requestHeaders.set('x-authenticated-user-name', user.name);
    }
    return NextResponse.next({
      request: {
        headers: requestHeaders,
      },
    });
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
};
