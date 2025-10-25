import { NextRequest, NextResponse } from 'next/server';
import {
  verifySessionToken,
  getSessionCookieName,
} from '@/lib/auth/session';

const PUBLIC_PATHS = new Set([
  '/login',
  '/api/auth/login',
  '/api/auth/logout',
  '/api/auth/session',
  '/_next/static',
  '/_next/image',
  '/favicon.ico',
]);

function isPublicPath(pathname: string) {
  if (pathname === '/') return false;
  if (PUBLIC_PATHS.has(pathname)) return true;
  return Array.from(PUBLIC_PATHS).some(
    (path) => path !== '/' && pathname.startsWith(`${path}/`)
  );
}

export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  if (
    pathname.startsWith('/_next/') ||
    pathname.startsWith('/static/') ||
    pathname.match(/\.(png|jpg|jpeg|gif|webp|svg|ico)$/i)
  ) {
    return NextResponse.next();
  }

  const sessionCookie = req.cookies.get(getSessionCookieName());
  const session = verifySessionToken(sessionCookie?.value);
  const isPublic = isPublicPath(pathname);

  if (session && pathname === '/login') {
    const url = req.nextUrl.clone();
    url.pathname = '/';
    url.searchParams.delete('next');
    return NextResponse.redirect(url);
  }

  if (!session && !isPublic) {
    if (pathname.startsWith('/api/')) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
    const loginUrl = req.nextUrl.clone();
    loginUrl.pathname = '/login';
    loginUrl.searchParams.set('next', pathname);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
};
