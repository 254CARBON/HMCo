import { NextRequest, NextResponse } from 'next/server';
import {
  createSessionToken,
  getSessionCookieName,
  getSessionCookieOptions,
} from '@/lib/auth/session';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const username = (body.username || '').trim();
    const password = body.password || '';

    if (!username || !password) {
      return NextResponse.json(
        { error: 'Username and password are required' },
        { status: 400 }
      );
    }

    const expectedUsername = process.env.PORTAL_USERNAME || 'admin';
    const expectedPassword =
      process.env.PORTAL_PASSWORD || process.env.PORTAL_SECRET || 'admin123';

    if (username !== expectedUsername || password !== expectedPassword) {
      return NextResponse.json(
        { error: 'Invalid credentials' },
        { status: 401 }
      );
    }

    const sessionToken = createSessionToken(username);
    const response = NextResponse.json({ user: { username } });

    response.cookies.set({
      name: getSessionCookieName(),
      value: sessionToken,
      ...getSessionCookieOptions(),
    });

    return response;
  } catch (error) {
    console.error('Login error:', error);
    return NextResponse.json(
      { error: 'Unable to process login' },
      { status: 500 }
    );
  }
}
