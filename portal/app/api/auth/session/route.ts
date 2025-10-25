import { NextRequest, NextResponse } from 'next/server';
import {
  verifySessionToken,
  getSessionCookieName,
} from '@/lib/auth/session';

export async function GET(req: NextRequest) {
  const cookie = req.cookies.get(getSessionCookieName());
  const session = verifySessionToken(cookie?.value);

  if (!session) {
    return NextResponse.json({ user: null }, { status: 401 });
  }

  return NextResponse.json({
    user: {
      username: session.username,
      expiresAt: session.expiresAt,
    },
  });
}
