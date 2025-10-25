import { NextResponse } from 'next/server';
import { getSessionCookieName } from '@/lib/auth/session';

export async function POST() {
  const response = NextResponse.json({ success: true });

  response.cookies.set({
    name: getSessionCookieName(),
    value: '',
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production' ? true : false,
    sameSite: 'lax',
    path: '/',
    maxAge: 0,
  });

  return response;
}
