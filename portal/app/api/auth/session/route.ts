import { NextRequest, NextResponse } from 'next/server';
import { getUserFromRequest } from '@/lib/auth/cloudflare';

export async function GET(req: NextRequest) {
  const user = await getUserFromRequest(req);

  if (!user) {
    return NextResponse.json({ user: null }, { status: 401 });
  }

  return NextResponse.json({
    user,
  });
}
