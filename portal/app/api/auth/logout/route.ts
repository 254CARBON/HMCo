import { NextRequest, NextResponse } from 'next/server';
import {
  buildCloudflareLogoutUrl,
  isCloudflareAccessEnabled,
} from '@/lib/auth/cloudflare';

export async function GET(req: NextRequest) {
  const returnTo =
    req.nextUrl.searchParams.get('returnTo') ??
    req.headers.get('referer') ??
    req.nextUrl.origin;

  if (!isCloudflareAccessEnabled()) {
    return NextResponse.redirect(returnTo);
  }

  try {
    const logoutUrl = buildCloudflareLogoutUrl(returnTo);
    return NextResponse.redirect(logoutUrl);
  } catch (error) {
    console.error('Failed to construct Cloudflare Access logout URL', error);
    return NextResponse.json(
      { error: 'Unable to perform Cloudflare Access logout' },
      { status: 500 }
    );
  }
}

export async function POST(req: NextRequest) {
  return GET(req);
}
