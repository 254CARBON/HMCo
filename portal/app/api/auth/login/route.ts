import { NextRequest, NextResponse } from 'next/server';
import {
  buildCloudflareLoginUrl,
  isCloudflareAccessEnabled,
} from '@/lib/auth/cloudflare';

export async function GET(req: NextRequest) {
  const nextUrl =
    req.nextUrl.searchParams.get('next') ??
    req.headers.get('referer') ??
    req.nextUrl.origin;

  if (!isCloudflareAccessEnabled()) {
    return NextResponse.redirect(nextUrl);
  }

  try {
    const loginUrl = buildCloudflareLoginUrl(nextUrl);
    return NextResponse.redirect(loginUrl);
  } catch (error) {
    console.error('Failed to construct Cloudflare Access login URL', error);
    return NextResponse.json(
      { error: 'Cloudflare Access login unavailable' },
      { status: 500 }
    );
  }
}

export async function POST(req: NextRequest) {
  return GET(req);
}
