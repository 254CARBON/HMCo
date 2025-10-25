import { NextRequest, NextResponse } from 'next/server';

const API_BASE = process.env.API_URL || 'http://localhost:3001';

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const providerId = searchParams.get('providerId');
    const status = searchParams.get('status');
    const limit = searchParams.get('limit') || '50';
    const offset = searchParams.get('offset') || '0';
    const sortBy = searchParams.get('sortBy') || 'createdAt';
    const sortOrder = searchParams.get('sortOrder') || 'desc';

    const query = new URLSearchParams();
    if (providerId) query.append('providerId', providerId);
    if (status) query.append('status', status);
    query.append('limit', limit);
    query.append('offset', offset);
    query.append('sortBy', sortBy);
    query.append('sortOrder', sortOrder);

    const response = await fetch(`${API_BASE}/api/runs?${query}`, {
      headers: {
        'Authorization': req.headers.get('authorization') || '',
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: 'Failed to fetch runs' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Runs list error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    if (!body.providerId) {
      return NextResponse.json(
        { error: 'Missing required field: providerId' },
        { status: 400 }
      );
    }

    const response = await fetch(`${API_BASE}/api/runs`, {
      method: 'POST',
      headers: {
        'Authorization': req.headers.get('authorization') || '',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(error, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 201 });
  } catch (error) {
    console.error('Run creation error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
