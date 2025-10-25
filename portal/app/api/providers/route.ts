import { NextRequest, NextResponse } from 'next/server';

const API_BASE = process.env.API_URL || 'http://localhost:3001';

export async function GET(req: NextRequest) {
  try {
    // Get query parameters for filtering/pagination
    const { searchParams } = new URL(req.url);
    const status = searchParams.get('status');
    const limit = searchParams.get('limit') || '50';
    const offset = searchParams.get('offset') || '0';

    const query = new URLSearchParams();
    if (status) query.append('status', status);
    query.append('limit', limit);
    query.append('offset', offset);

    const response = await fetch(`${API_BASE}/api/providers?${query}`, {
      headers: {
        'Authorization': req.headers.get('authorization') || '',
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: 'Failed to fetch providers' },
        { status: response.status }
      );
    }

    const data = await response.json();
    const providers = Array.isArray(data.providers)
      ? data.providers.map((p: any) => ({
          id: p.id,
          name: p.name,
          type: p.type,
          status: p.status,
          lastRunAt: p.last_run_at ?? null,
          nextRunAt: p.next_run_at ?? null,
          totalRuns: p.total_runs ?? 0,
          successRate: p.success_rate ?? 100,
        }))
      : [];
    return NextResponse.json({ providers, total: data.total ?? providers.length });
  } catch (error) {
    console.error('Provider list error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    // Validate required fields
    if (!body.name || !body.type) {
      return NextResponse.json(
        { error: 'Missing required fields: name, type' },
        { status: 400 }
      );
    }

    const response = await fetch(`${API_BASE}/api/providers`, {
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
    console.error('Provider creation error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
