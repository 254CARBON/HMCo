import { NextRequest, NextResponse } from 'next/server';

const API_BASE = process.env.API_URL || 'http://localhost:3001';

export async function GET(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const response = await fetch(`${API_BASE}/api/providers/${params.id}`, {
      headers: {
        'Authorization': req.headers.get('authorization') || '',
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: 'Provider not found' },
        { status: response.status }
      );
    }

    const p = await response.json();
    const provider = {
      id: p.id,
      name: p.name,
      type: p.type,
      status: p.status,
      lastRunAt: p.last_run_at ?? null,
      nextRunAt: p.next_run_at ?? null,
      totalRuns: p.total_runs ?? 0,
      successRate: p.success_rate ?? 100,
      uis: p.uis,
      config: p.config,
      schedule: p.schedule,
      createdAt: p.created_at,
      updatedAt: p.updated_at,
    };
    return NextResponse.json(provider);
  } catch (error) {
    console.error('Provider detail error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function PATCH(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const body = await req.json();

    const response = await fetch(`${API_BASE}/api/providers/${params.id}`, {
      method: 'PATCH',
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
    return NextResponse.json(data);
  } catch (error) {
    console.error('Provider update error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function DELETE(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const response = await fetch(`${API_BASE}/api/providers/${params.id}`, {
      method: 'DELETE',
      headers: {
        'Authorization': req.headers.get('authorization') || '',
      },
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: 'Failed to delete provider' },
        { status: response.status }
      );
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Provider deletion error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
