import { NextRequest, NextResponse } from 'next/server';

/**
 * API route for scenario simulation
 * Proxies requests to congestion-sim service
 */

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // In production, proxy to congestion-sim service
    const congestionSimUrl = process.env.CONGESTION_SIM_URL || 'http://localhost:8001';
    
    const response = await fetch(`${congestionSimUrl}/simulate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    
    if (!response.ok) {
      throw new Error(`Simulation failed: ${response.statusText}`);
    }
    
    const data = await response.json();
    
    return NextResponse.json(data);
    
  } catch (error: any) {
    console.error('Scenario simulation error:', error);
    return NextResponse.json(
      { error: error.message || 'Simulation failed' },
      { status: 500 }
    );
  }
}
