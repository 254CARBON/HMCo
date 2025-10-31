'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';

/**
 * Constraint & Outage Impact Simulator UI
 * 
 * Features:
 * - What-if scenario builder
 * - Outage/derate configuration
 * - LMP delta visualization with confidence bands
 * - Target: <2s scenario latency
 */

interface Outage {
  id: string;
  fromNode: string;
  toNode: string;
  startTime: string;
  duration: number;
}

interface ScenarioResult {
  runId: string;
  lmpDeltas: Record<string, number>;
  confidence: Record<string, number>;
  computeTimeMs: number;
}

export default function ScenariosPage() {
  const [iso, setIso] = useState('CAISO');
  const [outages, setOutages] = useState<Outage[]>([]);
  const [results, setResults] = useState<ScenarioResult | null>(null);
  const [loading, setLoading] = useState(false);

  const addOutage = () => {
    const newOutage: Outage = {
      id: `outage-${Date.now()}`,
      fromNode: '',
      toNode: '',
      startTime: new Date().toISOString(),
      duration: 60, // minutes
    };
    setOutages([...outages, newOutage]);
  };

  const removeOutage = (id: string) => {
    setOutages(outages.filter(o => o.id !== id));
  };

  const runScenario = async () => {
    setLoading(true);
    
    try {
      const response = await fetch('/api/scenarios/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          iso,
          outages: outages.map(o => ({
            from_node: o.fromNode,
            to_node: o.toNode,
            start_time: o.startTime,
            duration_minutes: o.duration,
          })),
        }),
      });

      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Scenario simulation failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">Constraint & Outage Simulator</h1>
        <div className="flex gap-2">
          <Button onClick={addOutage}>Add Outage</Button>
          <Button 
            onClick={runScenario} 
            disabled={loading || outages.length === 0}
            variant="default"
          >
            {loading ? 'Running...' : 'Run Scenario'}
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Scenario Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">ISO</label>
            <select 
              value={iso} 
              onChange={(e) => setIso(e.target.value)}
              className="w-full p-2 border rounded"
            >
              <option value="CAISO">CAISO</option>
              <option value="MISO">MISO</option>
              <option value="SPP">SPP</option>
              <option value="ERCOT">ERCOT</option>
              <option value="PJM">PJM</option>
            </select>
          </div>

          <div className="space-y-3">
            <h3 className="text-lg font-semibold">Outages</h3>
            {outages.length === 0 ? (
              <p className="text-gray-500 italic">No outages configured. Click "Add Outage" to start.</p>
            ) : (
              outages.map((outage) => (
                <div key={outage.id} className="border rounded p-4 space-y-2">
                  <div className="flex justify-between items-start">
                    <div className="flex-1 grid grid-cols-2 gap-3">
                      <div>
                        <label className="block text-sm text-gray-600 mb-1">From Node</label>
                        <input
                          type="text"
                          value={outage.fromNode}
                          onChange={(e) => {
                            const updated = outages.map(o =>
                              o.id === outage.id ? { ...o, fromNode: e.target.value } : o
                            );
                            setOutages(updated);
                          }}
                          className="w-full p-2 border rounded"
                          placeholder="NODE_0001"
                        />
                      </div>
                      <div>
                        <label className="block text-sm text-gray-600 mb-1">To Node</label>
                        <input
                          type="text"
                          value={outage.toNode}
                          onChange={(e) => {
                            const updated = outages.map(o =>
                              o.id === outage.id ? { ...o, toNode: e.target.value } : o
                            );
                            setOutages(updated);
                          }}
                          className="w-full p-2 border rounded"
                          placeholder="NODE_0002"
                        />
                      </div>
                      <div>
                        <label className="block text-sm text-gray-600 mb-1">Start Time</label>
                        <input
                          type="datetime-local"
                          value={outage.startTime.slice(0, 16)}
                          onChange={(e) => {
                            const updated = outages.map(o =>
                              o.id === outage.id ? { ...o, startTime: new Date(e.target.value).toISOString() } : o
                            );
                            setOutages(updated);
                          }}
                          className="w-full p-2 border rounded"
                        />
                      </div>
                      <div>
                        <label className="block text-sm text-gray-600 mb-1">Duration (minutes)</label>
                        <input
                          type="number"
                          value={outage.duration}
                          onChange={(e) => {
                            const updated = outages.map(o =>
                              o.id === outage.id ? { ...o, duration: parseInt(e.target.value) } : o
                            );
                            setOutages(updated);
                          }}
                          className="w-full p-2 border rounded"
                        />
                      </div>
                    </div>
                    <Button
                      onClick={() => removeOutage(outage.id)}
                      variant="destructive"
                      size="sm"
                      className="ml-2"
                    >
                      Remove
                    </Button>
                  </div>
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>

      {results && (
        <Card>
          <CardHeader>
            <CardTitle>Scenario Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-4">
                <div className="p-4 bg-blue-50 rounded">
                  <div className="text-sm text-gray-600">Run ID</div>
                  <div className="text-lg font-semibold">{results.runId}</div>
                </div>
                <div className="p-4 bg-green-50 rounded">
                  <div className="text-sm text-gray-600">Compute Time</div>
                  <div className="text-lg font-semibold">{results.computeTimeMs}ms</div>
                  {results.computeTimeMs > 2000 && (
                    <div className="text-xs text-red-600 mt-1">⚠️ Exceeds 2s target</div>
                  )}
                </div>
                <div className="p-4 bg-purple-50 rounded">
                  <div className="text-sm text-gray-600">Affected Nodes</div>
                  <div className="text-lg font-semibold">
                    {Object.keys(results.lmpDeltas).length}
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">LMP Deltas (Top 10)</h3>
                <div className="border rounded overflow-hidden">
                  <table className="w-full">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-2 text-left">Node</th>
                        <th className="px-4 py-2 text-right">Delta ($/MWh)</th>
                        <th className="px-4 py-2 text-right">Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(results.lmpDeltas)
                        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                        .slice(0, 10)
                        .map(([node, delta]) => (
                          <tr key={node} className="border-t">
                            <td className="px-4 py-2">{node}</td>
                            <td className={`px-4 py-2 text-right font-mono ${
                              delta > 0 ? 'text-red-600' : 'text-green-600'
                            }`}>
                              {delta > 0 ? '+' : ''}{delta.toFixed(2)}
                            </td>
                            <td className="px-4 py-2 text-right">
                              {((results.confidence[node] || 0) * 100).toFixed(1)}%
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
