'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

/**
 * Trader Copilot: Scenario → Decision Loop
 * 
 * Orchestrates:
 * 1. Scenario definition
 * 2. Forecast generation
 * 3. Strategy optimization
 * 4. Risk analysis
 * 5. Decision recommendations
 * 
 * Target: <60s from scenario to decision plan
 */

interface CopilotState {
  status: 'idle' | 'running' | 'complete' | 'error';
  stage: string;
  progress: number;
  startTime: number | null;
  elapsedSeconds: number;
}

interface DecisionPlan {
  actions: Array<{
    type: 'bid' | 'hedge' | 'hold';
    hub: string;
    quantity: number;
    price: number;
    confidence: number;
  }>;
  riskMetrics: {
    expectedPnL: number;
    var95: number;
    sharpe: number;
    maxDrawdown: number;
  };
  drivers: Array<{
    factor: string;
    impact: number;
    shapValue: number;
  }>;
}

export default function CopilotPage() {
  const [copilotState, setCopilotState] = useState<CopilotState>({
    status: 'idle',
    stage: '',
    progress: 0,
    startTime: null,
    elapsedSeconds: 0,
  });

  const [scenarioInput, setScenarioInput] = useState({
    description: '',
    hubs: ['HUB_01', 'HUB_02', 'HUB_03', 'HUB_04', 'HUB_05'],
    timeHorizon: '4h',
    riskBudget: 100000,
  });

  const [decisionPlan, setDecisionPlan] = useState<DecisionPlan | null>(null);

  const runCopilot = async () => {
    const startTime = Date.now();
    
    setCopilotState({
      status: 'running',
      stage: 'Initializing...',
      progress: 0,
      startTime,
      elapsedSeconds: 0,
    });

    try {
      // Stage 1: Compile scenario
      await updateProgress('Compiling scenario...', 10, startTime);
      await new Promise(r => setTimeout(r, 2000));

      // Stage 2: Generate forecasts
      await updateProgress('Generating LMP forecasts...', 30, startTime);
      await new Promise(r => setTimeout(r, 5000));

      // Stage 3: Run strategy optimizer
      await updateProgress('Optimizing strategy...', 60, startTime);
      await new Promise(r => setTimeout(r, 8000));

      // Stage 4: Compute P&L distribution
      await updateProgress('Computing P&L distribution...', 80, startTime);
      await new Promise(r => setTimeout(r, 3000));

      // Stage 5: Extract drivers (SHAP)
      await updateProgress('Analyzing key drivers...', 95, startTime);
      await new Promise(r => setTimeout(r, 2000));

      // Generate decision plan
      const plan = generateDecisionPlan();
      setDecisionPlan(plan);

      const elapsedSeconds = (Date.now() - startTime) / 1000;

      setCopilotState({
        status: 'complete',
        stage: 'Complete',
        progress: 100,
        startTime,
        elapsedSeconds,
      });

    } catch (error) {
      setCopilotState(prev => ({
        ...prev,
        status: 'error',
        stage: 'Error occurred',
      }));
      console.error('Copilot error:', error);
    }
  };

  const updateProgress = async (stage: string, progress: number, startTime: number) => {
    const elapsedSeconds = (Date.now() - startTime) / 1000;
    setCopilotState(prev => ({
      ...prev,
      stage,
      progress,
      elapsedSeconds,
    }));
  };

  const generateDecisionPlan = (): DecisionPlan => {
    // Mock decision plan generation
    return {
      actions: [
        { type: 'bid', hub: 'HUB_01', quantity: 50, price: 42.5, confidence: 0.85 },
        { type: 'hedge', hub: 'HUB_02', quantity: -30, price: 38.2, confidence: 0.78 },
        { type: 'bid', hub: 'HUB_03', quantity: 25, price: 45.0, confidence: 0.82 },
        { type: 'hold', hub: 'HUB_04', quantity: 0, price: 0, confidence: 0.90 },
        { type: 'hedge', hub: 'HUB_05', quantity: -15, price: 40.5, confidence: 0.75 },
      ],
      riskMetrics: {
        expectedPnL: 12500,
        var95: -8500,
        sharpe: 1.85,
        maxDrawdown: 0.12,
      },
      drivers: [
        { factor: 'Congestion on PATH_A', impact: 0.35, shapValue: 4200 },
        { factor: 'Wind generation forecast', impact: 0.28, shapValue: 3500 },
        { factor: 'Gas price movement', impact: 0.22, shapValue: 2750 },
        { factor: 'Load forecast', impact: 0.15, shapValue: 1875 },
      ],
    };
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">Trader Copilot</h1>
        <div className="text-sm text-gray-500">
          Scenario → Decision in &lt;60s
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Scenario Definition</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">What-If Scenario</label>
            <textarea
              value={scenarioInput.description}
              onChange={(e) => setScenarioInput({ ...scenarioInput, description: e.target.value })}
              className="w-full p-3 border rounded min-h-[100px]"
              placeholder="Example: Outage on LINE_X @ 16:00, duration 2 hours, with increased load forecast..."
            />
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Hubs</label>
              <div className="text-sm text-gray-600">
                {scenarioInput.hubs.join(', ')}
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Time Horizon</label>
              <select
                value={scenarioInput.timeHorizon}
                onChange={(e) => setScenarioInput({ ...scenarioInput, timeHorizon: e.target.value })}
                className="w-full p-2 border rounded"
              >
                <option value="1h">1 Hour</option>
                <option value="4h">4 Hours</option>
                <option value="1d">1 Day</option>
                <option value="1w">1 Week</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Risk Budget ($)</label>
              <input
                type="number"
                value={scenarioInput.riskBudget}
                onChange={(e) => setScenarioInput({ ...scenarioInput, riskBudget: parseInt(e.target.value) })}
                className="w-full p-2 border rounded"
              />
            </div>
          </div>

          <Button
            onClick={runCopilot}
            disabled={copilotState.status === 'running' || !scenarioInput.description}
            className="w-full"
            size="lg"
          >
            {copilotState.status === 'running' ? 'Running Copilot...' : 'Run Copilot Analysis'}
          </Button>
        </CardContent>
      </Card>

      {copilotState.status !== 'idle' && (
        <Card>
          <CardHeader>
            <CardTitle>Analysis Progress</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span>{copilotState.stage}</span>
                <span>{copilotState.progress}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${copilotState.progress}%` }}
                />
              </div>
            </div>

            <div className="flex justify-between text-sm">
              <span className="text-gray-600">Elapsed Time</span>
              <span className="font-mono">
                {copilotState.elapsedSeconds.toFixed(1)}s
                {copilotState.elapsedSeconds > 60 && (
                  <span className="text-red-600 ml-2">⚠️ Exceeds 60s target</span>
                )}
              </span>
            </div>
          </CardContent>
        </Card>
      )}

      {decisionPlan && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Recommended Actions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {decisionPlan.actions.map((action, idx) => (
                  <div key={idx} className="border rounded p-4 flex justify-between items-center">
                    <div className="flex items-center gap-4">
                      <div className={`px-3 py-1 rounded text-sm font-medium ${
                        action.type === 'bid' ? 'bg-green-100 text-green-800' :
                        action.type === 'hedge' ? 'bg-blue-100 text-blue-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {action.type.toUpperCase()}
                      </div>
                      <div>
                        <div className="font-medium">{action.hub}</div>
                        <div className="text-sm text-gray-600">
                          {action.quantity !== 0 ? `${action.quantity > 0 ? '+' : ''}${action.quantity} MW @ $${action.price.toFixed(2)}` : 'No action'}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-gray-600">Confidence</div>
                      <div className="text-lg font-semibold">
                        {(action.confidence * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Risk Metrics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Expected P&L</span>
                  <span className="font-semibold text-green-600">
                    ${decisionPlan.riskMetrics.expectedPnL.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">VaR (95%)</span>
                  <span className="font-semibold text-red-600">
                    ${decisionPlan.riskMetrics.var95.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Sharpe Ratio</span>
                  <span className="font-semibold">
                    {decisionPlan.riskMetrics.sharpe.toFixed(2)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Max Drawdown</span>
                  <span className="font-semibold">
                    {(decisionPlan.riskMetrics.maxDrawdown * 100).toFixed(1)}%
                  </span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Key Drivers (SHAP)</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {decisionPlan.drivers.map((driver, idx) => (
                  <div key={idx}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-700">{driver.factor}</span>
                      <span className="font-mono">${driver.shapValue.toLocaleString()}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-purple-600 h-2 rounded-full"
                        style={{ width: `${driver.impact * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>

          <Card className="border-2 border-blue-500">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-sm text-gray-600">Audit Trail</div>
                  <div className="text-lg font-semibold">
                    Ready for approval and execution
                  </div>
                </div>
                <div className="flex gap-2">
                  <Button variant="outline">Export Report</Button>
                  <Button variant="default">Submit for Approval</Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
