/**
 * Attribution Dashboard Layout
 */

import React from 'react';

export default function AttributionLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b border-gray-200 mb-6">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center space-x-4">
            <a href="/copilot" className="text-blue-600 hover:text-blue-800">
              ← Back to Copilot
            </a>
            <span className="text-gray-300">|</span>
            <span className="font-semibold">Attribution</span>
          </div>
        </div>
      </nav>
      {children}
    </div>
  );
}
