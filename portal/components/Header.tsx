'use client'

import { Menu, LogOut, User } from 'lucide-react'
import { useState } from 'react'

export default function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <header className="sticky top-0 z-50 border-b border-slate-800 bg-slate-950/80 backdrop-blur">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16 text-slate-100">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-carbon rounded-xl flex items-center justify-center shadow-lg shadow-carbon/30">
              <span className="text-white font-bold text-lg">254</span>
            </div>
            <div>
              <h1 className="text-lg font-semibold">254Carbon</h1>
              <p className="text-xs text-slate-400 uppercase tracking-wide">Analytics Hub</p>
            </div>
          </div>

          <div className="hidden md:flex items-center space-x-8">
            <a href="#insights" className="text-sm font-medium text-slate-300 hover:text-white transition-colors">
              Insights
            </a>
            <a href="#services" className="text-sm font-medium text-slate-300 hover:text-white transition-colors">
              Services
            </a>
            <a href="#docs" className="text-sm font-medium text-slate-300 hover:text-white transition-colors">
              Documentation
            </a>
            <button className="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-slate-300 hover:text-red-400 transition-colors">
              <LogOut className="w-4 h-4" />
              <span>Sign out</span>
            </button>
            <div className="flex items-center justify-center w-10 h-10 rounded-full bg-slate-800 border border-slate-700">
              <User className="w-4 h-4 text-slate-300" />
            </div>
          </div>

          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 hover:bg-slate-800 rounded-lg transition-colors text-slate-300"
          >
            <Menu className="w-6 h-6" />
          </button>
        </div>

        {mobileMenuOpen && (
          <div className="md:hidden pb-4 space-y-2 text-slate-200">
            <a href="#insights" className="block px-4 py-2 hover:bg-slate-900 rounded-lg transition-colors">
              Insights
            </a>
            <a href="#services" className="block px-4 py-2 hover:bg-slate-900 rounded-lg transition-colors">
              Services
            </a>
            <a href="#docs" className="block px-4 py-2 hover:bg-slate-900 rounded-lg transition-colors">
              Documentation
            </a>
            <button className="w-full text-left px-4 py-2 hover:bg-slate-900 rounded-lg transition-colors flex items-center space-x-2">
              <LogOut className="w-4 h-4" />
              <span>Sign Out</span>
            </button>
          </div>
        )}
      </nav>
    </header>
  )
}
