import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: '254Carbon - Data Platform Portal',
  description: 'Central landing portal and SSO dashboard for 254carbon cluster',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-slate-100 antialiased">
        {children}
      </body>
    </html>
  )
}
