import Navigation from '@/components/Navigation';

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-screen flex-col bg-slate-950 text-slate-100">
      <Navigation />
      <main className="flex-1 pb-16">{children}</main>
    </div>
  );
}
