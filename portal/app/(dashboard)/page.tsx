import Header from '@/components/Header'
import ServiceGrid from '@/components/ServiceGrid'
import {
  Activity,
  ArrowUpRight,
  BarChart2,
  CalendarClock,
  Flame,
  LineChart,
  Search,
  ShieldCheck,
  TrendingUp,
  Zap,
} from 'lucide-react'

const KPI_CARDS = [
  { label: 'Global Volume', value: '12.4B', delta: '+8.3% WoW', intent: 'positive' as const },
  { label: 'Daily Active Analysts', value: '318', delta: '+12.4% vs 30d', intent: 'positive' as const },
  { label: 'Latency (P95)', value: '212ms', delta: '-17% vs last week', intent: 'positive' as const },
]

const DASHBOARD_SHORTCUTS = [
  { title: 'Executive Market Pulse', description: 'Real-time macro indicators & liquidity trends', icon: BarChart2 },
  { title: 'Revenue & Bookings', description: 'Pipeline velocity, ARR risk, and forecast bridge', icon: LineChart },
  { title: 'Risk Exposure Monitor', description: 'Counterparty risk, stress tests, and alert ladder', icon: ShieldCheck },
]

const MARKET_MOVERS = [
  { name: 'Global Commodities', change: '+2.8%', note: 'Energy complex leads broad gains', trend: 'up' as const },
  { name: 'FX Volatility Index', change: '-1.4%', note: 'USD consolidates despite rate speculation', trend: 'down' as const },
  { name: 'Emerging Markets ETF', change: '+3.2%', note: 'Capital inflows accelerate in APAC', trend: 'up' as const },
]

const UPCOMING_RELEASES = [
  { title: 'US Non-Farm Payrolls', time: 'Fri • 08:30 EST', impact: 'High' },
  { title: 'FOMC Minutes', time: 'Wed • 14:00 EST', impact: 'Medium' },
  { title: 'EU CPI Flash', time: 'Fri • 11:00 CET', impact: 'Medium' },
]

const KNOWLEDGE_BASE = [
  { title: 'Building portfolio dashboards with Superset', href: '#', tag: 'Guide' },
  { title: 'Publishing Delta Lake metrics into DataHub', href: '#', tag: 'How-To' },
  { title: 'Best practices for orchestrating ML pipelines', href: '#', tag: 'Playbook' },
]

export default function Home() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <Header />

      <main className="pb-16">
        <section className="relative overflow-hidden border-b border-slate-900/60 bg-gradient-to-br from-slate-950 via-slate-950 to-slate-900">
          <div className="absolute inset-0 -z-10 bg-[radial-gradient(circle_at_top,_rgba(99,102,241,0.22),transparent_45%)]" />
          <div className="mx-auto max-w-7xl px-4 pb-16 pt-12 sm:px-6 lg:px-8 lg:pb-20 lg:pt-16">
            <div className="grid gap-10 lg:grid-cols-[1.25fr_0.9fr]">
              <div className="relative overflow-hidden rounded-3xl border border-slate-800 bg-slate-900/70 p-8 shadow-xl shadow-slate-950/60">
                <div className="absolute -right-32 -top-32 h-64 w-64 rounded-full bg-carbon/20 blur-3xl" />
                <div className="flex items-center gap-3 text-sm text-slate-400">
                  <Activity className="h-4 w-4 text-carbon" />
                  <span>Analytics Control Center</span>
                  <span className="text-slate-600">•</span>
                  <span>Market & Operations Intelligence</span>
                </div>
                <h1 className="mt-6 text-3xl font-semibold leading-tight text-white sm:text-4xl lg:text-5xl">
                  Navigate the 254Carbon market intelligence stack
                </h1>
                <p className="mt-4 max-w-2xl text-base text-slate-300 sm:text-lg">
                  Track liquidity shifts, monitor risk posture, and jump straight into the dashboards powering mission-critical decisions.
                </p>

                <div className="mt-6">
                  <div className="relative">
                    <Search className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                    <input
                      className="w-full rounded-2xl border border-slate-800 bg-slate-900/70 py-3 pl-12 pr-4 text-sm text-white outline-none ring-carbon/20 transition focus:border-carbon focus:ring-2"
                      placeholder="Search dashboards, datasets, or teams…"
                      type="search"
                    />
                  </div>
                  <div className="mt-3 flex flex-wrap gap-3 text-xs text-slate-400">
                    <span className="rounded-full bg-slate-900/80 px-3 py-1">Top queries: market snapshot, mlops pipeline, credit risk</span>
                  </div>
                </div>

                <div className="mt-8 grid gap-4 sm:grid-cols-3">
                  {KPI_CARDS.map(card => (
                    <div
                      key={card.label}
                      className="rounded-2xl border border-slate-800 bg-slate-900/90 p-4 shadow-inner shadow-slate-950/30"
                    >
                      <p className="text-xs uppercase tracking-wide text-slate-400">{card.label}</p>
                      <p className="mt-2 text-2xl font-semibold text-white">{card.value}</p>
                      <span
                        className={`mt-3 inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px] font-medium ${
                          card.intent === 'positive'
                            ? 'bg-emerald-500/10 text-emerald-300'
                            : 'bg-rose-500/10 text-rose-300'
                        }`}
                      >
                        <TrendingUp className="h-3 w-3" />
                        {card.delta}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-4">
                <div className="rounded-3xl border border-slate-800 bg-slate-900/60 p-6 shadow-lg shadow-slate-950/40">
                  <div className="flex items-center justify-between">
                    <div>
                      <h2 className="text-lg font-semibold text-white">Market pulse</h2>
                      <p className="text-xs text-slate-400">Real-time insights across asset classes</p>
                    </div>
                    <span className="inline-flex items-center gap-2 rounded-full bg-slate-900 px-3 py-1 text-xs text-slate-400">
                      <Zap className="h-3.5 w-3.5 text-carbon" />
                      Live feed
                    </span>
                  </div>

                  <div className="mt-6 space-y-4">
                    {MARKET_MOVERS.map(mover => (
                      <div
                        key={mover.name}
                        className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4"
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm font-semibold text-white">{mover.name}</p>
                            <p className="text-xs text-slate-400">{mover.note}</p>
                          </div>
                          <span
                            className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium ${
                              mover.trend === 'up'
                                ? 'bg-emerald-500/10 text-emerald-300'
                                : 'bg-rose-500/10 text-rose-300'
                            }`}
                          >
                            {mover.change}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="rounded-3xl border border-slate-800 bg-slate-900/60 p-6 shadow-lg shadow-slate-950/40">
                  <div className="flex items-center gap-3">
                    <Flame className="h-5 w-5 text-amber-300" />
                    <h3 className="text-sm font-semibold uppercase tracking-wide text-amber-200">Critical Alerts</h3>
                  </div>
                  <ul className="mt-4 space-y-3 text-sm text-slate-300">
                    <li className="flex items-start gap-3 rounded-2xl border border-amber-500/10 bg-amber-500/5 p-3">
                      <span className="mt-0.5 inline-flex h-2 w-2 rounded-full bg-amber-400" />
                      <div>
                        <p className="font-medium text-white">LakeFS commit latency elevated</p>
                        <p className="text-xs text-amber-200">Investigate load pattern on S3 gateway cluster</p>
                      </div>
                    </li>
                    <li className="flex items-start gap-3 rounded-2xl bg-slate-900/70 p-3">
                      <span className="mt-0.5 inline-flex h-2 w-2 rounded-full bg-emerald-400" />
                      <div>
                        <p className="font-medium text-white">DataHub ingestion SLA</p>
                        <p className="text-xs text-slate-400">Last ETL run completed in 26m • within target</p>
                      </div>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="insights" className="mx-auto max-w-7xl px-4 py-14 sm:px-6 lg:px-8">
          <div className="grid gap-10 lg:grid-cols-[1.25fr_0.9fr]">
            <div className="space-y-8">
              <div className="rounded-3xl border border-slate-800 bg-slate-900/50 p-6 shadow-lg shadow-slate-950/30">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-semibold text-white">Featured dashboards</h2>
                    <p className="text-sm text-slate-400">Jump straight into curated analytics views</p>
                  </div>
                  <button className="inline-flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900/80 px-4 py-2 text-xs font-medium text-slate-300 transition hover:border-carbon hover:text-white">
                    Browse all
                    <ArrowUpRight className="h-3.5 w-3.5" />
                  </button>
                </div>
                <div className="mt-6 grid gap-4 md:grid-cols-2">
                  {DASHBOARD_SHORTCUTS.map(shortcut => (
                    <div
                      key={shortcut.title}
                      className="group rounded-2xl border border-slate-800 bg-slate-900/70 p-5 transition hover:border-carbon/60 hover:bg-slate-900"
                    >
                      <shortcut.icon className="h-5 w-5 text-carbon" />
                      <h3 className="mt-4 text-base font-semibold text-white group-hover:text-carbon-light">
                        {shortcut.title}
                      </h3>
                      <p className="mt-2 text-sm text-slate-400">{shortcut.description}</p>
                      <span className="mt-4 inline-flex items-center gap-1 text-xs font-medium text-carbon">
                        Open dashboard
                        <ArrowUpRight className="h-3 w-3" />
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-3xl border border-slate-800 bg-slate-900/50 p-6 shadow-lg shadow-slate-950/30">
                <div className="flex items-center gap-3">
                  <CalendarClock className="h-5 w-5 text-slate-200" />
                  <div>
                    <h2 className="text-xl font-semibold text-white">Upcoming market releases</h2>
                    <p className="text-sm text-slate-400">Plan your research & trading coverage</p>
                  </div>
                </div>
                <ul className="mt-5 space-y-4">
                  {UPCOMING_RELEASES.map(release => (
                    <li
                      key={release.title}
                      className="flex items-center justify-between rounded-2xl bg-slate-900/70 p-4"
                    >
                      <div>
                        <p className="text-sm font-medium text-white">{release.title}</p>
                        <p className="text-xs text-slate-400">{release.time}</p>
                      </div>
                      <span className="inline-flex items-center rounded-full bg-slate-800 px-3 py-1 text-xs font-medium text-slate-300">
                        {release.impact} impact
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <aside className="space-y-8">
              <div className="rounded-3xl border border-slate-800 bg-slate-900/50 p-6 shadow-lg shadow-slate-950/30">
                <div className="flex items-center gap-3">
                  <Zap className="h-5 w-5 text-carbon" />
                  <h3 className="text-lg font-semibold text-white">Operational shortcuts</h3>
                </div>
                <ul className="mt-4 space-y-3 text-sm text-slate-300">
                  <li className="flex items-center justify-between rounded-2xl bg-slate-900/70 p-4">
                    <span>Run Spark quality checks</span>
                    <ArrowUpRight className="h-4 w-4 text-slate-500" />
                  </li>
                  <li className="flex items-center justify-between rounded-2xl bg-slate-900/70 p-4">
                    <span>Review MLFlow model registry</span>
                    <ArrowUpRight className="h-4 w-4 text-slate-500" />
                  </li>
                  <li className="flex items-center justify-between rounded-2xl bg-slate-900/70 p-4">
                    <span>Audit access policies in Vault</span>
                    <ArrowUpRight className="h-4 w-4 text-slate-500" />
                  </li>
                </ul>
              </div>

              <div className="rounded-3xl border border-slate-800 bg-slate-900/50 p-6 shadow-lg shadow-slate-950/30">
                <h3 className="text-lg font-semibold text-white">Knowledge base</h3>
                <p className="mt-1 text-sm text-slate-400">Guides to accelerate analytics delivery</p>
                <ul className="mt-4 space-y-3 text-sm">
                  {KNOWLEDGE_BASE.map(item => (
                    <li key={item.title}>
                      <a
                        className="flex items-center justify-between rounded-2xl border border-slate-800 bg-slate-900/70 px-4 py-3 text-slate-300 transition hover:border-carbon hover:text-white"
                        href={item.href}
                      >
                        <div>
                          <p className="text-sm font-medium">{item.title}</p>
                          <p className="text-xs uppercase tracking-wide text-slate-500">{item.tag}</p>
                        </div>
                        <ArrowUpRight className="h-3.5 w-3.5" />
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            </aside>
          </div>
        </section>

        <section className="border-y border-slate-900/60 bg-slate-950/80 py-16">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <h2 className="text-2xl font-semibold text-white">Explore platform services</h2>
                <p className="text-sm text-slate-400">
                  Launch analytics applications, storage consoles, and orchestration tools with one click.
                </p>
              </div>
              <button className="inline-flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900/60 px-4 py-2 text-xs font-medium text-slate-300 transition hover:border-carbon hover:text-white">
                View architecture map
                <ArrowUpRight className="h-3.5 w-3.5" />
              </button>
            </div>
            <div className="mt-10">
              <ServiceGrid />
            </div>
          </div>
        </section>

        <section id="docs" className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
          <div className="rounded-3xl border border-slate-800 bg-slate-900/50 p-8 shadow-xl shadow-slate-950/30">
            <div className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
              <div>
                <h2 className="text-2xl font-semibold text-white">Documentation & runbooks</h2>
                <p className="mt-3 text-sm text-slate-400">
                  Everything you need to deliver analytics and market insights—conventions, runbooks, and quick-reference guides.
                </p>
                <ol className="mt-6 space-y-4 text-sm text-slate-300">
                  <li className="flex gap-3 rounded-2xl bg-slate-900/70 p-4">
                    <span className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-carbon text-xs font-semibold text-white">
                      1
                    </span>
                    <span>Follow the analytics onboarding checklist & configure SSO access for your team.</span>
                  </li>
                  <li className="flex gap-3 rounded-2xl bg-slate-900/70 p-4">
                    <span className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-carbon text-xs font-semibold text-white">
                      2
                    </span>
                    <span>Deploy sample dashboards or ML pipelines using the curated Superset and Spark templates.</span>
                  </li>
                  <li className="flex gap-3 rounded-2xl bg-slate-900/70 p-4">
                    <span className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-carbon text-xs font-semibold text-white">
                      3
                    </span>
                    <span>Review runbooks for incident response and resilience testing to keep insights flowing.</span>
                  </li>
                </ol>
              </div>

              <div className="flex flex-col justify-between gap-6">
                <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
                  <p className="text-xs uppercase tracking-wide text-slate-500">Featured manual</p>
                  <h3 className="mt-2 text-lg font-semibold text-white">Cloudflare Access stabilization playbook</h3>
                  <p className="mt-2 text-sm text-slate-400">
                    Reference architecture, failover design, and runbooks for perimeter security.
                  </p>
                  <button className="mt-4 inline-flex items-center gap-2 text-sm font-medium text-carbon hover:text-carbon-light">
                    Open guide
                    <ArrowUpRight className="h-3.5 w-3.5" />
                  </button>
                </div>
                <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
                  <p className="text-xs uppercase tracking-wide text-slate-500">Popular resource</p>
                  <h3 className="mt-2 text-lg font-semibold text-white">Fresh cluster deployment roadmap</h3>
                  <p className="mt-2 text-sm text-slate-400">
                    Step-by-step activities for greenfield environments, including observability and data lake rollout.
                  </p>
                  <button className="mt-4 inline-flex items-center gap-2 text-sm font-medium text-carbon hover:text-carbon-light">
                    View checklist
                    <ArrowUpRight className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t border-slate-900/60 bg-slate-950/80">
        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-4 py-8 text-center text-xs text-slate-500 sm:flex-row sm:text-left">
          <p>254Carbon Market Intelligence Portal</p>
          <p>Powered by Next.js • Cloudflare Access • Superset • DataHub</p>
        </div>
      </footer>
    </div>
  )
}
