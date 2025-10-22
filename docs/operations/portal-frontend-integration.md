# Portal Frontend Integration: Service Directory

The portal frontend should query the API mounted at `/api/services` on the same host. Example in React/Next.js.

Environment
- `NEXT_PUBLIC_SERVICES_API_BASE` → `/api/services`

Fetch services
```ts
// lib/services.ts
export type Service = { id: string; name: string; url: string; category?: string; description?: string; icon?: string };
export type ServiceStatus = { id: string; status: 'ok'|'warn'|'err'|'unknown'; code?: number; latencyMs?: number; checkedAt?: string };

const base = process.env.NEXT_PUBLIC_SERVICES_API_BASE || '/api/services';

export async function listServices(): Promise<Service[]> {
  const res = await fetch(`${base}`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to load services');
  return res.json();
}

export async function getStatuses(mode: 'auto'|'internal'|'external' = 'auto'): Promise<ServiceStatus[]> {
  const res = await fetch(`${base}/status?mode=${mode}`, { cache: 'no-store' });
  if (!res.ok) return [];
  return res.json();
}
```

UI tiles
```tsx
// app/services/page.tsx (Next.js App Router)
import { listServices } from '@/lib/services';

export default async function ServicesPage() {
  const services = await listServices();
  return (
    <main className="px-4 py-6 grid gap-3 grid-cols-1 md:grid-cols-2 xl:grid-cols-3">
      {services.map(s => (
        <article key={s.id} className="rounded-lg border border-slate-700 bg-slate-900 p-3">
          <header className="flex items-center justify-between">
            <h3 className="font-semibold flex items-center gap-2"><span>{s.icon ?? '🔗'}</span> {s.name}</h3>
            <span data-status={s.id} className="text-xs rounded-full border px-2 py-0.5">Unknown</span>
          </header>
          <p className="text-slate-400 text-sm mt-1 min-h-[1.5rem]">{s.description}</p>
          <footer className="flex gap-2 mt-2">
            <a href={s.url} target="_blank" className="btn">Open</a>
            <button className="btn btn-ghost" data-copy={s.url}>Copy Link</button>
          </footer>
        </article>
      ))}
      <script dangerouslySetInnerHTML={{__html: `
        (async function(){
          try{ const res = await fetch('${process.env.NEXT_PUBLIC_SERVICES_API_BASE || '/api/services'}/status?mode=auto');
                if(!res.ok) return; const st = await res.json();
                const map = Object.fromEntries(st.map(s => [s.id, s.status]));
                document.querySelectorAll('[data-status]').forEach(el => {
                  const id = el.getAttribute('data-status');
                  const k = map[id] || 'unknown';
                  el.textContent = k === 'ok' ? 'Reachable' : k === 'err' ? 'Unreachable' : k === 'warn' ? 'Slow' : 'Unknown';
                });
          }catch(e){}
        })();
      `}} />
    </main>
  );
}
```

Notes
- Status endpoint runs server-side or client-side; prefer client-side refresh every 30s.
- Links may require SSO; use Cloudflare Access apps/tokens on the API side for status checks only. User navigation follows normal Access flows.
- If you need per-service shortcuts in the command palette, build them from the `listServices()` data.

