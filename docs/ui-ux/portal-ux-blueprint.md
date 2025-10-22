---
title: Portal UX Blueprint
description: Terminal-inspired, data-dense portal UX guided by Bloomberg-like patterns with accessible, modern ergonomics.
---

# Portal UX Blueprint

This blueprint translates a terminal-like, data-dense aesthetic (inspired by Bloomberg Terminal) into a modern, accessible web portal. It emphasizes speed (keyboard-first), clarity at scale (dense but readable), and actionable context (watchlists, alerts, drill downs).

References informing this direction
- Bloomberg/Terminal patterns: dense information, watchlists, tickers, keyboard commands
- Industry/Research catalogs: curated drill-ins by sector/topic with clear scoping
- Desktop Neo/Tile concepts: flexible, composable card grid for dashboard surfaces

Objectives
1) Speed: keyboard-first navigation and command execution
2) Signal over noise: highlight anomalies, trends, tasks
3) Consistency: predictable layouts and interactions across modules
4) Shareability: URL-synced state for views, filters, and time ranges
5) Accessibility: WCAG AA, full keyboard support, readable at density

Personas & Top Tasks
- Operator: monitor health, triage alerts, perform quick actions
- Analyst: explore data, compare entities, build/save views
- Executive: scan KPIs, understand changes, request follow-ups

Information Architecture
- Global: Top bar (brand, environment badge, search/palette, user), Left dock (primary modules), Ticker strip (optional), Content area (cards, tables, charts), Right drawer (details/context), Command palette overlay
- Primary Modules (example): Overview, Entities, Datasets, Analytics, Alerts, Settings
- Secondary Spaces: Saved Views, Watchlists, Workbench (scratch pad), Admin

Navigation Patterns
- Command palette (Cmd/Ctrl+K) with actions, views, entities, and help
- Breadcrumbs reflect module → collection → entity
- Left dock uses short labels + icons; hover prefetch likely targets
- Search supports filters (e.g., type:entity tag:finance)

Layout Primitives
- App shell: top bar, left dock, content grid, optional right drawer
- Cards: header (title, actions), body (content), footer (context/links)
- Side drawer: entity summary + tabs (Overview, Activity, Related)
- Table pane: saved views, column presets, sticky headers, bulk actions
- KPI rail: compact, color-coded tiles with spark lines and delta

Components (Core)
- Watchlist: pin entities; color-coded status; quick actions
- Ticker strip: horizontally scrolling signals (alerts, events, metrics)
- Table: resize/reorder/visibility; URL-synced filters; quick filter chips
- Chart tiles: time series, bar, heatmap; compare ranges; hover insights
- Alerts: rules + noise controls; snooze; assign; audit trail
- Search/Palette: fast fuzzy + scoped filters; recent; tips
- Service Directory: discover services from cluster ingress; link out with status; SSO-aware

States & Feedback
- Loading: skeletons (not spinners), optimistic updates for low-risk actions
- Empty: guidance + primary CTA to populate (import, connect, add)
- Errors: clear cause + retry; link to diagnostics/details
- Destructive: toast with Undo (10–20s); audit for all critical actions

Accessibility
- Keyboard: Tab order, skip link, focus ring, palette (Cmd/Ctrl+K), help (?), global shortcuts for common actions (s, / search; g then e for Entities)
- Contrast: adhere to WCAG AA; provide reduced-motion preference
- Labels: aria-labels/roles; table headers, sortable controls, live regions

Visual Tokens (Dark-first)
- Background: #0B0D10; Surface: #12141A; Border: #1C2230
- Text: #E6E8EB; Muted: #8A9099; Line: #2A3242
- Accent: #4DB5FF; Positive: #6EF77A; Negative: #FF6B6B; Warning: #F7C948; Info: #9B5DE5
- Radius: 8px; Spacing: 4, 8, 12, 16, 24; Elevation: 0, 2, 6, 12

Copy & Semantics
- Titles are nouns; actions are verbs; avoid jargon
- Deltas use sign (+/−) and color; always include text labels for colorblind users
- Time ranges are explicit (UTC/Local); show compare context when applied

Page Templates
1) Overview
   - Top KPIs (5–7) with deltas; last updated; quick time range selector
   - Ticker strip for noteworthy changes; filter by severity/type
   - Card grid: key lists (alerts, recent activity, saved views)
2) Entities Index
   - Facets: industry, status, tags; quick filters; save view
   - Table with presets (Minimal, Monitoring, Analysis)
   - Right drawer for selected entity details; keyboard open/close
3) Entity Detail
   - Header with status, key metrics, actions (watch, share)
   - Tabs: Overview, Metrics, Related, Activity; persistent sub-filters
4) Analytics
   - Workspace with panel layout; save/share query presets
   - Chart tiles; compare ranges; annotations and notes

Keyboard Shortcuts (sample)
- Cmd/Ctrl+K: Command palette
- / or s: Focus search
- g then o/e/d/a: Go to Overview/Entities/Datasets/Analytics
- ?: Shortcut overlay
- j/k or ↑/↓: Navigate lists; Enter: open; x: select

Performance UX
- Route-based code splitting; lazy-load heavy charts
- Prefetch likely next routes on hover/focus
- Debounce search/filters; keep-alive for hot endpoints
- Preserve form and scroll state on navigation

Instrumentation
- Track usage funnels for top tasks; rage clicks; empty/error rates
- Capture client errors (Sentry); measure Core Web Vitals
- Feature flags for gradual rollout of nav and critical patterns

Roadmap (Phased)
Phase 1 (2 sprints)
- Command palette + search; environment badge; skeleton loaders
- Table saved views with URL-synced filters; empty/error states
- Watchlist + ticker prototype; keyboard overlay
Phase 2 (2–3 sprints)
- Side drawer; entity detail template; alerts workflow
- KPI rail with compare ranges; chart tiles; annotations
- Accessibility hardening + reduced-motion variants
Phase 3
- Analytics workspace; shareable notebooks/presets
- Feature flags + A/B for nav copy and grouping

Deliverables in this repo
- docs/ui-ux/portal-ux-blueprint.md (this document)
- docs/ui-ux/keyboard-shortcuts.md (detailed map)
- prototype/terminal/ (static HTML/CSS/JS demo of shell, ticker, table, and palette)
  - services.json (service registry consumed by the directory)
  - Use scripts/generate-service-registry.sh to build services.json from k8s/ingress/ingress-rules.yaml
5) Services
   - Filterable card grid from a registry (JSON or API)
   - Actions: Open, Copy Link, Check (ping), Favorite
   - Status from best-effort pings or health endpoints; SSO-aware

