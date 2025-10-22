(() => {
  const qs = (s, el=document) => el.querySelector(s);
  const qsa = (s, el=document) => Array.from(el.querySelectorAll(s));

  const palette = qs('#palette');
  const help = qs('#help');
  const paletteInput = qs('#palette-input');
  const paletteList = qs('#palette-list');
  const search = qs('#global-search');
  const openPaletteBtn = qs('#open-palette');
  const openHelpBtn = qs('#open-help');
  const views = {
    overview: qs('#view-overview'),
    services: qs('#view-services'),
  };
  const navItems = qsa('.sidebar .nav-item');
  const serviceGrid = qs('#service-grid');
  const serviceFilter = qs('#service-filter');
  let services = [];

  function navigate(view) {
    Object.values(views).forEach(v => v && v.classList.remove('active'));
    if (views[view]) views[view].classList.add('active');
    navItems.forEach(a => {
      if (a.dataset.view === view) a.classList.add('active'); else a.classList.remove('active');
    });
    announce(`Navigate: ${view}`);
  }

  navItems.forEach(a => a.addEventListener('click', (e) => {
    const view = a.dataset.view;
    if (!view) return;
    e.preventDefault();
    navigate(view);
  }));

  // Load services.json and render directory
  async function loadServices() {
    try {
      // Prefer live API when available (same-origin), fallback to local JSON
      let res = await fetch('/api/services');
      if (!res.ok) throw new Error('API not available');
      services = await res.json();
    } catch (e) {
      try {
        const res2 = await fetch('services.json');
        services = await res2.json();
      } catch (err) {
        console.error('Failed to load services registry', err);
        services = [];
      }
    }
    renderServices(services);
    addServicesToPalette(services);
    // Trigger initial status load if API exists
    try { await loadStatus(); } catch {}
  }

  function renderServices(list) {
    if (!serviceGrid) return;
    serviceGrid.innerHTML = '';
    list.forEach(svc => {
      const card = document.createElement('div');
      card.className = 'service-card';
      card.dataset.id = svc.id;
      card.innerHTML = `
        <div class="service-head">
          <div class="service-name"><span>${svc.icon || '🔗'}</span> ${svc.name}</div>
          <div class="status unknown" data-status>Unknown</div>
        </div>
        <div class="service-desc">${svc.description || ''}</div>
        <div class="muted small">${svc.category || ''} • ${new URL(svc.url).host}</div>
        <div class="service-actions">
          <button class="btn small" data-open>Open</button>
          <button class="btn small ghost" data-copy>Copy Link</button>
          <button class="btn small ghost" data-check>Check</button>
        </div>
      `;
      const statusEl = card.querySelector('[data-status]');
      card.querySelector('[data-open]').addEventListener('click', () => window.open(svc.url, '_blank'));
      card.querySelector('[data-copy]').addEventListener('click', async () => {
        try { await navigator.clipboard.writeText(svc.url); announce('Link copied'); } catch {}
      });
      card.querySelector('[data-check]').addEventListener('click', () => pingService(svc, statusEl));
      serviceGrid.appendChild(card);
    });
  }

  function addServicesToPalette(list) {
    list.forEach(svc => {
      const li = document.createElement('li');
      li.dataset.action = `open:service:${svc.id}`;
      li.textContent = `Open: ${svc.name}`;
      paletteList.appendChild(li);
      li.addEventListener('click', () => runCommand(li.dataset.action));
    });
  }

  function pingService(svc, statusEl) {
    // Best-effort CORS-agnostic ping using image probe to favicon
    try {
      const url = new URL(svc.url);
      const img = new Image();
      const t = setTimeout(() => setStatus('warn'), 5000);
      img.onload = () => { clearTimeout(t); setStatus('ok'); };
      img.onerror = () => { clearTimeout(t); setStatus('err'); };
      img.src = `${url.origin}/favicon.ico?__t=${Date.now()}`;
    } catch { setStatus('unknown'); }

    function setStatus(kind) {
      statusEl.classList.remove('ok','warn','err','unknown');
      statusEl.classList.add(kind || 'unknown');
      statusEl.textContent = kind === 'ok' ? 'Reachable' : kind === 'err' ? 'Unreachable' : kind === 'warn' ? 'Slow' : 'Unknown';
    }
  }

  async function loadStatus(mode='auto') {
    const res = await fetch(`/api/services/status?mode=${encodeURIComponent(mode)}`);
    if (!res.ok) return;
    const statuses = await res.json();
    statuses.forEach(s => {
      const card = serviceGrid && serviceGrid.querySelector(`.service-card[data-id="${s.id}"]`);
      if (!card) return;
      const statusEl = card.querySelector('[data-status]');
      setStatus(statusEl, s.status);
    });
  }

  function setStatus(el, kind) {
    if (!el) return;
    el.classList.remove('ok','warn','err','unknown');
    el.classList.add(kind || 'unknown');
    el.textContent = kind === 'ok' ? 'Reachable' : kind === 'err' ? 'Unreachable' : kind === 'warn' ? 'Slow' : 'Unknown';
  }

  if (serviceFilter) {
    serviceFilter.addEventListener('input', () => {
      const q = serviceFilter.value.trim().toLowerCase();
      const filtered = services.filter(s =>
        [s.name, s.category, s.description, s.url].filter(Boolean).some(v => v.toLowerCase().includes(q))
      );
      renderServices(filtered);
    });
  }

  function openOverlay(el) {
    el.classList.remove('hidden');
    // Delay focus to allow paint
    setTimeout(() => {
      const input = el.querySelector('input');
      if (input) input.focus();
    }, 0);
  }
  function closeOverlay(el) { el.classList.add('hidden'); }

  function togglePalette() {
    if (palette.classList.contains('hidden')) openOverlay(palette); else closeOverlay(palette);
  }
  function toggleHelp() {
    if (help.classList.contains('hidden')) openOverlay(help); else closeOverlay(help);
  }

  // Filter palette items by input value
  function filterPalette() {
    const query = paletteInput.value.trim().toLowerCase();
    qsa('li', paletteList).forEach(li => {
      const text = li.textContent.toLowerCase();
      li.style.display = text.includes(query) ? '' : 'none';
    });
    const firstVisible = qsa('li', paletteList).find(li => li.style.display !== 'none');
    qsa('li', paletteList).forEach(li => li.classList.remove('active'));
    if (firstVisible) firstVisible.classList.add('active');
  }

  // Keyboard handling
  function isInput(el) {
    return ['INPUT','TEXTAREA'].includes(el.tagName) || el.isContentEditable;
  }

  document.addEventListener('keydown', (e) => {
    const mod = e.ctrlKey || e.metaKey;
    // Go-to sequences (g then key)
    if (!isInput(document.activeElement) && e.key.toLowerCase() === 'g') {
      goMode = true; goTimer && clearTimeout(goTimer);
      goTimer = setTimeout(() => { goMode = false; }, 1500);
      return;
    }
    if (goMode && !mod) {
      const k = e.key.toLowerCase();
      const map = { o: 'overview', e: 'entities', s: 'services', d: 'datasets', a: 'analytics', l: 'alerts' };
      const view = map[k];
      if (view) { e.preventDefault(); navigate(view); goMode = false; return; }
    }
    if (mod && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      togglePalette();
      return;
    }
    if (!isInput(document.activeElement) && (e.key === '/' || e.key.toLowerCase() === 's')) {
      e.preventDefault();
      search.focus();
      return;
    }
    if (e.key === '?') {
      e.preventDefault();
      toggleHelp();
      return;
    }
    if (e.key === 'Escape') {
      [palette, help].forEach(closeOverlay);
      return;
    }
    // Palette navigation
    if (!palette.classList.contains('hidden')) {
      const items = qsa('li', paletteList).filter(li => li.style.display !== 'none');
      let idx = items.findIndex(li => li.classList.contains('active'));
      if (e.key === 'ArrowDown') { e.preventDefault(); items.forEach(li=>li.classList.remove('active')); items[Math.min(idx+1, items.length-1)].classList.add('active'); }
      if (e.key === 'ArrowUp') { e.preventDefault(); items.forEach(li=>li.classList.remove('active')); items[Math.max(idx-1, 0)].classList.add('active'); }
      if (e.key === 'Enter') { e.preventDefault(); const active = items[idx>=0?idx:0]; if (active) runCommand(active.dataset.action); }
    }
  });

  paletteInput.addEventListener('input', filterPalette);
  openPaletteBtn.addEventListener('click', togglePalette);
  openHelpBtn.addEventListener('click', toggleHelp);
  palette.addEventListener('click', (e) => { if (e.target === palette) closeOverlay(palette); });
  help.addEventListener('click', (e) => { if (e.target === help) closeOverlay(help); });
  qsa('#palette-list li').forEach(li => li.addEventListener('click', () => runCommand(li.dataset.action)));

  function runCommand(action) {
    switch (action) {
      case 'goto:overview':
        navigate('overview');
        break;
      case 'goto:services':
        navigate('services');
        break;
      case 'goto:entities':
      case 'goto:datasets':
      case 'goto:analytics':
      case 'goto:alerts':
        announce(`Navigate: ${action.split(':')[1]}`);
        break;
      default:
        if (action && action.startsWith('open:service:')) {
          const id = action.split(':')[2];
          const svc = services.find(s => s.id === id);
          if (svc) window.open(svc.url, '_blank');
          announce(`Open service: ${id}`);
          break;
        }
      case 'help':
        closeOverlay(palette); openOverlay(help); break;
      default:
        announce('Command not implemented');
    }
    closeOverlay(palette);
  }

  // Simple ARIA live region for announcements
  const live = document.createElement('div');
  live.setAttribute('aria-live', 'polite');
  live.setAttribute('aria-atomic', 'true');
  live.className = 'sr-only';
  document.body.appendChild(live);
  function announce(msg) { live.textContent = msg; }

  // Add a visually hidden class
  const style = document.createElement('style');
  style.textContent = `.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0;}`;
  document.head.appendChild(style);

  // go-to mode state
  let goMode = false;
  let goTimer = null;

  // Initialize
  navigate('overview');
  loadServices();
  // Periodic status refresh when Services view active
  setInterval(() => {
    if (views.services && views.services.classList.contains('active')) {
      loadStatus().catch(() => {});
    }
  }, 30000);
})();
