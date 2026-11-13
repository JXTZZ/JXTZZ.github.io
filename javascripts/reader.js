/* Native details works without JS; this adds hash reveal and current-position hints. */
(() => {
  const toc = document.querySelector('.mobile-toc');
  const reveal = () => {
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
    const target = document.getElementById(id);
    if (!target) return;
    for (let parent = target.parentElement; parent; parent = parent.parentElement) {
      if (parent.tagName === 'DETAILS') { parent.open = true; parent.dataset.anchorReveal = ''; }
    }
  };
  reveal();
  addEventListener('hashchange', reveal);
  toc?.addEventListener('click', event => {
    const link = event.target.closest('a');
    if (!link) return;
    toc.open = false;
    const target = document.getElementById(decodeURIComponent(link.hash.slice(1)));
    if (target) {
      target.tabIndex = -1;
      requestAnimationFrame(() => target.focus({ preventScroll: true }));
    }
  });
  if (toc && 'IntersectionObserver' in window) {
    const links = [...toc.querySelectorAll('a')];
    const headings = links.map(a => document.getElementById(decodeURIComponent(a.hash.slice(1)))).filter(Boolean);
    const active = new Set();
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) entry.isIntersecting ? active.add(entry.target) : active.delete(entry.target);
      const current = [...active].sort((a, b) => a.getBoundingClientRect().top - b.getBoundingClientRect().top)[0];
      if (!current) return;
      for (const link of links) {
        if (decodeURIComponent(link.hash.slice(1)) === current.id) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      }
    }, { rootMargin: '-80px 0px -55% 0px' });
    headings.forEach(heading => observer.observe(heading));
  }
  // Material creates its table scroll wrapper. Make that one horizontal region
  // keyboard-focusable instead of adding a second scroll container.
  requestAnimationFrame(() => {
    document.querySelectorAll('article table').forEach(table => {
      const region = table.closest('.md-typeset__scrollwrap') || table.parentElement;
      region.tabIndex = 0;
      region.setAttribute('role', 'region');
      region.setAttribute('aria-label', '数据表，可横向滚动');
    });
  });
})();
