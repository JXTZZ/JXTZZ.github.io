/* Add keyboard access to Material's existing label controls; keep its state machine. */
(() => {
  const controls = [...document.querySelectorAll('.md-header__button[for]')];
  for (const control of controls) {
    const id = control.htmlFor;
    const input = document.getElementById(id);
    if (!input) continue;
    // Labels forward focus to their hidden checkbox in Chromium. A focusable
    // child keeps the native label click wiring and gives the icon a real stop.
    const focus = document.createElement('span');
    focus.className = 'header-control-focus';
    focus.tabIndex = 0;
    focus.setAttribute('role', 'button');
    focus.setAttribute('aria-label', control.title || (id === '__drawer' ? '打开全站导航' : '搜索笔记'));
    focus.append(...control.childNodes);
    control.append(focus);
    if (input.type === 'checkbox') {
      const sync = () => focus.setAttribute('aria-expanded', String(input.checked));
      sync();
      input.addEventListener('change', sync);
    }
    control.addEventListener('keydown', event => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      event.preventDefault();
      event.stopPropagation();
      control.click();
      if (id.startsWith('__palette')) requestAnimationFrame(() => {
        controls.find(label => label.htmlFor.startsWith('__palette') && !label.hidden)?.querySelector('.header-control-focus').focus();
      });
    });
  }
  document.addEventListener('keydown', event => {
    const drawer = document.getElementById('__drawer');
    if (event.key === 'Escape' && drawer?.checked) {
      drawer.click();
      controls.find(label => label.htmlFor === '__drawer')?.querySelector('.header-control-focus').focus();
    }
  });
})();
