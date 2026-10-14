/* Research words gather into LoTusY. Pointer-driven only; no perpetual animation. */
(() => {
  const atlas = document.querySelector('[data-research-atlas]');
  if (!atlas) return;
  const canvas = atlas.querySelector('canvas'), context = canvas.getContext('2d');
  if (!context) return;
  const stage = atlas.querySelector('.atlas-stage');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const keywords = ['OpenVLA', 'DP3', 'SPOT', 'Diffusion', 'Drake', 'ACT', 'SE(3)', 'ALOHA', 'ORION', 'LIT', '3D', 'Control', 'Planning', 'RoboTok', 'Octo'];
  const mask = document.createElement('canvas'), ink = mask.getContext('2d', {willReadFrequently: true});
  if (!ink) return;
  const mw = 720, mh = 240;
  mask.width = mw; mask.height = mh;
  ink.font = '900 184px Arial, sans-serif';
  const logoSize = Math.min(184, 184 * mw * .96 / ink.measureText('LoTusY').width);
  ink.font = '900 ' + logoSize + 'px Arial, sans-serif';
  ink.textAlign = 'center'; ink.fillStyle = '#fff';
  ink.fillText('LoTusY', mw / 2, mh * .73);
  const pixels = ink.getImageData(0, 0, mw, mh).data;
  const inside = (x, y) => x >= 0 && x < mw && y >= 0 && y < mh && pixels[(Math.floor(y) * mw + Math.floor(x)) * 4 + 3] > 140;
  let seed = 1412;
  const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  const tiles = [];
  ink.font = '6.4px Consolas, monospace';
  for (let y = 24, row = 0; y < mh - 24; y += 8, row++) {
    for (let x = 8; x < mw - 8;) {
      const text = keywords[(tiles.length + row) % keywords.length], size = ink.measureText(text).width;
      if (inside(x, y - 2) && inside(x + size / 2, y - 2) && inside(x + size, y - 2)) {
        tiles.push({text, x: x + size / 2, y, size: 6.4, color: tiles.length % 3,
          cloudX: .07 + random() * .86, cloudY: .09 + random() * .82, depth: random(), phase: random() * Math.PI * 2});
        x += size + 2;
      } else x += 3;
    }
  }
  const words = [
    {text:'OpenVLA',x:.18,y:.14,color:0}, {text:'Diffusion',x:.74,y:.17,color:1},
    {text:'SE(3)',x:.9,y:.42,color:2}, {text:'Drake',x:.13,y:.74,color:0},
    {text:'SPOT',x:.77,y:.84,color:1}, {text:'3D Perception',x:.43,y:.91,color:2}
  ];
  let width = 0, height = 0, frame = 0, previous = 0, visible = true, dark = false;
  let morph = 1, targetMorph = 1, tiltX = 0, tiltY = 0, targetX = 0, targetY = 0, drag = null;
  let colors = [];
  const clamp = value => Math.max(0, Math.min(1, value));
  const interactive = () => visible && !document.hidden && !reduced.matches;

  function palette() {
    dark = document.body.dataset.mdColorScheme === 'slate';
    colors = dark ? ['132,172,255','190,155,255','105,203,226'] : ['40,101,200','131,85,187','43,133,165'];
  }
  function draw() {
    if (!width || !height) return;
    context.clearRect(0, 0, width, height);
    const scale = width * .92 / mw, cx = width / 2, cy = height * .5;
    const halo = context.createRadialGradient(cx, cy, 0, cx, cy, width * .6);
    halo.addColorStop(0, dark ? 'rgba(108,105,209,.12)' : 'rgba(142,148,231,.07)');
    halo.addColorStop(1, 'rgba(120,140,200,0)');
    context.fillStyle = halo; context.fillRect(0, 0, width, height);
    // Thin flowing connections behind the word mosaic.
    for (let band = 0; band < 2; band++) {
      context.beginPath();
      for (let i = 0; i <= 80; i++) {
        const x = width * (.05 + i / 80 * .9);
        const y = cy + Math.sin(i / 80 * Math.PI * 2 + band * 2.4 + tiltX) * height * .23;
        i ? context.lineTo(x, y) : context.moveTo(x, y);
      }
      context.strokeStyle = 'rgba(' + colors[band] + ',.1)'; context.lineWidth = .6; context.stroke();
    }
    // A faint stencil keeps the overall name readable even on small screens.
    context.font = '900 ' + logoSize * scale + 'px Arial, sans-serif';
    context.textAlign = 'center';
    context.fillStyle = 'rgba(' + colors[0] + ',' + morph * (dark ? .045 : .035) + ')';
    context.fillText('LoTusY', cx + tiltX * 5, cy + (mh * .73 - mh / 2) * scale);
    for (const tile of tiles) {
      const target = {x: cx + (tile.x - mw / 2) * scale, y: cy + (tile.y - mh / 2) * scale};
      const x = (width * tile.cloudX) * (1 - morph) + target.x * morph + tiltX * (4 + tile.depth * 10);
      const y = (height * tile.cloudY) * (1 - morph) + target.y * morph + tiltY * (3 + tile.depth * 7);
      const size = tile.size * scale * morph + (width < 430 ? 6 : 8 + tile.depth * 3) * (1 - morph);
      context.font = '500 ' + size + 'px Consolas, monospace';
      context.fillStyle = 'rgba(' + colors[tile.color] + ',' + (.35 + morph * .53 + tile.depth * .1) + ')';
      context.fillText(tile.text, x, y);
    }
    for (const [i, word] of words.entries()) {
      if (width < 430 && i > 3) continue;
      const size = width < 430 ? 10 : 12 + (i % 2);
      const x = width * word.x + tiltX * (i + 2), y = height * word.y + tiltY * (i + 2);
      context.font = '500 ' + size + 'px Consolas, monospace';
      context.fillStyle = 'rgba(' + colors[word.color] + ',' + (dark ? .83 : .75) + ')';
      context.fillText(word.text, x, y);
    }
    context.textAlign = 'start';
    atlas.dataset.ready = ''; atlas.dataset.word = 'LoTusY';
    atlas.dataset.formation = morph.toFixed(3);
  }
  function resize() {
    const rect = stage.getBoundingClientRect(), ratio = Math.min(devicePixelRatio || 1, 2);
    width = rect.width; height = rect.height;
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0); draw();
  }
  function tick(time) {
    frame = 0;
    if (!interactive()) return;
    const elapsed = previous ? Math.min(time - previous, 60) : 16, smooth = 1 - Math.exp(-elapsed / 110);
    previous = time;
    morph += (targetMorph - morph) * smooth;
    tiltX += (targetX - tiltX) * smooth; tiltY += (targetY - tiltY) * smooth;
    const moving = Math.abs(targetMorph - morph) + Math.abs(targetX - tiltX) + Math.abs(targetY - tiltY) > .0005;
    if (!moving) {morph = targetMorph; tiltX = targetX; tiltY = targetY; previous = 0;}
    draw(); if (moving) frame = requestAnimationFrame(tick);
  }
  function start() {if (interactive() && !frame) frame = requestAnimationFrame(tick);}
  function reset() {
    cancelAnimationFrame(frame); frame = 0; previous = 0; drag = null;
    morph = targetMorph = 1; tiltX = tiltY = targetX = targetY = 0; draw();
  }
  stage.addEventListener('pointerdown', event => {
    if (!interactive() || event.pointerType === 'mouse') return;
    drag = {id:event.pointerId,x:event.clientX,y:event.clientY,morph};
  });
  stage.addEventListener('pointermove', event => {
    if (!interactive()) return;
    const rect = stage.getBoundingClientRect();
    if (event.pointerType === 'mouse') {
      const x = clamp((event.clientX - rect.left) / rect.width), y = clamp((event.clientY - rect.top) / rect.height);
      targetMorph = x; targetX = x * 2 - 1; targetY = y * 2 - 1; start();
    } else if (drag && drag.id === event.pointerId) {
      const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
      // Keep vertical page scrolling native. Only an intentional horizontal drag
      // changes the word formation; taps do not animate it.
      if (Math.abs(dx) < 8 || Math.abs(dy) > Math.abs(dx)) return;
      stage.setPointerCapture(event.pointerId);
      targetMorph = clamp(drag.morph + dx / rect.width * 1.6); start();
    }
  });
  stage.addEventListener('pointerleave', event => {
    if (event.pointerType !== 'mouse') return;
    targetMorph = 1; targetX = targetY = 0; start();
  });
  const release = event => {if (drag?.id === event.pointerId) drag = null;};
  stage.addEventListener('pointerup', release); stage.addEventListener('pointercancel', release);
  reduced.addEventListener('change', reset); document.addEventListener('visibilitychange', reset);
  new IntersectionObserver(([entry]) => {visible = entry.isIntersecting; if (!visible) reset();}, {threshold:.05}).observe(atlas);
  new ResizeObserver(resize).observe(stage);
  new MutationObserver(() => {palette(); draw();}).observe(document.body, {attributes:true,attributeFilter:['data-md-color-scheme']});
  palette(); resize();
})();
