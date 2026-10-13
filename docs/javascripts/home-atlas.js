/* A small, dependency-free point cloud for the personal homepage. */
(() => {
  const atlas = document.querySelector('[data-research-atlas]');
  if (!atlas) return;
  const canvas = atlas.querySelector('canvas');
  const context = canvas.getContext('2d');
  if (!context) return; // Keep the SVG visible if canvas is unavailable.
  const stage = atlas.querySelector('.atlas-stage');
  const interaction = atlas.closest('.home-hero');
  const button = atlas.querySelector('.atlas-motion');
  const label = button.querySelector('[data-motion-label]');
  const icon = button.querySelector('.atlas-motion-icon');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const finePointer = matchMedia('(hover: hover) and (pointer: fine)');
  const keywords = ['VLA', 'ACT', 'Diffusion', 'Point Cloud', 'Attention', 'Drake', 'SE(3)', 'Embodiment', 'Imitation', 'Planning', 'Control', 'Representation'];
  const words = [
    { text: '3D PERCEPTION', x: -.9, y: -.52, z: .72 },
    { text: 'POLICY LEARNING', x: .55, y: .48, z: .95 },
    { text: 'ROBOTICS', x: .3, y: -.92, z: .55 },
    { text: 'Diffusion Policy', x: -1.12, y: .76, z: .12 },
    { text: 'OpenVLA', x: 1.15, y: -.54, z: .2 },
    { text: 'ALOHA', x: 1.18, y: .78, z: -.1 },
    { text: 'SE(3)', x: -.9, y: -.9, z: -.4 },
    { text: 'Attention', x: .65, y: 1.18, z: .05 },
    { text: 'World Model', x: -1.2, y: -.03, z: -.25 },
  ];
  let width = 0, height = 0, radius = 0, points = [];
  let yaw = .34, pitch = -.22, targetYaw = .34, targetPitch = -.22;
  let orbit = 0, frame = 0, previous = 0, visible = true, paused = false;
  let colors = [];
  let seed = 1412;
  const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);

  function palette() {
    const dark = document.body.dataset.mdColorScheme === 'slate';
    colors = dark ? ['128,170,255', '161,142,255', '107,207,220'] : ['37,101,208', '123,91,200', '39,146,166'];
  }

  function resize() {
    const rect = stage.getBoundingClientRect();
    width = rect.width; height = rect.height;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    radius = Math.min(width * .3, height * .34);
    seed = 1412;
    const count = width < 430 ? 850 : 1550;
    const angle = Math.PI * (3 - Math.sqrt(5));
    points = Array.from({ length: count }, (_, i) => {
      const y = 1 - 2 * (i + .5) / count;
      const ring = Math.sqrt(1 - y * y);
      const a = angle * i;
      const r = .96 + random() * .09;
      return { x: Math.cos(a) * ring * r, y: y * r, z: Math.sin(a) * ring * r,
        color: i % 3, size: .5 + random() * .8, word: i % 43 === 0 ? keywords[Math.floor(random() * keywords.length)] : null };
    });
    draw();
  }

  function project(point) {
    const turn = yaw + Math.sin(orbit * .25) * .07;
    const sy = Math.sin(turn), cy = Math.cos(turn), sp = Math.sin(pitch), cp = Math.cos(pitch);
    const x = point.x * cy + point.z * sy;
    const depth = point.z * cy - point.x * sy;
    const y = point.y * cp - depth * sp;
    const z = point.y * sp + depth * cp;
    const perspective = 4.8 / (4.8 - z);
    return { x: width * .51 + x * radius * perspective,
      y: height * .5 + y * radius * perspective, z, scale: perspective };
  }

  function drawOrbit(tilt, phase, color) {
    context.beginPath();
    for (let i = 0; i <= 150; i++) {
      const angle = i / 150 * Math.PI * 2;
      const x = Math.cos(angle) * 1.38;
      const y = Math.sin(angle) * 1.38;
      const p = project({ x, y: y * Math.cos(tilt), z: y * Math.sin(tilt) });
      i ? context.lineTo(p.x, p.y) : context.moveTo(p.x, p.y);
    }
    context.strokeStyle = `rgba(${colors[color]},.2)`;
    context.lineWidth = .6;
    context.stroke();
    const a = orbit + phase;
    const p = project({ x: Math.cos(a) * 1.38, y: Math.sin(a) * 1.38 * Math.cos(tilt), z: Math.sin(a) * 1.38 * Math.sin(tilt) });
    context.beginPath(); context.arc(p.x, p.y, 3, 0, Math.PI * 2);
    context.fillStyle = `rgba(${colors[color]},.8)`; context.fill();
    context.beginPath(); context.arc(p.x, p.y, 6.5, 0, Math.PI * 2);
    context.strokeStyle = `rgba(${colors[color]},.25)`; context.stroke();
  }

  function draw() {
    if (!width || !height) return;
    context.clearRect(0, 0, width, height);
    drawOrbit(.28, .2, 0);
    drawOrbit(1.35, 2.5, 1);
    drawOrbit(-.9, 4.5, 2);
    const projected = points.map(p => ({ ...project(p), point: p })).sort((a, b) => a.z - b.z);
    for (const p of projected) {
      const opacity = .16 + (p.z + 1.1) / 2.2 * .63;
      context.fillStyle = `rgba(${colors[p.point.color]},${opacity})`;
      if (p.point.word) {
        context.font = `${Math.max(5.8, radius * .044 * p.scale)}px Consolas, monospace`;
        context.fillText(p.point.word, p.x, p.y);
      } else {
        context.beginPath(); context.arc(p.x, p.y, p.point.size * p.scale, 0, Math.PI * 2); context.fill();
      }
    }
    const dark = document.body.dataset.mdColorScheme === 'slate';
    for (const [i, word] of words.entries()) {
      const p = project(word);
      const size = Math.max(width < 430 ? 7 : 9, radius * (i < 3 ? .075 : .059));
      context.font = `${i < 3 ? 600 : 400} ${size}px Consolas, monospace`;
      context.textAlign = 'center';
      const measure = context.measureText(word.text);
      // A quiet backing keeps the foreground keywords legible over the cloud.
      context.fillStyle = dark ? 'rgba(22,22,23,.86)' : 'rgba(255,255,255,.88)';
      context.fillRect(p.x - measure.width / 2 - 4, p.y - size + 1, measure.width + 8, size + 4);
      context.fillStyle = i < 3 ? `rgba(${colors[i]},.96)` : (dark ? 'rgba(210,215,232,.82)' : 'rgba(71,79,103,.78)');
      context.fillText(word.text, p.x, p.y);
    }
    context.textAlign = 'start';
    atlas.dataset.ready = '';
  }

  function canAnimate() {
    return visible && !document.hidden && !paused && !reduced.matches && finePointer.matches;
  }
  function tick(time) {
    frame = 0;
    if (!canAnimate()) return;
    const elapsed = previous ? time - previous : 34;
    if (elapsed >= 32) {
      const delta = Math.min(elapsed, 80);
      const smooth = 1 - Math.exp(-delta / 150);
      yaw += (targetYaw - yaw) * smooth;
      pitch += (targetPitch - pitch) * smooth;
      orbit += delta * .00015;
      draw(); previous = time;
    }
    frame = requestAnimationFrame(tick);
  }
  function sync() {
    cancelAnimationFrame(frame); frame = 0; previous = 0;
    button.hidden = reduced.matches || !finePointer.matches;
    button.setAttribute('aria-pressed', String(paused));
    button.setAttribute('aria-label', paused ? '继续研究星图动画' : '暂停研究星图动画');
    label.textContent = paused ? '继续动画' : '暂停动画';
    icon.textContent = paused ? '▷' : 'Ⅱ';
    if (reduced.matches || !finePointer.matches) { yaw = targetYaw = .34; pitch = targetPitch = -.22; }
    draw();
    if (canAnimate()) frame = requestAnimationFrame(tick);
  }

  interaction.addEventListener('pointermove', event => {
    if (!canAnimate() || event.target.closest('button')) return;
    const rect = interaction.getBoundingClientRect();
    const x = Math.max(-1, Math.min(1, (event.clientX - rect.left) / rect.width * 2 - 1));
    const y = Math.max(-1, Math.min(1, (event.clientY - rect.top) / rect.height * 2 - 1));
    targetYaw = .34 + x * .48; targetPitch = -.22 - y * .3;
  });
  interaction.addEventListener('pointerleave', () => { targetYaw = .34; targetPitch = -.22; });
  button.addEventListener('click', () => { paused = !paused; sync(); });
  reduced.addEventListener('change', sync);
  finePointer.addEventListener('change', sync);
  document.addEventListener('visibilitychange', sync);
  new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; sync(); }, { threshold: .05 }).observe(atlas);
  new ResizeObserver(resize).observe(stage);
  new MutationObserver(() => { palette(); draw(); }).observe(document.body, { attributes: true, attributeFilter: ['data-md-color-scheme'] });
  palette(); resize(); sync();
})();
