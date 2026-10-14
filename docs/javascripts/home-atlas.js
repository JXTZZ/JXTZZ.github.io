/* A quiet orbital study: pointer-driven, with no perpetual animation or dependencies. */
(() => {
  const atlas = document.querySelector('[data-research-atlas]');
  if (!atlas) return;
  const canvas = atlas.querySelector('canvas');
  const context = canvas.getContext('2d');
  if (!context) return;
  const stage = atlas.querySelector('.atlas-stage');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const pointer = matchMedia('(hover: hover) and (pointer: fine)');
  const homeYaw = .36, homePitch = -.26;
  let width = 0, height = 0, radius = 0, points = [];
  let yaw = homeYaw, pitch = homePitch, targetYaw = homeYaw, targetPitch = homePitch;
  let frame = 0, previous = 0, visible = true, dark = false;
  let colors = [];

  function palette() {
    dark = document.body.dataset.mdColorScheme === 'slate';
    colors = dark ? ['132,172,255', '180,160,255', '109,202,226'] :
      ['66,114,214', '125,105,206', '72,153,192'];
  }
  function project(point) {
    const sy = Math.sin(yaw), cy = Math.cos(yaw), sp = Math.sin(pitch), cp = Math.cos(pitch);
    const x = point.x * cy + point.z * sy, depth = point.z * cy - point.x * sy;
    const y = point.y * cp - depth * sp, z = point.y * sp + depth * cp;
    const scale = 5 / (5 - z);
    return {x: width * .52 + x * radius * scale, y: height * .49 + y * radius * scale, z, scale};
  }
  function path(points, color, opacity, lineWidth = .6, dashed = false) {
    context.beginPath();
    points.forEach((p, i) => i ? context.lineTo(p.x, p.y) : context.moveTo(p.x, p.y));
    context.strokeStyle = 'rgba(' + color + ',' + opacity + ')';
    context.lineWidth = lineWidth;
    context.setLineDash(dashed ? [2, 5] : []);
    context.stroke(); context.setLineDash([]);
  }
  function orbit(tilt, phase, color, front) {
    const points = [];
    for (let i = 0; i <= 180; i++) {
      const a = i / 180 * Math.PI * 2;
      const p = project({x: Math.cos(a) * 1.34, y: Math.sin(a) * 1.34 * Math.cos(tilt), z: Math.sin(a) * 1.34 * Math.sin(tilt)});
      if ((p.z >= 0) === front) points.push(p);
      else if (points.length) {path(points, colors[color], front ? .46 : .13, front ? .8 : .6); points.length = 0;}
    }
    if (points.length) path(points, colors[color], front ? .46 : .13, front ? .8 : .6);
    if (!front) return;
    const p = project({x: Math.cos(phase) * 1.34, y: Math.sin(phase) * 1.34 * Math.cos(tilt), z: Math.sin(phase) * 1.34 * Math.sin(tilt)});
    const glow = context.createRadialGradient(p.x, p.y, 0, p.x, p.y, 14);
    glow.addColorStop(0, 'rgba(' + colors[color] + ',.25)'); glow.addColorStop(1, 'rgba(' + colors[color] + ',0)');
    context.fillStyle = glow; context.beginPath(); context.arc(p.x, p.y, 14, 0, Math.PI * 2); context.fill();
    context.fillStyle = 'rgba(' + colors[color] + ',.94)';
    context.beginPath(); context.arc(p.x, p.y, 3, 0, Math.PI * 2); context.fill();
    context.strokeStyle = dark ? '#d8e5ff' : '#fff'; context.lineWidth = 1.5; context.stroke();
  }
  function draw() {
    if (!width || !height) return;
    context.clearRect(0, 0, width, height);
    const cx = width * .52, cy = height * .49;
    // Grounding halo, translucent body and an off-centre specular highlight.
    const halo = context.createRadialGradient(cx, cy, radius * .5, cx, cy, radius * 1.7);
    halo.addColorStop(0, dark ? 'rgba(92,116,214,.15)' : 'rgba(118,151,225,.08)');
    halo.addColorStop(1, 'rgba(118,151,225,0)');
    context.fillStyle = halo; context.fillRect(0, 0, width, height);
    [[.42,.65,0],[1.27,2.5,1],[-.75,4.8,2]].forEach(a => orbit(...a, false));
    const body = context.createRadialGradient(cx - radius * .35, cy - radius * .45, 0, cx, cy, radius);
    body.addColorStop(0, dark ? 'rgba(142,161,231,.18)' : 'rgba(255,255,255,.8)');
    body.addColorStop(.55, dark ? 'rgba(64,83,154,.12)' : 'rgba(208,223,252,.17)');
    body.addColorStop(1, dark ? 'rgba(67,92,175,.22)' : 'rgba(138,165,227,.15)');
    context.fillStyle = body; context.beginPath(); context.arc(cx, cy, radius, 0, Math.PI * 2); context.fill();
    context.strokeStyle = dark ? 'rgba(165,189,249,.28)' : 'rgba(115,146,209,.23)';
    context.lineWidth = .7; context.stroke();
    // Sparse latitude ribbons give the cloud shape without a wall of tiny labels.
    for (const latitude of [-.6, -.25, .2, .6]) {
      const ring = Math.sqrt(1 - latitude * latitude), points = [];
      for (let i = 0; i <= 120; i++) {
        const a = i / 120 * Math.PI * 2;
        points.push(project({x: Math.cos(a) * ring, y: latitude, z: Math.sin(a) * ring}));
      }
      path(points, colors[0], dark ? .11 : .1, .5, true);
    }
    const projected = points.map(p => ({...project(p), point: p})).sort((a, b) => a.z - b.z);
    for (const p of projected) {
      const opacity = .12 + Math.max(0, (p.z + 1) / 2) * (dark ? .68 : .57);
      context.fillStyle = 'rgba(' + colors[p.point.color] + ',' + opacity + ')';
      context.beginPath(); context.arc(p.x, p.y, p.point.size * p.scale, 0, Math.PI * 2); context.fill();
    }
    [[.42,.65,0],[1.27,2.5,1],[-.75,4.8,2]].forEach(a => orbit(...a, true));
    context.beginPath(); context.arc(cx, cy, radius, 3.8, 5.6);
    context.strokeStyle = dark ? 'rgba(218,229,255,.5)' : 'rgba(255,255,255,.96)';
    context.lineWidth = 1.4; context.stroke();
    atlas.dataset.ready = '';
  }
  function resize() {
    const rect = stage.getBoundingClientRect();
    width = rect.width; height = rect.height;
    const ratio = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    radius = Math.min(width * .3, height * .36);
    const count = width < 430 ? 280 : 640, angle = Math.PI * (3 - Math.sqrt(5));
    points = Array.from({length: count}, (_, i) => {
      const y = 1 - 2 * (i + .5) / count, ring = Math.sqrt(1 - y * y), a = angle * i;
      return {x: Math.cos(a) * ring, y, z: Math.sin(a) * ring, color: i % 3, size: .6 + (i % 7) * .13};
    });
    draw();
  }
  const interactive = () => visible && !document.hidden && !reduced.matches && pointer.matches;
  function tick(time) {
    frame = 0;
    if (!interactive()) return;
    const elapsed = previous ? Math.min(time - previous, 60) : 16;
    previous = time;
    const smooth = 1 - Math.exp(-elapsed / 95);
    yaw += (targetYaw - yaw) * smooth; pitch += (targetPitch - pitch) * smooth;
    const settling = Math.abs(targetYaw - yaw) + Math.abs(targetPitch - pitch) > .0002;
    if (!settling) {yaw = targetYaw; pitch = targetPitch; previous = 0;}
    draw();
    if (settling) frame = requestAnimationFrame(tick);
  }
  function start() {
    if (interactive() && !frame) frame = requestAnimationFrame(tick);
  }
  function reset() {
    cancelAnimationFrame(frame); frame = 0; previous = 0;
    yaw = targetYaw = homeYaw; pitch = targetPitch = homePitch; draw();
  }
  stage.addEventListener('pointermove', event => {
    if (!interactive()) return;
    const rect = stage.getBoundingClientRect();
    targetYaw = homeYaw + ((event.clientX - rect.left) / rect.width * 2 - 1) * .3;
    targetPitch = homePitch - ((event.clientY - rect.top) / rect.height * 2 - 1) * .2;
    start();
  });
  stage.addEventListener('pointerleave', () => {targetYaw = homeYaw; targetPitch = homePitch; start();});
  reduced.addEventListener('change', reset); pointer.addEventListener('change', reset);
  document.addEventListener('visibilitychange', reset);
  new IntersectionObserver(([entry]) => {visible = entry.isIntersecting; if (!visible) reset();}, {threshold: .05}).observe(atlas);
  new ResizeObserver(resize).observe(stage);
  new MutationObserver(() => {palette(); draw();}).observe(document.body, {attributes: true, attributeFilter: ['data-md-color-scheme']});
  palette(); resize();
})();
