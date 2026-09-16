/* Keep HTML pages from different GitHub Pages deployments in sync. */
(() => {
  const current = document.querySelector('meta[name="site-version"]')?.content;

  if (!current || current === "local" || location.protocol === "file:") return;

  const marker = new URL("/site-version.json", location.origin);
  marker.searchParams.set("check", Date.now().toString());

  fetch(marker, { cache: "no-store" })
    .then((response) => response.ok ? response.json() : null)
    .then((latest) => {
      if (!latest?.version || latest.version === current) return;

      const page = new URL(location.href);
      if (page.searchParams.get("site-version") === latest.version) return;

      page.searchParams.set("site-version", latest.version);
      location.replace(page);
    })
    .catch(() => {});
})();
