/**
 * art-rium shared service-worker logic.
 *
 * Each tool's sw.js is reduced to:
 *
 *   importScripts('/shared/sw-base.js');
 *   artRiumSetupSw({ cache: 'tool-name-v1', shell: ['/tools/x/', ...] });
 *
 * Optional `excludeFromCache` lets the dashboard sw skip /tools/* requests.
 */
self.artRiumSetupSw = ({ cache, shell, excludeFromCache = [] }) => {
  self.addEventListener('install', e => {
    e.waitUntil(
      caches.open(cache).then(c => c.addAll(shell)).then(() => self.skipWaiting())
    );
  });

  self.addEventListener('activate', e => {
    e.waitUntil(
      caches.keys().then(keys =>
        Promise.all(keys.filter(k => k !== cache).map(k => caches.delete(k)))
      ).then(() => self.clients.claim())
    );
  });

  /** Cache a successful GET response without holding up the reply to the page. */
  const store = (request, res) => {
    if (res.ok && request.method === 'GET') {
      const clone = res.clone();
      caches.open(cache).then(c => c.put(request, clone));
    }
    return res;
  };

  self.addEventListener('fetch', e => {
    const url = e.request.url;
    // Never intercept API calls, WebSocket upgrades, or per-sw exclusions.
    if (url.includes('/api/') || url.includes('/ws/')) return;
    if (excludeFromCache.some(prefix => url.includes(prefix))) return;

    // A tool's HTML *is* the tool: markup, styles and logic ship in one file.
    // Answering it from cache first pins the whole tool to whichever build the
    // browser happened to see first — a new feature then exists on one device
    // and is missing on another, with no way for the server to correct it
    // (its no-store headers never reach a request the sw already answered).
    // So HTML goes to the network first and only falls back to the cached copy
    // when there is no network, which is what the offline shell is for.
    const wantsHtml = e.request.mode === 'navigate' ||
      (e.request.headers.get('accept') || '').includes('text/html');

    if (wantsHtml) {
      e.respondWith(
        fetch(e.request)
          .then(res => store(e.request, res))
          .catch(() => caches.match(e.request).then(c => c || caches.match(shell[0])))
      );
      return;
    }

    // Everything else (icons, manifest) is versioned by the cache name and
    // safe to serve instantly.
    e.respondWith(
      caches.match(e.request).then(cached =>
        cached || fetch(e.request).then(res => store(e.request, res))
      )
    );
  });
};
